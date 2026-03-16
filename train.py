#!/usr/bin/env python3
"""Train an SMP UNet on GL261 and evaluate on the validation set.

Architecture: UNet with any timm encoder (ImageNet pretrained), 1-channel input,
binary segmentation output.

Usage:
    python train.py
    python train.py --encoder efficientnet-b4 --epochs 100
    python train.py --eval-only --checkpoint checkpoints/best_model.pth
"""

import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm

import segmentation_models_pytorch as smp
import segmentation_models_pytorch.losses as smp_losses

from gl261.dataset import GL261SegDataset

ROOT = Path(__file__).resolve().parent

DATA_DIR = ROOT / "data" / "processed" / "nnunet" / "Dataset501_GL261"
TRAIN_IMG_DIR = DATA_DIR / "imagesTr"
TRAIN_MSK_DIR = DATA_DIR / "labelsTr"
VAL_IMG_DIR = DATA_DIR / "imagesTs"
VAL_MSK_DIR = DATA_DIR / "labelsTs"
OUT_DIR = ROOT / "checkpoints"


# ---------------------------------------------------------------------------
# Compatibility
# ---------------------------------------------------------------------------

def _to_numpy(t: torch.Tensor) -> np.ndarray:
    """Convert torch tensor to numpy array (works around broken np<->torch in 2.7+)."""
    return np.from_dlpack(t.detach().cpu())


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------

class BCEDiceLoss(nn.Module):
    """Combined BCE + Dice loss for binary segmentation."""

    def __init__(self, bce_weight: float = 0.5):
        super().__init__()
        self.bce_weight = bce_weight
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce_loss = self.bce(logits, targets)
        probs = torch.sigmoid(logits)
        smooth = 1e-6
        intersection = (probs * targets).sum(dim=(2, 3))
        union = probs.sum(dim=(2, 3)) + targets.sum(dim=(2, 3))
        dice = (2.0 * intersection + smooth) / (union + smooth)
        dice_loss = 1.0 - dice.mean()
        return self.bce_weight * bce_loss + (1 - self.bce_weight) * dice_loss


class FocalTverskyLoss(nn.Module):
    """Focal Tversky loss for small object segmentation."""

    def __init__(self, alpha=0.3, beta=0.7, gamma=0.75):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)
        smooth = 1e-6
        tp = (probs * targets).sum(dim=(2, 3))
        fp = (probs * (1 - targets)).sum(dim=(2, 3))
        fn = ((1 - probs) * targets).sum(dim=(2, 3))
        tversky = (tp + smooth) / (tp + self.alpha * fp + self.beta * fn + smooth)
        focal_tversky = (1 - tversky).pow(self.gamma)
        return focal_tversky.mean()


def build_loss(loss_name: str) -> nn.Module:
    """Build loss function by name."""
    if loss_name == "bce_dice":
        return BCEDiceLoss(bce_weight=0.5)
    elif loss_name == "focal_tversky":
        return FocalTverskyLoss(alpha=0.3, beta=0.7, gamma=0.75)
    elif loss_name == "lovasz":
        return smp_losses.LovaszLoss(mode="binary", from_logits=True)
    else:
        raise ValueError(f"Unknown loss: {loss_name}")


def build_model(arch: str, encoder: str, in_channels: int = 1) -> nn.Module:
    """Build SMP model by architecture name."""
    kwargs = dict(
        encoder_name=encoder,
        encoder_weights="imagenet",
        in_channels=in_channels,
        classes=1,
        activation=None,
    )
    if arch == "unet":
        return smp.Unet(**kwargs)
    elif arch == "unetplusplus":
        return smp.UnetPlusPlus(**kwargs)
    elif arch == "manet":
        return smp.MAnet(**kwargs)
    elif arch == "deeplabv3plus":
        return smp.DeepLabV3Plus(**kwargs)
    elif arch == "fpn":
        return smp.FPN(**kwargs)
    else:
        raise ValueError(f"Unknown arch: {arch}")


# ---------------------------------------------------------------------------
# CutMix
# ---------------------------------------------------------------------------

def cutmix_batch(imgs, masks, alpha=1.0):
    """Apply CutMix augmentation to a batch."""
    bs, c, h, w = imgs.shape
    indices = torch.randperm(bs)
    lam = np.random.beta(alpha, alpha)

    cut_ratio = np.sqrt(1.0 - lam)
    cut_h = int(h * cut_ratio)
    cut_w = int(w * cut_ratio)
    cy = np.random.randint(0, h)
    cx = np.random.randint(0, w)
    y1 = max(0, cy - cut_h // 2)
    y2 = min(h, cy + cut_h // 2)
    x1 = max(0, cx - cut_w // 2)
    x2 = min(w, cx + cut_w // 2)

    imgs_mixed = imgs.clone()
    masks_mixed = masks.clone()
    imgs_mixed[:, :, y1:y2, x1:x2] = imgs[indices, :, y1:y2, x1:x2]
    masks_mixed[:, :, y1:y2, x1:x2] = masks[indices, :, y1:y2, x1:x2]
    return imgs_mixed, masks_mixed


# ---------------------------------------------------------------------------
# Metrics (inline, for training loop monitoring)
# ---------------------------------------------------------------------------

def compute_metrics(pred: np.ndarray, gt: np.ndarray) -> dict:
    """Compute Dice, IoU, precision, recall for a single binary mask pair."""
    pred_b = pred > 0
    gt_b = gt > 0

    if not pred_b.any() and not gt_b.any():
        return {"dice": 1.0, "iou": 1.0, "precision": 1.0, "recall": 1.0}
    if not pred_b.any() or not gt_b.any():
        prec = 1.0 if (not gt_b.any() and not pred_b.any()) else 0.0
        rec = 1.0 if (not pred_b.any() and not gt_b.any()) else 0.0
        return {"dice": 0.0, "iou": 0.0, "precision": prec, "recall": rec}

    tp = np.logical_and(pred_b, gt_b).sum()
    fp = np.logical_and(pred_b, ~gt_b).sum()
    fn = np.logical_and(~pred_b, gt_b).sum()

    dice = 2.0 * tp / (2.0 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0.0
    iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    return {
        "dice": float(dice),
        "iou": float(iou),
        "precision": float(precision),
        "recall": float(recall),
    }


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_one_epoch(model, loader, criterion, optimizer, device,
                    use_cutmix=False, scaler=None):
    model.train()
    total_loss = 0.0
    total_dice = 0.0
    n = 0

    for imgs, masks, _ in loader:
        imgs = imgs.to(device)
        masks = masks.to(device)

        if use_cutmix and np.random.rand() < 0.5:
            imgs, masks = cutmix_batch(imgs, masks)

        optimizer.zero_grad()
        if scaler is not None:
            with torch.amp.autocast("cuda"):
                logits = model(imgs)
                loss = criterion(logits, masks)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(imgs)
            loss = criterion(logits, masks)
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            logits_f = logits.float()
            preds = (torch.sigmoid(logits_f) > 0.5).float()
            smooth = 1e-6
            inter = (preds * masks).sum(dim=(2, 3))
            union = preds.sum(dim=(2, 3)) + masks.sum(dim=(2, 3))
            batch_dice = ((2.0 * inter + smooth) / (union + smooth)).mean().item()

        total_loss += loss.item() * imgs.size(0)
        total_dice += batch_dice * imgs.size(0)
        n += imgs.size(0)

    return total_loss / n, total_dice / n


@torch.no_grad()
def validate(model, loader, criterion, device):
    """Returns (val_loss, val_dice, tumor_only_dice)."""
    model.eval()
    total_loss = 0.0
    total_dice = 0.0
    tumor_dice_sum = 0.0
    tumor_n = 0
    n = 0

    for imgs, masks, _ in loader:
        imgs = imgs.to(device)
        masks = masks.to(device)

        logits = model(imgs)
        loss = criterion(logits, masks)

        preds = (torch.sigmoid(logits) > 0.5).float()
        smooth = 1e-6
        inter = (preds * masks).sum(dim=(2, 3))
        union = preds.sum(dim=(2, 3)) + masks.sum(dim=(2, 3))
        per_sample_dice = (2.0 * inter + smooth) / (union + smooth)
        batch_dice = per_sample_dice.mean().item()

        has_tumor = masks.sum(dim=(2, 3)).squeeze(1) > 0
        if has_tumor.any():
            tumor_dice_sum += per_sample_dice.squeeze(1)[has_tumor].sum().item()
            tumor_n += has_tumor.sum().item()

        total_loss += loss.item() * imgs.size(0)
        total_dice += batch_dice * imgs.size(0)
        n += imgs.size(0)

    tumor_dice = tumor_dice_sum / tumor_n if tumor_n > 0 else 0.0
    return total_loss / n, total_dice / n, tumor_dice


def _set_seed(seed: int):
    """Set random seeds for reproducibility."""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    seed = getattr(args, "seed", None)
    if seed is not None:
        _set_seed(seed)
        print(f"Random seed: {seed}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    img_dir = Path(args.img_dir) if args.img_dir else TRAIN_IMG_DIR
    msk_dir = Path(args.msk_dir) if args.msk_dir else TRAIN_MSK_DIR

    train_ds = GL261SegDataset(
        img_dir, msk_dir,
        img_size=args.img_size, augment=True,
        aug_preset=args.aug_preset,
    )

    val_img_dir = Path(args.val_img_dir) if args.val_img_dir else VAL_IMG_DIR
    val_msk_dir = Path(args.val_msk_dir) if args.val_msk_dir else VAL_MSK_DIR
    val_ds = GL261SegDataset(
        val_img_dir, val_msk_dir,
        img_size=args.img_size, augment=False,
    )
    print(f"Train samples: {len(train_ds)}, Val samples: {len(val_ds)}")

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
    )

    model = build_model(args.arch, args.encoder)
    model = model.to(device)

    criterion = build_loss(args.loss)

    # Differential LR: encoder gets lr * encoder_lr_mult
    encoder_params = list(model.encoder.parameters())
    decoder_params = [p for p in model.parameters() if not any(p is ep for ep in encoder_params)]
    enc_lr = args.lr * args.encoder_lr_mult
    param_groups = [
        {"params": encoder_params, "lr": enc_lr},
        {"params": decoder_params, "lr": args.lr},
    ]
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr, weight_decay=args.weight_decay)

    warmup_epochs = args.warmup_epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, args.epochs - warmup_epochs), eta_min=1e-6,
    )

    use_cutmix = args.cutmix
    use_amp = not args.no_amp
    scaler = torch.amp.GradScaler("cuda") if (use_amp and device.type == "cuda") else None

    print(f"Optimizer: AdamW, decoder_lr={args.lr}, encoder_lr={enc_lr:.1e}, wd={args.weight_decay}")
    print(f"Arch: {args.arch}, Encoder: {args.encoder}, Loss: {args.loss}")
    print(f"CutMix: {use_cutmix}, Aug: {args.aug_preset}, AMP: {scaler is not None}")

    best_score = 0.0
    best_epoch = -1
    METRIC_EMA_DECAY = 0.9
    ema_score = 0.0

    print(f"\nTraining for {args.epochs} epochs...")
    print(f"{'Epoch':>5} | {'TrLoss':>8} | {'TrDice':>8} | "
          f"{'VLoss':>8} | {'VDice':>8} | {'VT-Dice':>8} | {'EMA':>8} | {'LR':>10} | {'Time':>5}")
    print("-" * 95)

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        if warmup_epochs > 0 and epoch <= warmup_epochs:
            warmup_factor = epoch / warmup_epochs
            for pg, base_lr in zip(optimizer.param_groups, [enc_lr, args.lr]):
                pg["lr"] = base_lr * warmup_factor

        train_loss, train_dice = train_one_epoch(
            model, train_loader, criterion, optimizer, device,
            use_cutmix=use_cutmix, scaler=scaler,
        )
        val_loss, val_dice, val_tumor_dice = validate(model, val_loader, criterion, device)

        if epoch > warmup_epochs:
            scheduler.step()

        elapsed = time.time() - t0
        lr = optimizer.param_groups[0]["lr"]

        raw_score = val_tumor_dice
        if epoch == 1:
            ema_score = raw_score
        else:
            ema_score = METRIC_EMA_DECAY * ema_score + (1 - METRIC_EMA_DECAY) * raw_score

        if epoch <= 5 or epoch % 10 == 0 or epoch == args.epochs:
            print(f"{epoch:5d} | {train_loss:8.4f} | {train_dice:8.4f} | "
                  f"{val_loss:8.4f} | {val_dice:8.4f} | {val_tumor_dice:8.4f} | "
                  f"{ema_score:8.4f} | {lr:10.6f} | {elapsed:4.1f}s")

        if ema_score > best_score:
            best_score = ema_score
            best_epoch = epoch
            ckpt_path = out_dir / "best_model.pth"
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_dice": val_dice,
                "val_tumor_dice": val_tumor_dice,
                "val_loss": val_loss,
                "ema_score": ema_score,
                "arch": args.arch,
                "encoder": args.encoder,
                "img_size": args.img_size,
                "seed": seed,
            }, ckpt_path)

    print(f"\nBest tumor Dice (EMA): {best_score:.4f} at epoch {best_epoch}")
    print(f"Checkpoint saved to {out_dir / 'best_model.pth'}")

    return model


# ---------------------------------------------------------------------------
# D4 TTA (8-fold dihedral group: 4 rotations x 2 flips)
# ---------------------------------------------------------------------------

def _d4_transforms():
    """Return list of (forward_fn, inverse_fn) for D4 TTA."""
    def rot_k(x, k):
        return torch.rot90(x, k, [2, 3])
    def flip_h(x):
        return torch.flip(x, [3])

    transforms = []
    for k in range(4):
        transforms.append(
            (lambda x, k=k: rot_k(x, k),
             lambda x, k=k: rot_k(x, -k % 4))
        )
        transforms.append(
            (lambda x, k=k: rot_k(flip_h(x), k),
             lambda x, k=k: flip_h(rot_k(x, -k % 4)))
        )
    return transforms


@torch.no_grad()
def _predict_tta_d4(model, imgs, device):
    """D4 TTA: average probabilities over 8 transforms."""
    transforms = _d4_transforms()
    prob_sum = torch.zeros_like(imgs[:, :1])
    for fwd, inv in transforms:
        augmented = fwd(imgs)
        logits = model(augmented)
        probs = torch.sigmoid(logits)
        prob_sum += inv(probs)
    return prob_sum / len(transforms)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_model(model, args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    out_dir = Path(args.out_dir)
    tta_mode = args.tta

    eval_img_dir = Path(args.val_img_dir) if args.val_img_dir else VAL_IMG_DIR
    eval_msk_dir = Path(args.val_msk_dir) if args.val_msk_dir else VAL_MSK_DIR

    pred_suffix = "predictions_tta" if tta_mode == "d4" else "predictions"
    pred_dir = out_dir / pred_suffix
    # Clear stale predictions to prevent cross-model contamination
    if pred_dir.exists():
        import shutil
        shutil.rmtree(pred_dir)
    pred_dir.mkdir(parents=True, exist_ok=True)

    val_ds = GL261SegDataset(
        eval_img_dir, eval_msk_dir,
        img_size=args.img_size, augment=False,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
    )

    tta_label = f" (TTA={tta_mode})" if tta_mode != "none" else ""
    print(f"\nEvaluating on {len(val_ds)} images{tta_label}...")

    results = []

    for imgs, masks, case_ids in tqdm(val_loader, desc="Predicting"):
        imgs = imgs.to(device)

        if tta_mode == "d4":
            probs = _predict_tta_d4(model, imgs, device)
        else:
            logits = model(imgs)
            probs = torch.sigmoid(logits)

        preds_bin = _to_numpy((probs > 0.5).cpu()).astype(np.uint8)

        for i, case_id in enumerate(case_ids):
            pred_512 = preds_bin[i, 0]

            # Load native-resolution GT for fair comparison
            gt_native = np.array(Image.open(eval_msk_dir / f"{case_id}.png"))
            if gt_native.max() > 1:
                gt_native = (gt_native > 127).astype(np.uint8)
            else:
                gt_native = gt_native.astype(np.uint8)
            native_h, native_w = gt_native.shape[:2]

            # Resize prediction to native resolution
            pred_native = np.array(
                Image.fromarray(pred_512 * 255, mode="L").resize(
                    (native_w, native_h), Image.NEAREST
                )
            ) > 127

            # Save prediction PNG
            pred_img = Image.fromarray(pred_native.astype(np.uint8) * 255, mode="L")
            pred_img.save(pred_dir / f"{case_id}.png")

            # Compute metrics at native resolution
            pred_mask = pred_native.astype(np.uint8)
            metrics = compute_metrics(pred_mask, gt_native)
            has_tumor_gt = int(gt_native.any())
            has_tumor_pred = int(pred_mask.any())
            gt_area_pct = float(np.count_nonzero(gt_native) / gt_native.size * 100)

            results.append({
                "case_id": case_id,
                "dice": round(metrics["dice"], 4),
                "iou": round(metrics["iou"], 4),
                "precision": round(metrics["precision"], 4),
                "recall": round(metrics["recall"], 4),
                "has_tumor_gt": has_tumor_gt,
                "has_tumor_pred": has_tumor_pred,
                "gt_area_pct": round(gt_area_pct, 4),
            })

    # Save CSV
    csv_path = out_dir / "eval_summary.csv"
    fieldnames = ["case_id", "dice", "iou", "precision", "recall",
                  "has_tumor_gt", "has_tumor_pred", "gt_area_pct"]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    print(f"Saved {csv_path}")

    # Summary
    dices = [r["dice"] for r in results]
    tumor_dices = [r["dice"] for r in results if r["has_tumor_gt"] == 1]
    nontumor_results = [r for r in results if r["has_tumor_gt"] == 0]
    fp_count = sum(1 for r in nontumor_results if r["has_tumor_pred"] == 1)

    print(f"\n{'='*60}")
    print(f"Evaluation Summary (n={len(results)})")
    print(f"{'='*60}")
    print(f"Overall Dice:     {np.mean(dices):.4f} +/- {np.std(dices):.4f}")
    if tumor_dices:
        print(f"Tumor-only Dice:  {np.mean(tumor_dices):.4f} +/- {np.std(tumor_dices):.4f}  (n={len(tumor_dices)})")
    print(f"FP:               {fp_count}/{len(nontumor_results)}")
    print(f"\nPredictions saved to {pred_dir}")

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Train SMP UNet on GL261 ultrasound segmentation",
    )
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--img-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="Decoder learning rate (default: 1e-3)")
    parser.add_argument("--encoder-lr-mult", type=float, default=0.1,
                        help="Encoder LR = lr * this (default: 0.1)")
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--out-dir", type=str, default=str(OUT_DIR))
    parser.add_argument("--encoder", type=str, default="resnet34",
                        help="SMP encoder backbone (default: resnet34)")
    parser.add_argument("--arch", type=str, default="unet",
                        choices=["unet", "unetplusplus", "manet", "deeplabv3plus", "fpn"],
                        help="Decoder architecture (default: unet)")
    parser.add_argument("--loss", type=str, default="bce_dice",
                        choices=["bce_dice", "focal_tversky", "lovasz"],
                        help="Loss function (default: bce_dice)")
    parser.add_argument("--cutmix", action="store_true",
                        help="Enable CutMix augmentation")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed for reproducibility")
    parser.add_argument("--warmup-epochs", type=int, default=5,
                        help="Linear warmup epochs (default: 5)")
    parser.add_argument("--no-amp", action="store_true",
                        help="Disable automatic mixed precision")
    parser.add_argument("--aug-preset", type=str, default="base",
                        choices=["base", "medical_v1"],
                        help="Augmentation preset (medical_v1 adds elastic deform)")
    parser.add_argument("--tta", type=str, default="none",
                        choices=["none", "d4"],
                        help="Test-time augmentation (d4 = 8-fold dihedral)")
    parser.add_argument("--eval-only", action="store_true",
                        help="Skip training, load checkpoint and evaluate")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to model checkpoint (for --eval-only)")
    # Override data directories (for custom data layouts)
    parser.add_argument("--img-dir", type=str, default=None,
                        help="Override training image directory")
    parser.add_argument("--msk-dir", type=str, default=None,
                        help="Override training mask directory")
    parser.add_argument("--val-img-dir", type=str, default=None,
                        help="Override validation image directory")
    parser.add_argument("--val-msk-dir", type=str, default=None,
                        help="Override validation mask directory")
    return parser.parse_args()


def main():
    args = parse_args()

    if args.eval_only:
        ckpt_path = args.checkpoint or str(Path(args.out_dir) / "best_model.pth")
        if not Path(ckpt_path).exists():
            print(f"[ERROR] Checkpoint not found: {ckpt_path}")
            sys.exit(1)

        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        model = build_model(
            ckpt.get("arch", args.arch),
            ckpt.get("encoder", args.encoder),
        )
        model.load_state_dict(ckpt["model_state_dict"])
        print(f"Loaded checkpoint from {ckpt_path} (epoch {ckpt['epoch']}, "
              f"val_dice={ckpt['val_dice']:.4f})")
        evaluate_model(model, args)
    else:
        train(args)
        # Reload best checkpoint for evaluation
        best_ckpt = Path(args.out_dir) / "best_model.pth"
        if best_ckpt.exists():
            ckpt = torch.load(best_ckpt, map_location="cpu", weights_only=False)
            model = build_model(
                ckpt.get("arch", args.arch),
                ckpt.get("encoder", args.encoder),
            )
            model.load_state_dict(ckpt["model_state_dict"])
            print(f"\nReloaded best checkpoint (epoch {ckpt['epoch']}) for evaluation")
            evaluate_model(model, args)
        else:
            print("[WARN] No best_model.pth found -- skipping evaluation")


if __name__ == "__main__":
    main()

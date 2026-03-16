#!/usr/bin/env python3
"""Evaluate segmentation predictions on the GL261 validation set.

Computes per-image: Dice, IoU, Precision, Recall, HD95, Surface Dice.
Generates per-condition stratified summaries and visualizations.

Usage:
  python evaluate.py --pred-dir checkpoints/predictions
  python evaluate.py --pred-dir checkpoints/predictions --gt-dir data/processed/nnunet/Dataset501_GL261/labelsTs
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

from gl261.metrics import compute_all_metrics

ROOT = Path(__file__).resolve().parent

DEFAULT_GT_DIR = ROOT / "data" / "processed" / "nnunet" / "Dataset501_GL261" / "labelsTs"
DEFAULT_IMG_DIR = ROOT / "data" / "processed" / "nnunet" / "Dataset501_GL261" / "imagesTs"


# ---------------------------------------------------------------------------
# Condition mapping
# ---------------------------------------------------------------------------

def _load_condition_map(manifest_csv: Path = None) -> dict[str, str]:
    """Load case_id -> condition mapping from manifest.

    Case IDs (GL261_NNNN) are assigned globally in sorted(rec_id, filename) order
    across all splits, so we must replicate that global assignment.
    """
    if manifest_csv is None:
        manifest_csv = ROOT / "data" / "processed" / "csv" / "manifest_full.csv"
    if not manifest_csv.exists():
        return {}

    df = pd.read_csv(manifest_csv)
    df_sorted = df.sort_values(["rec_id", "filename"]).reset_index(drop=True)
    df_sorted["case_id"] = [f"GL261_{i+1:04d}" for i in range(len(df_sorted))]

    val_df = df_sorted[df_sorted["split"] == "val"]
    return dict(zip(val_df["case_id"], val_df["condition"]))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Evaluate GL261 segmentation")
    parser.add_argument("--pred-dir", type=Path, required=True,
                        help="Directory with prediction PNGs (case_id.png)")
    parser.add_argument("--gt-dir", type=Path, default=DEFAULT_GT_DIR)
    parser.add_argument("--img-dir", type=Path, default=DEFAULT_IMG_DIR)
    parser.add_argument("--out-dir", type=Path, default=None,
                        help="Output directory (default: pred-dir parent)")
    parser.add_argument("--model-name", type=str, default="Model",
                        help="Model name for display")
    args = parser.parse_args()

    pred_dir = args.pred_dir
    gt_dir = args.gt_dir
    img_dir = args.img_dir
    out_dir = args.out_dir or pred_dir.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    if not pred_dir.exists():
        print(f"[ERROR] Prediction dir not found: {pred_dir}")
        return

    n_gt = len(list(gt_dir.glob("*.png")))
    n_pred = len(list(pred_dir.glob("*.png")))
    if n_pred != n_gt:
        print(f"[WARN] Prediction count mismatch: {n_pred} predictions vs {n_gt} GT masks")

    condition_map = _load_condition_map()

    gt_files = sorted(gt_dir.glob("*.png"))
    print(f"Found {len(gt_files)} ground truth masks in {gt_dir}")

    results = []
    for gt_path in tqdm(gt_files, desc="Evaluating"):
        case_id = gt_path.stem
        pred_path = pred_dir / gt_path.name

        gt_arr = np.array(Image.open(gt_path))
        has_tumor_gt = bool(np.any(gt_arr > 0))

        if not pred_path.exists():
            print(f"[WARN] No prediction for {case_id}")
            continue

        pred_arr = np.array(Image.open(pred_path))

        if pred_arr.shape != gt_arr.shape:
            pred_pil = Image.fromarray(pred_arr)
            pred_pil = pred_pil.resize((gt_arr.shape[1], gt_arr.shape[0]), Image.NEAREST)
            pred_arr = np.array(pred_pil)

        metrics = compute_all_metrics(pred_arr, gt_arr)
        metrics.update({
            "case_id": case_id,
            "has_tumor_gt": int(has_tumor_gt),
            "has_tumor_pred": int(np.any(pred_arr > 0)),
            "gt_area_pct": round(float(np.count_nonzero(gt_arr) / gt_arr.size * 100), 4),
            "condition": condition_map.get(case_id, "unknown"),
        })
        results.append(metrics)

    if not results:
        print("No predictions found. Exiting.")
        return

    df = pd.DataFrame(results)
    csv_path = out_dir / "eval_summary.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nSaved {csv_path} ({len(df)} rows)")

    # -----------------------------------------------------------------------
    # Summary statistics
    # -----------------------------------------------------------------------

    def _summarize(sub_df: pd.DataFrame, label: str):
        n = len(sub_df)
        if n == 0:
            return
        d = sub_df["dice"]
        h = sub_df["hd95"].dropna()
        sd2 = sub_df["surface_dice_2px"]
        neg = sub_df[sub_df["has_tumor_gt"] == 0]
        neg_fp = neg["has_tumor_pred"].mean() if len(neg) > 0 else float("nan")
        hd_str = f"{h.mean():.1f}" if len(h) > 0 else "-"
        neg_fp_str = f"{neg_fp:.3f}" if not np.isnan(neg_fp) else "-"
        print(f"  {label:20s}  n={n:4d}  Dice={d.mean():.4f}+/-{d.std():.4f}  "
              f"HD95={hd_str:>6s}  SD@2px={sd2.mean():.4f}  NegFP={neg_fp_str}")

    print(f"\n{'='*80}")
    print(f"Evaluation Summary -- {args.model_name}")
    print(f"{'='*80}")

    _summarize(df, "Overall")

    tumor_df = df[df["has_tumor_gt"] == 1]
    nontumor_df = df[df["has_tumor_gt"] == 0]
    _summarize(tumor_df, "Tumor images")
    _summarize(nontumor_df, "Non-tumor images")

    conditions = sorted(df["condition"].unique())
    if len(conditions) > 1 and "unknown" not in conditions:
        print(f"\n  Per-condition:")
        for cond in conditions:
            cond_df = df[df["condition"] == cond]
            _summarize(cond_df, f"  {cond}")

    # -----------------------------------------------------------------------
    # Visualizations
    # -----------------------------------------------------------------------

    fig_dir = out_dir / "figures"
    fig_dir.mkdir(exist_ok=True)

    # 1. Dice distribution
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(df["dice"], bins=20, edgecolor="black", alpha=0.7)
    ax.axvline(df["dice"].mean(), color="red", linestyle="--",
               label=f"mean={df['dice'].mean():.3f}")
    ax.set_xlabel("Dice Score")
    ax.set_ylabel("Count")
    ax.set_title(f"{args.model_name} Dice Distribution on GL261 Val")
    ax.legend()
    fig.savefig(fig_dir / "dice_distribution.png", dpi=150, bbox_inches="tight")
    plt.close()

    # 2. Best / median / worst examples
    df_sorted = df.sort_values("dice")
    if len(df_sorted) >= 3 and img_dir.exists():
        examples = {
            "worst": df_sorted.iloc[0],
            "median": df_sorted.iloc[len(df_sorted)//2],
            "best": df_sorted.iloc[-1],
        }

        fig, axes = plt.subplots(3, 3, figsize=(15, 15))
        for row_idx, (label, case_row) in enumerate(examples.items()):
            cid = case_row["case_id"]
            img_path = img_dir / f"{cid}_0000.png"
            gt_path_ = gt_dir / f"{cid}.png"
            pred_path_ = pred_dir / f"{cid}.png"

            if img_path.exists():
                img_arr = np.array(Image.open(img_path))
                gt_arr = np.array(Image.open(gt_path_))
                pred_arr = np.array(Image.open(pred_path_)) if pred_path_.exists() else np.zeros_like(gt_arr)

                axes[row_idx, 0].imshow(img_arr, cmap="gray")
                axes[row_idx, 0].set_title(f"{label}: {cid}\nDice={case_row['dice']:.3f} [{case_row['condition']}]")
                axes[row_idx, 0].axis("off")

                overlay_gt = np.stack([img_arr]*3, -1) if img_arr.ndim == 2 else img_arr.copy()
                overlay_gt[gt_arr > 0] = [0, 255, 0]
                axes[row_idx, 1].imshow(overlay_gt)
                axes[row_idx, 1].set_title("Ground Truth")
                axes[row_idx, 1].axis("off")

                overlay_pred = np.stack([img_arr]*3, -1) if img_arr.ndim == 2 else img_arr.copy()
                overlay_pred[pred_arr > 0] = [255, 0, 0]
                axes[row_idx, 2].imshow(overlay_pred)
                axes[row_idx, 2].set_title("Prediction")
                axes[row_idx, 2].axis("off")

        fig.suptitle(f"{args.model_name} -- Best / Median / Worst", fontsize=14)
        fig.savefig(fig_dir / "example_predictions.png", dpi=150, bbox_inches="tight")
        plt.close()

    # 3. Tumor area vs Dice scatter
    if len(tumor_df) > 5:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.scatter(tumor_df["gt_area_pct"], tumor_df["dice"], alpha=0.5, s=20)
        ax.set_xlabel("GT Tumor Area (%)")
        ax.set_ylabel("Dice Score")
        ax.set_title(f"{args.model_name} -- Tumor Size vs Dice")
        ax.axhline(0.5, color="red", linestyle=":", alpha=0.5, label="Dice=0.5")
        ax.legend()
        fig.savefig(fig_dir / "tumor_area_vs_dice.png", dpi=150, bbox_inches="tight")
        plt.close()

    print(f"\nFigures saved to {fig_dir}")

    # Save compact summary JSON
    summary = {
        "model": args.model_name,
        "n_test": len(df),
        "overall_dice": round(df["dice"].mean(), 4),
        "tumor_dice": round(tumor_df["dice"].mean(), 4) if len(tumor_df) > 0 else None,
        "nontumor_dice": round(nontumor_df["dice"].mean(), 4) if len(nontumor_df) > 0 else None,
        "overall_hd95": round(df["hd95"].dropna().mean(), 2) if len(df["hd95"].dropna()) > 0 else None,
        "surface_dice_2px": round(df["surface_dice_2px"].mean(), 4),
        "neg_fp_rate": round(nontumor_df["has_tumor_pred"].mean(), 4) if len(nontumor_df) > 0 else None,
    }
    with (out_dir / "eval_metrics.json").open("w") as f:
        json.dump(summary, f, indent=2)
    print(f"Compact summary: {out_dir / 'eval_metrics.json'}")


if __name__ == "__main__":
    main()

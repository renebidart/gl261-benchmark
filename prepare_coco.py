#!/usr/bin/env python3
"""Convert GL261 nnU-Net data to COCO format for MMDetection / UltraSam.

Builds a COCO-style dataset from the existing nnU-Net-formatted GL261 split:
  - train: Dataset501_GL261/imagesTr + labelsTr
  - val:   Dataset501_GL261/imagesTs + labelsTs

It also converts the single-channel images to 3-channel RGB PNGs so the
OpenMMLab preprocessing path matches the upstream UltraSam assumptions.

Usage:
    python prepare_coco.py                       # uses defaults
    python prepare_coco.py --out-root /tmp/coco   # custom output dir

Requires: pycocotools  (pip install gl261-benchmark[coco])
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from pycocotools import mask as mask_utils
from scipy import ndimage
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent
NNUNET_DIR = ROOT / "data" / "processed" / "nnunet" / "Dataset501_GL261"
MANIFEST_CSV = ROOT / "data" / "processed" / "csv" / "manifest_full.csv"
DEFAULT_OUT = ROOT / "data" / "processed" / "coco" / "GL261"


@dataclass(frozen=True)
class SplitSpec:
    name: str
    img_dir: Path
    mask_dir: Path


SPLITS = (
    SplitSpec("train", NNUNET_DIR / "imagesTr", NNUNET_DIR / "labelsTr"),
    SplitSpec("val", NNUNET_DIR / "imagesTs", NNUNET_DIR / "labelsTs"),
)


def load_case_metadata() -> dict[str, dict[str, object]] | None:
    """Load manifest metadata if available. Returns None if manifest missing."""
    if not MANIFEST_CSV.exists():
        return None
    df = pd.read_csv(MANIFEST_CSV)
    df = df.sort_values(["rec_id", "filename"]).reset_index(drop=True)
    df["case_id"] = [f"GL261_{idx + 1:04d}" for idx in range(len(df))]
    return {
        row.case_id: {
            "mouse_id": row.mouse_id,
            "rec_id": row.rec_id,
            "condition": row.condition,
            "split": row.split,
            "has_tumor": int(row.has_tumor),
            "tumor_area_pct": float(row.tumor_area_pct),
            "filename": row.filename,
        }
        for row in df.itertuples(index=False)
    }


def ensure_rgb(src_path: Path, dst_path: Path) -> tuple[int, int]:
    """Convert grayscale image to 3-channel RGB PNG."""
    img = Image.open(src_path).convert("L")
    rgb = Image.merge("RGB", (img, img, img))
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    rgb.save(dst_path)
    return rgb.size


def copy_mask(src_path: Path, dst_path: Path) -> None:
    """Binarize and copy mask to output directory."""
    mask = Image.open(src_path).convert("L")
    mask = Image.fromarray((np.array(mask) > 0).astype(np.uint8) * 255, mode="L")
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    mask.save(dst_path)


def encode_rle(binary_mask: np.ndarray) -> dict[str, object]:
    """Encode binary mask as COCO RLE."""
    rle = mask_utils.encode(np.asfortranarray(binary_mask.astype(np.uint8)))
    rle["counts"] = rle["counts"].decode("utf-8")
    return rle


def coco_bbox(binary_mask: np.ndarray) -> list[int] | None:
    """Compute COCO-format bounding box [x, y, w, h] from binary mask."""
    ys, xs = np.where(binary_mask > 0)
    if len(xs) == 0:
        return None
    x1, y1 = int(xs.min()), int(ys.min())
    x2, y2 = int(xs.max()), int(ys.max())
    return [x1, y1, x2 - x1 + 1, y2 - y1 + 1]


def build_coco_for_split(
    split: SplitSpec,
    out_root: Path,
    case_meta: dict[str, dict[str, object]] | None,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    """Build COCO JSON and manifest rows for one split."""
    images = []
    annotations = []
    rows = []
    ann_id = 1

    out_img_dir = out_root / "images" / split.name
    out_mask_dir = out_root / "masks" / split.name

    img_paths = sorted(split.img_dir.glob("GL261_*_0000.png"))
    for img_idx, img_path in enumerate(tqdm(img_paths, desc=f"Prep {split.name}"), start=1):
        case_id = img_path.stem.replace("_0000", "")
        mask_path = split.mask_dir / f"{case_id}.png"
        if not mask_path.exists():
            raise FileNotFoundError(f"Missing mask for {img_path}: {mask_path}")

        out_img_path = out_img_dir / f"{case_id}.png"
        out_mask_path = out_mask_dir / f"{case_id}.png"
        width, height = ensure_rgb(img_path, out_img_path)
        copy_mask(mask_path, out_mask_path)

        images.append({
            "id": img_idx,
            "file_name": out_img_path.name,
            "width": width,
            "height": height,
        })

        mask_arr = np.array(Image.open(mask_path).convert("L"))
        binary = (mask_arr > 0).astype(np.uint8)
        labeled, n_components = ndimage.label(binary)
        for comp_idx in range(1, n_components + 1):
            comp = (labeled == comp_idx).astype(np.uint8)
            if comp.sum() == 0:
                continue
            bbox = coco_bbox(comp)
            if bbox is None:
                continue
            annotations.append({
                "id": ann_id,
                "image_id": img_idx,
                "category_id": 1,
                "segmentation": encode_rle(comp),
                "area": int(comp.sum()),
                "bbox": bbox,
                "iscrowd": 0,
            })
            ann_id += 1

        # Build manifest row with metadata if available
        row = {
            "case_id": case_id,
            "split": split.name,
            "image_path": str(out_img_path),
            "mask_path": str(out_mask_path),
        }
        if case_meta is not None:
            meta = case_meta.get(case_id)
            if meta is not None:
                row.update({
                    "mouse_id": meta["mouse_id"],
                    "rec_id": meta["rec_id"],
                    "condition": meta["condition"],
                    "has_tumor": meta["has_tumor"],
                    "tumor_area_pct": meta["tumor_area_pct"],
                })
        rows.append(row)

    coco = {
        "images": images,
        "annotations": annotations,
        "categories": [{"id": 1, "name": "tumor", "supercategory": "tumor"}],
    }
    return coco, rows


def write_csv(rows: list[dict[str, object]], path: Path) -> None:
    if not rows:
        raise ValueError(f"No rows to write for {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert GL261 nnU-Net data to COCO format for UltraSam / MMDetection"
    )
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    out_root = args.out_root
    out_root.mkdir(parents=True, exist_ok=True)

    case_meta = load_case_metadata()
    if case_meta is None:
        print("Warning: manifest_full.csv not found, COCO JSON will lack metadata fields")

    split_rows = []
    summary = {"source": str(NNUNET_DIR), "out_root": str(out_root), "splits": {}}
    for split in SPLITS:
        coco, rows = build_coco_for_split(split, out_root, case_meta)

        ann_path = out_root / "annotations" / f"{split.name}.GL261_coco.json"
        ann_path.parent.mkdir(parents=True, exist_ok=True)
        with ann_path.open("w", encoding="utf-8") as f:
            json.dump(coco, f)

        manifest_path = out_root / "metadata" / f"{split.name}_manifest.csv"
        write_csv(rows, manifest_path)
        split_rows.extend(rows)

        summary["splits"][split.name] = {
            "images": len(coco["images"]),
            "annotations": len(coco["annotations"]),
            "manifest_csv": str(manifest_path),
            "annotation_json": str(ann_path),
        }

    combined_path = out_root / "metadata" / "all_cases.csv"
    write_csv(split_rows, combined_path)
    summary["combined_manifest"] = str(combined_path)
    with (out_root / "metadata" / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

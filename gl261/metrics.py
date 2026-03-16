"""Segmentation metrics for binary mask evaluation.

Canonical implementations for Dice, IoU, HD95, Surface Dice, and
precision/recall. Used by evaluate.py.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree


def dice_score(pred: np.ndarray, gt: np.ndarray) -> float:
    """Per-image Dice coefficient for binary masks."""
    pred_b, gt_b = pred > 0, gt > 0
    if not pred_b.any() and not gt_b.any():
        return 1.0
    if not pred_b.any() or not gt_b.any():
        return 0.0
    intersection = np.logical_and(pred_b, gt_b).sum()
    return 2.0 * intersection / (pred_b.sum() + gt_b.sum())


def iou_score(pred: np.ndarray, gt: np.ndarray) -> float:
    """Per-image Intersection over Union for binary masks."""
    pred_b, gt_b = pred > 0, gt > 0
    if not pred_b.any() and not gt_b.any():
        return 1.0
    if not pred_b.any() or not gt_b.any():
        return 0.0
    intersection = np.logical_and(pred_b, gt_b).sum()
    union = np.logical_or(pred_b, gt_b).sum()
    return float(intersection / union)


def precision_recall(pred: np.ndarray, gt: np.ndarray) -> tuple[float, float]:
    """Precision and recall for binary masks."""
    pred_b, gt_b = pred > 0, gt > 0
    tp = np.logical_and(pred_b, gt_b).sum()
    prec = tp / pred_b.sum() if pred_b.any() else (1.0 if not gt_b.any() else 0.0)
    rec = tp / gt_b.sum() if gt_b.any() else (1.0 if not pred_b.any() else 0.0)
    return float(prec), float(rec)


def _extract_boundary(mask: np.ndarray) -> np.ndarray:
    """Extract boundary pixels using morphological erosion."""
    if not mask.any():
        return np.zeros_like(mask, dtype=bool)
    struct = ndimage.generate_binary_structure(2, 1)  # 4-connected
    eroded = ndimage.binary_erosion(mask, structure=struct)
    return mask & ~eroded


def hd95_score(pred: np.ndarray, gt: np.ndarray) -> float:
    """95th percentile Hausdorff distance. Returns NaN if either mask is empty."""
    pred_b, gt_b = pred > 0, gt > 0
    if not pred_b.any() and not gt_b.any():
        return 0.0
    if not pred_b.any() or not gt_b.any():
        return float("nan")
    try:
        from medpy.metric import hd95
        return float(hd95(pred_b, gt_b))
    except Exception:
        pred_boundary = _extract_boundary(pred_b)
        gt_boundary = _extract_boundary(gt_b)
        pred_pts = np.argwhere(pred_boundary)
        gt_pts = np.argwhere(gt_boundary)
        if len(pred_pts) == 0 or len(gt_pts) == 0:
            return float("nan")
        gt_tree = cKDTree(gt_pts)
        pred_tree = cKDTree(pred_pts)
        d_pred_to_gt = gt_tree.query(pred_pts, k=1)[0]
        d_gt_to_pred = pred_tree.query(gt_pts, k=1)[0]
        return float(np.percentile(np.concatenate([d_pred_to_gt, d_gt_to_pred]), 95))


def surface_dice(pred: np.ndarray, gt: np.ndarray, tolerance: float = 2.0) -> float:
    """Normalized Surface Dice at given pixel tolerance.

    For each boundary pixel in pred, check if it's within tolerance of any
    boundary pixel in gt, and vice versa.
    """
    pred_b, gt_b = pred > 0, gt > 0
    if not pred_b.any() and not gt_b.any():
        return 1.0
    if not pred_b.any() or not gt_b.any():
        return 0.0

    pred_boundary = _extract_boundary(pred_b)
    gt_boundary = _extract_boundary(gt_b)

    n_pred = pred_boundary.sum()
    n_gt = gt_boundary.sum()
    if n_pred == 0 and n_gt == 0:
        return 1.0
    if n_pred == 0 or n_gt == 0:
        return 0.0

    dist_to_gt = ndimage.distance_transform_edt(~gt_boundary)
    dist_to_pred = ndimage.distance_transform_edt(~pred_boundary)

    pred_within = (dist_to_gt[pred_boundary] <= tolerance).sum()
    gt_within = (dist_to_pred[gt_boundary] <= tolerance).sum()

    return float((pred_within + gt_within) / (n_pred + n_gt))


def compute_all_metrics(pred: np.ndarray, gt: np.ndarray) -> dict:
    """Compute all metrics for a single prediction-GT pair."""
    d = dice_score(pred, gt)
    iou = iou_score(pred, gt)
    prec, rec = precision_recall(pred, gt)
    h95 = hd95_score(pred, gt)
    sd1 = surface_dice(pred, gt, tolerance=1.0)
    sd2 = surface_dice(pred, gt, tolerance=2.0)
    sd5 = surface_dice(pred, gt, tolerance=5.0)

    return {
        "dice": round(d, 4),
        "iou": round(iou, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "hd95": round(h95, 2) if not np.isnan(h95) else None,
        "surface_dice_1px": round(sd1, 4),
        "surface_dice_2px": round(sd2, 4),
        "surface_dice_5px": round(sd5, 4),
    }

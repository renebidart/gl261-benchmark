"""Regression tests for GL261 metric calculations.

Guards against bugs that silently corrupt reported numbers:
1. NegFP must be computed from GT-negative frames only.
2. Dice/IoU edge cases (both empty, one empty, etc.).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from gl261.metrics import dice_score, iou_score, precision_recall, surface_dice


# ---------------------------------------------------------------------------
# NegFP: must only count false positives on GT-negative frames
# ---------------------------------------------------------------------------


def _neg_fp(df: pd.DataFrame) -> float:
    """Compute NegFP rate: fraction of GT-negative images with false positive."""
    neg_df = df[df["has_tumor_gt"] == 0]
    return neg_df["has_tumor_pred"].mean() if len(neg_df) > 0 else 0.0


def test_negfp_ignores_tumor_frames() -> None:
    """NegFP must not count true-positive predictions on tumor frames."""
    df = pd.DataFrame({
        "has_tumor_gt":   [1, 1, 1, 0, 0, 0, 0, 0],
        "has_tumor_pred": [1, 1, 1, 1, 0, 0, 0, 0],
    })
    # 3 tumor frames all predicted positive (correct), 1/5 negative frames FP
    assert _neg_fp(df) == 1.0 / 5.0


def test_negfp_all_negative_no_fp() -> None:
    df = pd.DataFrame({
        "has_tumor_gt":   [0, 0, 0],
        "has_tumor_pred": [0, 0, 0],
    })
    assert _neg_fp(df) == 0.0


def test_negfp_all_negative_all_fp() -> None:
    df = pd.DataFrame({
        "has_tumor_gt":   [0, 0, 0],
        "has_tumor_pred": [1, 1, 1],
    })
    assert _neg_fp(df) == 1.0


def test_negfp_no_negative_frames() -> None:
    """Mouse with only tumor frames: NegFP should be 0."""
    df = pd.DataFrame({
        "has_tumor_gt":   [1, 1],
        "has_tumor_pred": [1, 1],
    })
    assert _neg_fp(df) == 0.0


# ---------------------------------------------------------------------------
# Metric edge cases
# ---------------------------------------------------------------------------


def test_dice_both_empty() -> None:
    pred = np.zeros((10, 10), dtype=np.uint8)
    gt = np.zeros((10, 10), dtype=np.uint8)
    assert dice_score(pred, gt) == 1.0


def test_dice_pred_empty_gt_not() -> None:
    pred = np.zeros((10, 10), dtype=np.uint8)
    gt = np.ones((10, 10), dtype=np.uint8)
    assert dice_score(pred, gt) == 0.0


def test_dice_perfect_overlap() -> None:
    mask = np.zeros((10, 10), dtype=np.uint8)
    mask[3:7, 3:7] = 1
    assert dice_score(mask, mask) == 1.0


def test_iou_both_empty() -> None:
    pred = np.zeros((10, 10), dtype=np.uint8)
    gt = np.zeros((10, 10), dtype=np.uint8)
    assert iou_score(pred, gt) == 1.0


def test_iou_perfect_overlap() -> None:
    mask = np.zeros((10, 10), dtype=np.uint8)
    mask[3:7, 3:7] = 1
    assert iou_score(mask, mask) == 1.0


def test_precision_recall_perfect() -> None:
    mask = np.zeros((10, 10), dtype=np.uint8)
    mask[3:7, 3:7] = 1
    prec, rec = precision_recall(mask, mask)
    assert prec == 1.0
    assert rec == 1.0


def test_precision_recall_all_fp() -> None:
    pred = np.ones((10, 10), dtype=np.uint8)
    gt = np.zeros((10, 10), dtype=np.uint8)
    prec, rec = precision_recall(pred, gt)
    assert prec == 0.0
    # GT is empty but pred is non-empty: recall=0.0 by convention
    assert rec == 0.0


def test_surface_dice_both_empty() -> None:
    pred = np.zeros((10, 10), dtype=np.uint8)
    gt = np.zeros((10, 10), dtype=np.uint8)
    assert surface_dice(pred, gt) == 1.0


def test_surface_dice_perfect() -> None:
    mask = np.zeros((20, 20), dtype=np.uint8)
    mask[5:15, 5:15] = 1
    assert surface_dice(mask, mask) == 1.0

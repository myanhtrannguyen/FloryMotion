"""
metrics.py

Segmentation metrics for binary flower segmentation.

Metrics:
- Dice
- IoU
- Precision
- HD95
- Boundary F-score

Input convention:
- pred and target can be NumPy arrays or PyTorch tensors.
- Accepted shapes:
    [H, W]
    [B, H, W]
    [B, 1, H, W]
- pred can be binary mask, probability map, or logits/probabilities.
- target should be binary mask or mask-like array.

Foreground = 1
Background = 0
"""

from __future__ import annotations

from typing import Dict, Tuple, Union

import numpy as np
from scipy import ndimage

ArrayLike = Union[np.ndarray, "object"]


def _to_numpy(x: ArrayLike) -> np.ndarray:
    """Convert NumPy array or PyTorch tensor to NumPy array."""
    if hasattr(x, "detach"):
        x = x.detach().cpu().numpy()

    return np.asarray(x)


def _squeeze_channel(x: np.ndarray) -> np.ndarray:
    """
    Convert [B, 1, H, W] to [B, H, W].
    Keep [H, W] and [B, H, W] unchanged.
    """
    if x.ndim == 4 and x.shape[1] == 1:
        x = x[:, 0]

    return x


def _ensure_batch(x: np.ndarray) -> np.ndarray:
    """Convert [H, W] to [1, H, W]."""
    if x.ndim == 2:
        x = x[None, ...]

    return x


def _binarize(x: ArrayLike, threshold: float = 0.5) -> np.ndarray:
    """
    Convert input mask/probability map to binary mask.
    Values greater than threshold are treated as foreground.
    """
    x = _to_numpy(x)
    x = _squeeze_channel(x)
    x = _ensure_batch(x)

    if x.dtype == np.bool_:
        return x.astype(np.uint8)

    return (x > threshold).astype(np.uint8)


def _validate_shapes(pred: np.ndarray, target: np.ndarray) -> None:
    if pred.shape != target.shape:
        raise ValueError(
            f"Shape mismatch: pred shape {pred.shape}, target shape {target.shape}"
        )

    if pred.ndim != 3:
        raise ValueError(
            f"Expected masks with shape [B, H, W] after preprocessing, got {pred.shape}"
        )


def dice_score(
    pred: ArrayLike,
    target: ArrayLike,
    threshold: float = 0.5,
    eps: float = 1e-7,
) -> float:
    """
    Dice = 2TP / (2TP + FP + FN)

    Higher is better.
    """
    pred = _binarize(pred, threshold)
    target = _binarize(target, threshold)
    _validate_shapes(pred, target)

    scores = []

    for p, g in zip(pred, target):
        p = p.astype(bool)
        g = g.astype(bool)

        intersection = np.logical_and(p, g).sum()
        denominator = p.sum() + g.sum()

        if denominator == 0:
            scores.append(1.0)
        else:
            dice = (2.0 * intersection + eps) / (denominator + eps)
            scores.append(float(dice))

    return float(np.mean(scores))


def iou_score(
    pred: ArrayLike,
    target: ArrayLike,
    threshold: float = 0.5,
    eps: float = 1e-7,
) -> float:
    """
    IoU / Jaccard = TP / (TP + FP + FN)

    Higher is better.
    """
    pred = _binarize(pred, threshold)
    target = _binarize(target, threshold)
    _validate_shapes(pred, target)

    scores = []

    for p, g in zip(pred, target):
        p = p.astype(bool)
        g = g.astype(bool)

        intersection = np.logical_and(p, g).sum()
        union = np.logical_or(p, g).sum()

        if union == 0:
            scores.append(1.0)
        else:
            iou = (intersection + eps) / (union + eps)
            scores.append(float(iou))

    return float(np.mean(scores))


def precision_score(
    pred: ArrayLike,
    target: ArrayLike,
    threshold: float = 0.5,
    eps: float = 1e-7,
) -> float:
    """
    Precision = TP / (TP + FP)

    Higher is better.
    """
    pred = _binarize(pred, threshold)
    target = _binarize(target, threshold)
    _validate_shapes(pred, target)

    scores = []

    for p, g in zip(pred, target):
        p = p.astype(bool)
        g = g.astype(bool)

        tp = np.logical_and(p, g).sum()
        fp = np.logical_and(p, np.logical_not(g)).sum()

        if tp + fp == 0:
            scores.append(1.0 if g.sum() == 0 else 0.0)
        else:
            precision = (tp + eps) / (tp + fp + eps)
            scores.append(float(precision))

    return float(np.mean(scores))


def _surface_distances(pred_mask: np.ndarray, gt_mask: np.ndarray) -> np.ndarray:
    """
    Compute symmetric surface distances:
    - distance from predicted boundary to ground-truth boundary
    - distance from ground-truth boundary to predicted boundary
    """
    pred_mask = pred_mask.astype(bool)
    gt_mask = gt_mask.astype(bool)

    if pred_mask.sum() == 0 and gt_mask.sum() == 0:
        return np.array([0.0], dtype=np.float32)

    if pred_mask.sum() == 0 or gt_mask.sum() == 0:
        return np.array([np.inf], dtype=np.float32)

    structure = ndimage.generate_binary_structure(2, 1)

    pred_border = np.logical_xor(
        pred_mask,
        ndimage.binary_erosion(pred_mask, structure=structure),
    )

    gt_border = np.logical_xor(
        gt_mask,
        ndimage.binary_erosion(gt_mask, structure=structure),
    )

    dt_pred = ndimage.distance_transform_edt(~pred_border)
    dt_gt = ndimage.distance_transform_edt(~gt_border)

    distances_pred_to_gt = dt_gt[pred_border]
    distances_gt_to_pred = dt_pred[gt_border]

    distances = np.concatenate(
        [distances_pred_to_gt, distances_gt_to_pred]
    ).astype(np.float32)

    return distances


def hd95(
    pred: ArrayLike,
    target: ArrayLike,
    threshold: float = 0.5,
) -> float:
    """
    HD95: 95th percentile Hausdorff Distance.

    Lower is better.

    Unit:
        pixels, unless masks are resampled with physical spacing elsewhere.
    """
    pred = _binarize(pred, threshold)
    target = _binarize(target, threshold)
    _validate_shapes(pred, target)

    scores = []

    for p, g in zip(pred, target):
        distances = _surface_distances(p, g)

        if np.isinf(distances).any():
            h, w = p.shape
            image_diagonal = np.sqrt(h * h + w * w)
            scores.append(float(image_diagonal))
        else:
            scores.append(float(np.percentile(distances, 95)))

    return float(np.mean(scores))


def _mask_to_boundary(mask: np.ndarray) -> np.ndarray:
    """Convert binary mask to 1-pixel boundary map."""
    mask = mask.astype(bool)

    if mask.sum() == 0:
        return np.zeros_like(mask, dtype=np.uint8)

    structure = ndimage.generate_binary_structure(2, 1)

    eroded = ndimage.binary_erosion(
        mask,
        structure=structure,
        border_value=0,
    )

    boundary = np.logical_xor(mask, eroded)

    return boundary.astype(np.uint8)


def boundary_f_score(
    pred: ArrayLike,
    target: ArrayLike,
    threshold: float = 0.5,
    tolerance: int = 2,
    eps: float = 1e-7,
) -> Tuple[float, float, float]:
    """
    Boundary F-score for binary segmentation.

    A predicted boundary pixel is considered correct if it lies within
    `tolerance` pixels of a ground-truth boundary pixel.

    Returns:
        boundary_f_score, boundary_precision, boundary_recall
    """
    pred = _binarize(pred, threshold)
    target = _binarize(target, threshold)
    _validate_shapes(pred, target)

    bf_scores = []
    bp_scores = []
    br_scores = []

    structure = ndimage.generate_binary_structure(2, 1)

    for p, g in zip(pred, target):
        p_boundary = _mask_to_boundary(p)
        g_boundary = _mask_to_boundary(g)

        if p_boundary.sum() == 0 and g_boundary.sum() == 0:
            bf_scores.append(1.0)
            bp_scores.append(1.0)
            br_scores.append(1.0)
            continue

        if p_boundary.sum() == 0 or g_boundary.sum() == 0:
            bf_scores.append(0.0)
            bp_scores.append(0.0)
            br_scores.append(0.0)
            continue

        p_boundary_dilated = ndimage.binary_dilation(
            p_boundary,
            structure=structure,
            iterations=tolerance,
        )

        g_boundary_dilated = ndimage.binary_dilation(
            g_boundary,
            structure=structure,
            iterations=tolerance,
        )

        precision_match = np.logical_and(
            p_boundary,
            g_boundary_dilated,
        ).sum()

        recall_match = np.logical_and(
            g_boundary,
            p_boundary_dilated,
        ).sum()

        boundary_precision = (precision_match + eps) / (
            p_boundary.sum() + eps
        )

        boundary_recall = (recall_match + eps) / (
            g_boundary.sum() + eps
        )

        bf = (
            2.0 * boundary_precision * boundary_recall
            / (boundary_precision + boundary_recall + eps)
        )

        bf_scores.append(float(bf))
        bp_scores.append(float(boundary_precision))
        br_scores.append(float(boundary_recall))

    return (
        float(np.mean(bf_scores)),
        float(np.mean(bp_scores)),
        float(np.mean(br_scores)),
    )


def compute_all_metrics(
    pred: ArrayLike,
    target: ArrayLike,
    threshold: float = 0.5,
    boundary_tolerance: int = 2,
) -> Dict[str, float]:
    """
    Compute all segmentation metrics.

    Returns:
        dict with keys:
        - dice
        - iou
        - precision
        - hd95
        - boundary_f_score
        - boundary_precision
        - boundary_recall
    """
    bf, bp, br = boundary_f_score(
        pred,
        target,
        threshold=threshold,
        tolerance=boundary_tolerance,
    )

    return {
        "dice": dice_score(
            pred,
            target,
            threshold=threshold,
        ),
        "iou": iou_score(
            pred,
            target,
            threshold=threshold,
        ),
        "precision": precision_score(
            pred,
            target,
            threshold=threshold,
        ),
        "hd95": hd95(
            pred,
            target,
            threshold=threshold,
        ),
        "boundary_f_score": bf,
        "boundary_precision": bp,
        "boundary_recall": br,
    }


if __name__ == "__main__":
    # Small sanity check.
    gt = np.zeros((1, 128, 128), dtype=np.uint8)
    pred = np.zeros((1, 128, 128), dtype=np.uint8)

    gt[:, 30:90, 30:90] = 1
    pred[:, 32:92, 32:92] = 1

    metrics = compute_all_metrics(pred, gt)

    for name, value in metrics.items():
        print(f"{name}: {value:.4f}")

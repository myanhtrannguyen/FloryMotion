"""
Image style-transfer metrics.

Metrics included:
- SSIM
- LPIPS
- Edge Similarity
- FID
- KID
- CLIP-style score
- Mask Consistency
- Boundary Consistency

Input convention:
- Images can be PIL images, NumPy arrays, or PyTorch tensors.
- Accepted image shapes include HWC, CHW, BHWC, and BCHW.
- Image values can be uint8 [0, 255], float [0, 1], or float [-1, 1].
"""

from __future__ import annotations

from typing import Callable, Dict, Optional, Sequence, Tuple, Union

import numpy as np
from PIL import Image
from scipy import linalg, ndimage

ArrayLike = Union[np.ndarray, Image.Image, "object"]


def _to_numpy(x: ArrayLike) -> np.ndarray:
    if isinstance(x, Image.Image):
        return np.asarray(x.convert("RGB"))
    if hasattr(x, "detach"):
        x = x.detach().cpu().numpy()
    return np.asarray(x)


def _as_image_batch(images: ArrayLike | Sequence[ArrayLike]) -> np.ndarray:
    if isinstance(images, (list, tuple)):
        batch = [_as_image_batch(image)[0] for image in images]
        return np.stack(batch, axis=0)

    x = _to_numpy(images)

    if x.ndim == 2:
        x = x[:, :, None]

    if x.ndim == 3:
        if x.shape[0] in {1, 3} and x.shape[-1] not in {1, 3}:
            x = np.transpose(x, (1, 2, 0))
        x = x[None, ...]
    elif x.ndim == 4:
        if x.shape[1] in {1, 3} and x.shape[-1] not in {1, 3}:
            x = np.transpose(x, (0, 2, 3, 1))
    else:
        raise ValueError(f"Expected image shape HWC/CHW/BHWC/BCHW, got {x.shape}")

    x = x.astype(np.float32)
    if x.max(initial=0.0) > 2.0:
        x = x / 255.0
    elif x.min(initial=0.0) < 0.0:
        x = (x + 1.0) / 2.0

    x = np.clip(x, 0.0, 1.0)
    if x.shape[-1] == 1:
        x = np.repeat(x, 3, axis=-1)

    return x


def _validate_image_pair(pred: np.ndarray, target: np.ndarray) -> None:
    if pred.shape != target.shape:
        raise ValueError(f"Shape mismatch: pred shape {pred.shape}, target shape {target.shape}")
    if pred.ndim != 4 or pred.shape[-1] != 3:
        raise ValueError(f"Expected RGB image batch [B, H, W, 3], got {pred.shape}")


def ssim_score(pred: ArrayLike, target: ArrayLike, window_size: int = 11) -> float:
    """
    Structural Similarity Index. Higher is better.
    """
    pred_np = _as_image_batch(pred)
    target_np = _as_image_batch(target)
    _validate_image_pair(pred_np, target_np)

    try:
        from skimage.metrics import structural_similarity

        scores = [
            structural_similarity(p, t, channel_axis=-1, data_range=1.0)
            for p, t in zip(pred_np, target_np)
        ]
        return float(np.mean(scores))
    except ImportError:
        print("Warning: skimage is not installed, falling back to a simple SSIM approximation. For accurate SSIM, install scikit-image with `pip install scikit-image`.")
        # return _ssim_numpy(pred_np, target_np, window_size=window_size)


# def _ssim_numpy(pred: np.ndarray, target: np.ndarray, window_size: int = 11) -> float:
#     sigma = 1.5
#     c1 = 0.01**2
#     c2 = 0.03**2
#     scores = []

#     for p, t in zip(pred, target):
#         per_channel = []
#         for channel in range(3):
#             x = p[:, :, channel]
#             y = t[:, :, channel]
#             mu_x = ndimage.gaussian_filter(x, sigma=sigma, truncate=((window_size - 1) / 2) / sigma)
#             mu_y = ndimage.gaussian_filter(y, sigma=sigma, truncate=((window_size - 1) / 2) / sigma)
#             mu_x2 = mu_x * mu_x
#             mu_y2 = mu_y * mu_y
#             mu_xy = mu_x * mu_y
#             sigma_x2 = ndimage.gaussian_filter(x * x, sigma=sigma) - mu_x2
#             sigma_y2 = ndimage.gaussian_filter(y * y, sigma=sigma) - mu_y2
#             sigma_xy = ndimage.gaussian_filter(x * y, sigma=sigma) - mu_xy
#             numerator = (2.0 * mu_xy + c1) * (2.0 * sigma_xy + c2)
#             denominator = (mu_x2 + mu_y2 + c1) * (sigma_x2 + sigma_y2 + c2)
#             per_channel.append(np.mean(numerator / (denominator + 1e-12)))
#         scores.append(float(np.mean(per_channel)))

#     return float(np.mean(scores))


def edge_similarity(pred: ArrayLike, target: ArrayLike, eps: float = 1e-8) -> float:
    """
    Cosine similarity between Sobel edge-magnitude maps. Higher is better.
    """
    pred_np = _as_image_batch(pred)
    target_np = _as_image_batch(target)
    _validate_image_pair(pred_np, target_np)

    scores = []
    for p, t in zip(pred_np, target_np):
        p_edge = _edge_magnitude(p).reshape(-1)
        t_edge = _edge_magnitude(t).reshape(-1)
        score = np.dot(p_edge, t_edge) / (np.linalg.norm(p_edge) * np.linalg.norm(t_edge) + eps)
        scores.append(float(score))

    return float(np.mean(scores))


def _edge_magnitude(image: np.ndarray) -> np.ndarray:
    gray = 0.299 * image[:, :, 0] + 0.587 * image[:, :, 1] + 0.114 * image[:, :, 2]
    dx = ndimage.sobel(gray, axis=1)
    dy = ndimage.sobel(gray, axis=0)
    return np.sqrt(dx * dx + dy * dy).astype(np.float32)


def lpips_distance(
    pred: ArrayLike,
    target: ArrayLike,
    model: Optional[object] = None,
    net: str = "alex",
    device: Optional[str] = None,
) -> float:
    """
    LPIPS perceptual distance. Lower is better.

    Requires the optional `lpips` package unless a compatible model is passed.
    """
    import torch

    pred_tensor = _torch_image_batch(pred, device=device, value_range="-1_1")
    target_tensor = _torch_image_batch(target, device=device, value_range="-1_1")

    if model is None:
        try:
            import lpips
        except ImportError as exc:
            raise ImportError("LPIPS requires `pip install lpips` or a preloaded model.") from exc

        model = lpips.LPIPS(net=net).to(pred_tensor.device)

    model.eval()
    with torch.no_grad():
        values = model(pred_tensor, target_tensor)

    return float(values.mean().detach().cpu().item())


def _torch_image_batch(images: ArrayLike, device: Optional[str] = None, value_range: str = "0_1"):
    import torch

    x = _as_image_batch(images)
    x = torch.from_numpy(x.transpose(0, 3, 1, 2)).float()
    if value_range == "-1_1":
        x = x * 2.0 - 1.0
    if device is not None:
        x = x.to(device)
    return x


def fid_score(
    real: ArrayLike | np.ndarray,
    generated: ArrayLike | np.ndarray,
    feature_extractor: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    eps: float = 1e-6,
) -> float:
    """
    Frechet Inception Distance. Lower is better.

    For standard FID, pass an Inception feature extractor returning [N, D].
    Without one, the function uses 8x8 resized RGB pixels as lightweight
    features. Pass an Inception feature extractor for publishable FID/KID.
    """
    real_features = _extract_features(real, feature_extractor)
    generated_features = _extract_features(generated, feature_extractor)

    mu_real = real_features.mean(axis=0)
    mu_generated = generated_features.mean(axis=0)
    cov_real = np.cov(real_features, rowvar=False)
    cov_generated = np.cov(generated_features, rowvar=False)

    diff = mu_real - mu_generated
    covmean, _ = linalg.sqrtm((cov_real + eps * np.eye(cov_real.shape[0])) @ (cov_generated + eps * np.eye(cov_generated.shape[0])), disp=False)
    if np.iscomplexobj(covmean):
        covmean = covmean.real

    fid = diff.dot(diff) + np.trace(cov_real + cov_generated - 2.0 * covmean)
    return float(max(fid, 0.0))


def kid_score(
    real: ArrayLike | np.ndarray,
    generated: ArrayLike | np.ndarray,
    feature_extractor: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    degree: int = 3,
    gamma: Optional[float] = None,
    coef0: float = 1.0,
) -> float:
    """
    Kernel Inception Distance using polynomial MMD. Lower is better.

    For standard KID, pass an Inception feature extractor returning [N, D].
    """
    x = _extract_features(real, feature_extractor)
    y = _extract_features(generated, feature_extractor)
    if gamma is None:
        gamma = 1.0 / x.shape[1]

    k_xx = (gamma * x @ x.T + coef0) ** degree
    k_yy = (gamma * y @ y.T + coef0) ** degree
    k_xy = (gamma * x @ y.T + coef0) ** degree

    m = x.shape[0]
    n = y.shape[0]
    if m < 2 or n < 2:
        raise ValueError("KID requires at least two real and two generated samples.")

    mmd = (k_xx.sum() - np.trace(k_xx)) / (m * (m - 1))
    mmd += (k_yy.sum() - np.trace(k_yy)) / (n * (n - 1))
    mmd -= 2.0 * k_xy.mean()
    return float(mmd)


def _extract_features(images_or_features: ArrayLike | np.ndarray, feature_extractor: Optional[Callable[[np.ndarray], np.ndarray]]) -> np.ndarray:
    x = np.asarray(images_or_features)
    if x.ndim == 2 and feature_extractor is None:
        return x.astype(np.float64)

    images = _as_image_batch(images_or_features)
    if feature_extractor is not None:
        features = feature_extractor(images)
        if hasattr(features, "detach"):
            features = features.detach().cpu().numpy()
        return np.asarray(features, dtype=np.float64)

    small = np.stack([np.asarray(Image.fromarray((image * 255).astype(np.uint8)).resize((8, 8), Image.Resampling.BILINEAR), dtype=np.float32) / 255.0 for image in images])
    return small.reshape(small.shape[0], -1).astype(np.float64)


def clip_style_score(
    images: ArrayLike,
    text_prompts: str | Sequence[str],
    model: Optional[object] = None,
    preprocess: Optional[Callable[[Image.Image], object]] = None,
    tokenizer: Optional[Callable[[Sequence[str]], object]] = None,
    device: Optional[str] = None,
) -> float:
    """
    CLIP image-text cosine score. Higher is better.

    Pass a preloaded CLIP/open_clip model plus tokenizer/preprocess for fully
    controlled evaluation. If `model` is omitted, the function tries the
    optional `clip` package.
    """
    import torch

    prompts = [text_prompts] if isinstance(text_prompts, str) else list(text_prompts)
    if not prompts:
        raise ValueError("text_prompts must contain at least one prompt.")

    if model is None:
        try:
            import clip
        except ImportError as exc:
            raise ImportError("CLIP-style score requires a preloaded CLIP model or `pip install git+https://github.com/openai/CLIP.git`.") from exc

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        model, preprocess = clip.load("ViT-B/32", device=device)
        tokenizer = clip.tokenize

    if device is None:
        device = next(model.parameters()).device

    image_batch = _as_image_batch(images)
    pil_images = [Image.fromarray((image * 255).astype(np.uint8)) for image in image_batch]
    if preprocess is None:
        image_tensor = _torch_image_batch(image_batch, device=str(device), value_range="0_1")
    else:
        image_tensor = torch.stack([preprocess(image) for image in pil_images]).to(device)

    if tokenizer is None:
        raise ValueError("A tokenizer is required when using a custom CLIP model.")

    text_tokens = tokenizer(prompts).to(device)
    model.eval()
    with torch.no_grad():
        image_features = model.encode_image(image_tensor)
        text_features = model.encode_text(text_tokens)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        scores = image_features @ text_features.T

    return float(scores.max(dim=1).values.mean().detach().cpu().item())


def _binarize_mask(mask: ArrayLike, threshold: float = 0.5) -> np.ndarray:
    x = _to_numpy(mask)
    if x.ndim == 4 and x.shape[1] == 1:
        x = x[:, 0]
    elif x.ndim == 4 and x.shape[-1] == 1:
        x = x[:, :, :, 0]
    elif x.ndim == 3 and x.shape[0] == 1:
        x = x[0][None, ...]
    elif x.ndim == 3 and x.shape[-1] == 1:
        x = x[:, :, 0][None, ...]
    elif x.ndim == 2:
        x = x[None, ...]

    if x.ndim != 3:
        raise ValueError(f"Expected mask shape [H,W], [B,H,W], [B,1,H,W], or [B,H,W,1], got {x.shape}")

    if x.max(initial=0.0) > 2.0:
        x = x / 255.0

    return (x > threshold).astype(np.uint8)


def mask_consistency(mask_before: ArrayLike, mask_after: ArrayLike, threshold: float = 0.5, eps: float = 1e-7) -> float:
    """
    IoU between masks before and after style transfer. Higher is better.
    """
    before = _binarize_mask(mask_before, threshold)
    after = _binarize_mask(mask_after, threshold)
    if before.shape != after.shape:
        raise ValueError(f"Shape mismatch: before shape {before.shape}, after shape {after.shape}")

    scores = []
    for b, a in zip(before.astype(bool), after.astype(bool)):
        union = np.logical_or(b, a).sum()
        if union == 0:
            scores.append(1.0)
        else:
            scores.append(float((np.logical_and(b, a).sum() + eps) / (union + eps)))
    return float(np.mean(scores))


def boundary_consistency(
    mask_before: ArrayLike,
    mask_after: ArrayLike,
    threshold: float = 0.5,
    tolerance: int = 2,
    eps: float = 1e-7,
) -> Tuple[float, float, float]:
    """
    Boundary F-score between masks before and after style transfer.

    Returns:
        boundary_f_score, boundary_precision, boundary_recall
    """
    before = _binarize_mask(mask_before, threshold)
    after = _binarize_mask(mask_after, threshold)
    if before.shape != after.shape:
        raise ValueError(f"Shape mismatch: before shape {before.shape}, after shape {after.shape}")

    f_scores = []
    precision_scores = []
    recall_scores = []
    structure = ndimage.generate_binary_structure(2, 1)

    for b, a in zip(before, after):
        b_boundary = _mask_to_boundary(b)
        a_boundary = _mask_to_boundary(a)

        if b_boundary.sum() == 0 and a_boundary.sum() == 0:
            f_scores.append(1.0)
            precision_scores.append(1.0)
            recall_scores.append(1.0)
            continue
        if b_boundary.sum() == 0 or a_boundary.sum() == 0:
            f_scores.append(0.0)
            precision_scores.append(0.0)
            recall_scores.append(0.0)
            continue

        b_dilated = ndimage.binary_dilation(b_boundary, structure=structure, iterations=tolerance)
        a_dilated = ndimage.binary_dilation(a_boundary, structure=structure, iterations=tolerance)

        precision = (np.logical_and(a_boundary, b_dilated).sum() + eps) / (a_boundary.sum() + eps)
        recall = (np.logical_and(b_boundary, a_dilated).sum() + eps) / (b_boundary.sum() + eps)
        f_score = 2.0 * precision * recall / (precision + recall + eps)

        f_scores.append(float(f_score))
        precision_scores.append(float(precision))
        recall_scores.append(float(recall))

    return float(np.mean(f_scores)), float(np.mean(precision_scores)), float(np.mean(recall_scores))


def _mask_to_boundary(mask: np.ndarray) -> np.ndarray:
    mask = mask.astype(bool)
    if mask.sum() == 0:
        return np.zeros_like(mask, dtype=np.uint8)
    structure = ndimage.generate_binary_structure(2, 1)
    eroded = ndimage.binary_erosion(mask, structure=structure, border_value=0)
    return np.logical_xor(mask, eroded).astype(np.uint8)


def compute_all_metrics(
    generated: ArrayLike,
    reference: ArrayLike,
    real_distribution: Optional[ArrayLike] = None,
    generated_distribution: Optional[ArrayLike] = None,
    mask_before: Optional[ArrayLike] = None,
    mask_after: Optional[ArrayLike] = None,
    feature_extractor: Optional[Callable[[np.ndarray], np.ndarray]] = None,
) -> Dict[str, float]:
    """
    Compute lightweight deterministic metrics and optional distribution/mask metrics.
    """
    metrics = {
        "ssim": ssim_score(generated, reference),
        "edge_similarity": edge_similarity(generated, reference),
    }

    if real_distribution is not None and generated_distribution is not None:
        metrics["fid"] = fid_score(real_distribution, generated_distribution, feature_extractor=feature_extractor)
        metrics["kid"] = kid_score(real_distribution, generated_distribution, feature_extractor=feature_extractor)

    if mask_before is not None and mask_after is not None:
        bf, bp, br = boundary_consistency(mask_before, mask_after)
        metrics["mask_consistency"] = mask_consistency(mask_before, mask_after)
        metrics["boundary_consistency"] = bf
        metrics["boundary_precision"] = bp
        metrics["boundary_recall"] = br

    return metrics


if __name__ == "__main__":
    image_a = np.zeros((2, 64, 64, 3), dtype=np.float32)
    image_b = image_a.copy()
    image_b[:, 16:48, 16:48, :] = 1.0
    mask_a = np.zeros((2, 64, 64), dtype=np.uint8)
    mask_b = mask_a.copy()
    mask_b[:, 16:48, 16:48] = 1

    print(compute_all_metrics(image_a, image_b, mask_before=mask_a, mask_after=mask_b))

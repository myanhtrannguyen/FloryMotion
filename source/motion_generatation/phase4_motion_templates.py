#!/usr/bin/env python3
"""
phase4_motion_templates.py

Generate 12 coarse keyframes from:
- a cartoonized flower image,
- a flower mask,
- a 12-stage botanical CIG JSON file.

The script preserves the original background and only transforms the flower foreground.
These 12 generated images are intended as coarse visual guidance frames. They can be
optionally passed to an image-to-image or inpainting model for refinement.

Requirements:
    pip install pillow numpy opencv-python imageio

Example:
    python phase4_motion_templates.py \
      --image_path /path/to/cartoon_image.png \
      --mask_path /path/to/segmim_06765.jpg \
      --cig_path /path/to/001_pink_primrose.json \
      --output_dir outputs/phase4_keyframes \
      --image_size 512 \
      --make_gif
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple

import cv2
import imageio.v2 as imageio
import numpy as np
from PIL import Image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate 12 coarse flower motion keyframes from CIG."
    )
    parser.add_argument("--image_path", required=True, type=str)
    parser.add_argument("--mask_path", required=True, type=str)
    parser.add_argument("--cig_path", required=True, type=str)
    parser.add_argument("--output_dir", required=True, type=str)
    parser.add_argument("--image_size", default=512, type=int)
    parser.add_argument("--fps", default=8, type=int)
    parser.add_argument("--make_gif", action="store_true")
    parser.add_argument("--mask_dilate", default=1, type=int)
    return parser.parse_args()


def load_rgb_image(path: str | Path, size: int) -> np.ndarray:
    image = Image.open(path).convert("RGB")
    image = image.resize((size, size), Image.BICUBIC)
    return np.array(image).astype(np.uint8)


def load_flower_mask(path: str | Path, size: int, dilate_iter: int = 1) -> np.ndarray:
    """
    Load Oxford-style flower mask or normal binary mask.
    Oxford masks often preserve flower pixels and replace background with blue.
    """
    mask_img = Image.open(path).convert("RGB")
    mask_img = mask_img.resize((size, size), Image.NEAREST)
    mask_rgb = np.array(mask_img).astype(np.int16)

    r = mask_rgb[:, :, 0]
    g = mask_rgb[:, :, 1]
    b = mask_rgb[:, :, 2]

    blue_background = (b > 100) & (b > r + 25) & (b > g + 25)
    foreground = (~blue_background).astype(np.uint8)

    fg_ratio = foreground.mean()
    if fg_ratio > 0.95 or fg_ratio < 0.01:
        gray = np.array(Image.open(path).convert("L").resize((size, size), Image.NEAREST))
        foreground = (gray > 127).astype(np.uint8)

    if dilate_iter > 0:
        kernel = np.ones((3, 3), np.uint8)
        foreground = cv2.dilate(foreground, kernel, iterations=dilate_iter)

    return foreground.astype(np.float32)


def load_cig(path: str | Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        cig = json.load(f)
    if "stages" not in cig or len(cig["stages"]) != 12:
        raise ValueError("CIG must contain exactly 12 stages in the 'stages' field.")
    return cig


def parse_percent(value: Any, default: float = 50.0) -> float:
    text = str(value)
    match = re.search(r"(\d+(?:\.\d+)?)", text)
    return float(match.group(1)) if match else default


def estimate_center(mask: np.ndarray) -> Tuple[float, float]:
    ys, xs = np.where(mask > 0.5)
    if len(xs) == 0:
        h, w = mask.shape
        return w / 2.0, h / 2.0
    return float(xs.mean()), float(ys.mean())


def smoothstep(x: float) -> float:
    x = min(1.0, max(0.0, float(x)))
    return x * x * (3.0 - 2.0 * x)


def adjust_color_for_stage(fg: np.ndarray, mask: np.ndarray, stage: Dict[str, Any]) -> np.ndarray:
    """
    Coarse appearance guidance:
    - early bud stages: darker and greener
    - peak bloom stages: more vibrant
    - fading stages: desaturated and brownish
    """
    out = fg.astype(np.float32).copy()
    phase = str(stage.get("growth_phase", "")).lower()
    stage_id = int(stage.get("stage_id", 1))

    petal_desc = stage.get("petal_description", {})
    color_text = str(petal_desc.get("color_transition", "")).lower()

    if stage_id <= 3:
        out *= 0.85
        out[:, :, 1] = np.minimum(255, out[:, :, 1] * 1.08 + 8)

    if "peak" in phase or stage_id in [6, 7, 8]:
        out *= 1.06
        out[:, :, 0] = np.minimum(255, out[:, :, 0] * 1.06 + 3)

    if stage_id >= 9 or "fading" in color_text or "brown" in color_text or "dry" in color_text:
        gray = out.mean(axis=2, keepdims=True)
        desat_strength = min(0.70, 0.18 * (stage_id - 8))
        out = out * (1.0 - desat_strength) + gray * desat_strength

        brown = np.array([120, 82, 45], dtype=np.float32)
        brown_strength = min(0.50, 0.12 * (stage_id - 8))
        out = out * (1.0 - brown_strength) + brown * brown_strength

    return np.clip(out * mask[:, :, None], 0, 255).astype(np.uint8)


def affine_foreground(
    fg: np.ndarray,
    alpha: np.ndarray,
    center: Tuple[float, float],
    scale_x: float,
    scale_y: float,
    angle_deg: float = 0.0,
    shift_x: float = 0.0,
    shift_y: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray]:
    h, w = alpha.shape
    cx, cy = center

    angle = math.radians(angle_deg)
    cos_a = math.cos(angle)
    sin_a = math.sin(angle)

    m = np.array(
        [
            [scale_x * cos_a, -scale_y * sin_a, 0.0],
            [scale_x * sin_a,  scale_y * cos_a, 0.0],
        ],
        dtype=np.float32,
    )
    m[0, 2] = cx + shift_x - m[0, 0] * cx - m[0, 1] * cy
    m[1, 2] = cy + shift_y - m[1, 0] * cx - m[1, 1] * cy

    warped_fg = cv2.warpAffine(
        fg, m, (w, h), flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0)
    )
    warped_alpha = cv2.warpAffine(
        alpha, m, (w, h), flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT, borderValue=0
    )
    return warped_fg, np.clip(warped_alpha, 0.0, 1.0)


def radial_bloom_warp(
    fg: np.ndarray,
    alpha: np.ndarray,
    center: Tuple[float, float],
    amount: float,
) -> Tuple[np.ndarray, np.ndarray]:
    h, w = alpha.shape
    cx, cy = center

    yy, xx = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    dx = xx - cx
    dy = yy - cy
    radius = np.sqrt(dx * dx + dy * dy)
    radius_norm = radius / (radius.max() + 1e-6)

    factor = 1.0 + amount * (radius_norm ** 0.8)
    src_x = cx + dx / factor
    src_y = cy + dy / factor

    map_x = src_x.astype(np.float32)
    map_y = src_y.astype(np.float32)

    warped_fg = cv2.remap(fg, map_x, map_y, interpolation=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
    warped_alpha = cv2.remap(alpha, map_x, map_y, interpolation=cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return warped_fg, np.clip(warped_alpha, 0.0, 1.0)


def droop_warp(
    fg: np.ndarray,
    alpha: np.ndarray,
    center: Tuple[float, float],
    amount: float,
) -> Tuple[np.ndarray, np.ndarray]:
    h, w = alpha.shape
    cx, cy = center

    yy, xx = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    y_norm = np.clip((yy - cy) / max(1.0, h - cy), 0.0, 1.0)

    dy = amount * 45.0 * (y_norm ** 1.5)
    dx = -amount * 0.10 * (xx - cx) * y_norm

    map_x = (xx - dx).astype(np.float32)
    map_y = (yy - dy).astype(np.float32)

    warped_fg = cv2.remap(fg, map_x, map_y, interpolation=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
    warped_alpha = cv2.remap(alpha, map_x, map_y, interpolation=cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return warped_fg, np.clip(warped_alpha, 0.0, 1.0)


def compute_stage_transform(stage: Dict[str, Any]) -> Dict[str, float]:
    stage_id = int(stage.get("stage_id", 1))
    petal = stage.get("petal_description", {})
    opening = parse_percent(petal.get("opening_degree"), default=50.0)
    open_norm = opening / 100.0

    if stage_id <= 8:
        p = smoothstep(open_norm)
        scale_x = 0.45 + 0.65 * p
        scale_y = 0.95 + 0.05 * p
        radial_amount = -0.25 + 0.45 * p
        droop_amount = 0.0
        shift_y = 0.0
    else:
        decay = smoothstep((stage_id - 8) / 4.0)
        scale_x = 1.0 - 0.35 * decay
        scale_y = 1.0 - 0.20 * decay
        radial_amount = 0.15 - 0.35 * decay
        droop_amount = decay
        shift_y = 8.0 * decay

    angle = -4.0 * smoothstep((stage_id - 8) / 4.0) if stage_id >= 9 else 0.0

    return {
        "scale_x": float(scale_x),
        "scale_y": float(scale_y),
        "radial_amount": float(radial_amount),
        "droop_amount": float(droop_amount),
        "shift_y": float(shift_y),
        "angle": float(angle),
    }


def composite(background: np.ndarray, fg: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    alpha3 = np.clip(alpha[:, :, None], 0.0, 1.0)
    out = fg.astype(np.float32) * alpha3 + background.astype(np.float32) * (1.0 - alpha3)
    return np.clip(out, 0, 255).astype(np.uint8)


def build_stage_prompt(cig: Dict[str, Any], stage: Dict[str, Any]) -> Dict[str, str]:
    flower_name = cig.get("flower_name", "flower")
    petal = stage.get("petal_description", {})
    sepal = stage.get("sepal_description", {})
    stamen = stage.get("stamen_description", {})
    pistil = stage.get("pistil_description", {})

    positive = (
        f"Clean cartoon-style {flower_name}, same flower identity, same background, same camera view. "
        f"Stage {stage.get('stage_id')}: {stage.get('stage_name')}. "
        f"Growth phase: {stage.get('growth_phase')}. "
        f"Flower state: {stage.get('bud_or_flower_state')}. "
        f"Overall shape: {stage.get('overall_flower_shape')}. "
        f"Petals: visibility {petal.get('visibility')}, outer petals {petal.get('outer_petals')}, "
        f"inner petals {petal.get('inner_petals')}, shape {petal.get('shape')}, "
        f"orientation {petal.get('orientation')}, opening degree {petal.get('opening_degree')}, "
        f"curvature {petal.get('curvature')}, edge shape {petal.get('edge_shape')}, "
        f"texture {petal.get('texture')}, color {petal.get('color_transition')}. "
        f"Sepals: {sepal.get('visibility')}, {sepal.get('position')}, {sepal.get('orientation')}, color {sepal.get('color')}. "
        f"Stamen: {stamen.get('visibility')}, {stamen.get('position')}, {stamen.get('shape')}, color {stamen.get('color')}. "
        f"Pistil: {pistil.get('visibility')}, {pistil.get('position')}, {pistil.get('shape')}, color {pistil.get('color')}. "
        f"Flower center: {stage.get('flower_center_description')}. "
        f"Appearance guidance: {stage.get('appearance_guidance_for_generation')}. "
        f"Motion guidance: {stage.get('motion_guidance_for_animation')}. "
        "Preserve the existing cartoon style and background. Only refine the flower region."
    )

    negative = (
        f"{stage.get('negative_constraints', '')}, changed background, different camera angle, "
        "extra flowers, extra petals appearing from nowhere, rubber-like stretching, "
        "distorted flower, inconsistent cartoon style, blurry, low quality, artifacts"
    )

    return {
        "positive_prompt": " ".join(positive.split()),
        "negative_prompt": " ".join(negative.split()),
    }


def generate_keyframes(
    image_path: str | Path,
    mask_path: str | Path,
    cig_path: str | Path,
    output_dir: str | Path,
    image_size: int = 512,
    fps: int = 8,
    make_gif: bool = False,
    mask_dilate: int = 1,
) -> None:
    output_dir = Path(output_dir)
    frames_dir = output_dir / "frames"
    prompts_dir = output_dir / "prompts"
    meta_dir = output_dir / "metadata"

    frames_dir.mkdir(parents=True, exist_ok=True)
    prompts_dir.mkdir(parents=True, exist_ok=True)
    meta_dir.mkdir(parents=True, exist_ok=True)

    image = load_rgb_image(image_path, image_size)
    mask = load_flower_mask(mask_path, image_size, dilate_iter=mask_dilate)
    cig = load_cig(cig_path)

    flower_name = cig.get("flower_name", "flower")
    class_label = cig.get("class_label", "unknown")
    cig_id = Path(cig_path).stem

    center = estimate_center(mask)
    alpha = mask.astype(np.float32)
    fg_base = (image.astype(np.float32) * alpha[:, :, None]).astype(np.uint8)
    background = (image.astype(np.float32) * (1.0 - alpha[:, :, None])).astype(np.uint8)

    frame_paths: List[str] = []
    prompt_records: List[Dict[str, Any]] = []

    for stage in cig["stages"]:
        stage_id = int(stage["stage_id"])
        transform = compute_stage_transform(stage)

        fg_color = adjust_color_for_stage(fg_base, alpha, stage)

        warped_fg, warped_alpha = affine_foreground(
            fg_color,
            alpha,
            center=center,
            scale_x=transform["scale_x"],
            scale_y=transform["scale_y"],
            angle_deg=transform["angle"],
            shift_y=transform["shift_y"],
        )

        warped_fg, warped_alpha = radial_bloom_warp(
            warped_fg,
            warped_alpha,
            center=center,
            amount=transform["radial_amount"],
        )

        if transform["droop_amount"] > 0:
            warped_fg, warped_alpha = droop_warp(
                warped_fg,
                warped_alpha,
                center=center,
                amount=transform["droop_amount"],
            )

        frame = composite(background, warped_fg, warped_alpha)

        frame_name = f"{cig_id}_stage_{stage_id:02d}.png"
        frame_path = frames_dir / frame_name
        Image.fromarray(frame).save(frame_path)
        frame_paths.append(str(frame_path))

        prompt_record = {
            "cig_id": cig_id,
            "class_label": class_label,
            "flower_name": flower_name,
            "stage_id": stage_id,
            "stage_name": stage.get("stage_name"),
            "frame_range": stage.get("frame_range"),
            "coarse_frame_path": str(frame_path),
            "transform": transform,
            **build_stage_prompt(cig, stage),
        }
        prompt_records.append(prompt_record)

    prompts_path = prompts_dir / f"{cig_id}_stage_prompts.jsonl"
    with prompts_path.open("w", encoding="utf-8") as f:
        for record in prompt_records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    metadata = {
        "cig_id": cig_id,
        "class_label": class_label,
        "flower_name": flower_name,
        "source_image": str(image_path),
        "mask_path": str(mask_path),
        "cig_path": str(cig_path),
        "num_keyframes": len(frame_paths),
        "image_size": image_size,
        "center": {"x": center[0], "y": center[1]},
        "frames": frame_paths,
        "prompts_path": str(prompts_path),
        "note": (
            "These are 12 coarse motion-template keyframes. "
            "They are intended to be refined by an image-to-image or inpainting model."
        ),
    }

    metadata_path = meta_dir / f"{cig_id}_metadata.json"
    with metadata_path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    if make_gif:
        gif_path = output_dir / f"{cig_id}_coarse_preview.gif"
        frames_np = [imageio.imread(p) for p in frame_paths]
        imageio.mimsave(gif_path, frames_np, duration=1.0 / fps)
        metadata["coarse_gif_path"] = str(gif_path)
        with metadata_path.open("w", encoding="utf-8") as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)

    print(f"Saved {len(frame_paths)} coarse keyframes to: {frames_dir}")
    print(f"Saved stage prompts to: {prompts_path}")
    print(f"Saved metadata to: {metadata_path}")


def main() -> None:
    args = parse_args()
    generate_keyframes(
        image_path=args.image_path,
        mask_path=args.mask_path,
        cig_path=args.cig_path,
        output_dir=args.output_dir,
        image_size=args.image_size,
        fps=args.fps,
        make_gif=args.make_gif,
        mask_dilate=args.mask_dilate,
    )


if __name__ == "__main__":
    main()

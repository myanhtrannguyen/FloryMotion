from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from PIL import Image, ImageFilter


def denormalize(tensor: torch.Tensor) -> torch.Tensor:
    return ((tensor.detach().cpu() + 1.0) * 0.5).clamp(0.0, 1.0)


def tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
    image = denormalize(tensor)
    if image.ndim == 4:
        image = image[0]
    image_np = (image.numpy().transpose(1, 2, 0) * 255.0).astype(np.uint8)
    return Image.fromarray(image_np)


def save_image(tensor: torch.Tensor, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tensor_to_pil(tensor).save(path)


def save_image_grid(tensors: Iterable[torch.Tensor], path: str | Path, columns: int = 4) -> None:
    images = [tensor_to_pil(tensor) for tensor in tensors]
    if not images:
        return

    width, height = images[0].size
    rows = int(np.ceil(len(images) / columns))
    grid = Image.new("RGB", (columns * width, rows * height))
    for index, image in enumerate(images):
        x = (index % columns) * width
        y = (index // columns) * height
        grid.paste(image, (x, y))

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    grid.save(path)


def style_edge_fake(style_batch: torch.Tensor) -> torch.Tensor:
    """
    Build CartoonGAN edge-promoting fake samples from style images online.

    The original Lua repo expects a train_B_edge folder. This project only uses
    `{domain}/style`, so edge maps are generated from style images at runtime.
    """
    device = style_batch.device
    images = denormalize(style_batch)
    edge_tensors = []
    for image in images:
        image_np = (image.numpy().transpose(1, 2, 0) * 255.0).astype(np.uint8)
        pil = Image.fromarray(image_np).convert("L")
        pil = pil.filter(ImageFilter.FIND_EDGES).filter(ImageFilter.GaussianBlur(radius=1.0))
        edge_np = np.asarray(pil, dtype=np.float32) / 255.0
        edge_rgb = np.repeat(edge_np[:, :, None], 3, axis=2)
        edge_tensors.append(torch.from_numpy(edge_rgb.transpose(2, 0, 1)))

    edge = torch.stack(edge_tensors).to(device=device, dtype=style_batch.dtype)
    return edge * 2.0 - 1.0


def total_variation_loss(image: torch.Tensor) -> torch.Tensor:
    loss_h = torch.mean(torch.abs(image[:, :, 1:, :] - image[:, :, :-1, :]))
    loss_w = torch.mean(torch.abs(image[:, :, :, 1:] - image[:, :, :, :-1]))
    return loss_h + loss_w


def save_json(data: dict, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

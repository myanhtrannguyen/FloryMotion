from __future__ import annotations

import argparse
import pathlib
import sys
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from source.image_style_tranform.dataset import (
    CartoonGANEvalDataset,
    image_to_tensor,
    list_image_files,
    resize_image,
)
from source.image_style_tranform.metrics import (
    boundary_consistency,
    clip_style_score,
    edge_similarity,
    fid_score,
    kid_score,
    lpips_distance,
    mask_consistency,
    ssim_score,
)
from source.image_style_tranform.models.cartoon_gan.model import CartoonGenerator
from source.image_style_tranform.models.cartoon_gan.utils import save_image, save_json


def tensor_batch_to_numpy(batch: torch.Tensor) -> np.ndarray:
    batch = ((batch.detach().cpu() + 1.0) * 0.5).clamp(0.0, 1.0)
    return batch.numpy().transpose(0, 2, 3, 1)


def load_checkpoint(checkpoint_path: Path, device: torch.device) -> dict:
    original_posix_path = pathlib.PosixPath
    if sys.platform.startswith("win"):
        pathlib.PosixPath = pathlib.WindowsPath

    try:
        try:
            return torch.load(checkpoint_path, map_location=device, weights_only=False)
        except TypeError:
            return torch.load(checkpoint_path, map_location=device)
    finally:
        pathlib.PosixPath = original_posix_path


def load_generator(checkpoint_path: Path, device: torch.device, generator_blocks: int = 4) -> CartoonGenerator:
    checkpoint = load_checkpoint(checkpoint_path, device)
    args = checkpoint.get("args", {})
    blocks = int(args.get("generator_blocks", generator_blocks))
    generator = CartoonGenerator(num_residual_blocks=blocks)
    generator.load_state_dict(checkpoint["generator"])
    generator.to(device)
    generator.eval()
    return generator


def load_style_images(
    data_root: Path,
    domain: str,
    image_size: int,
    max_samples: int,
) -> np.ndarray:
    paths = list_image_files(data_root / domain / "style")[:max_samples]
    tensors: List[torch.Tensor] = []
    for path in paths:
        image = Image.open(path).convert("RGB")
        image = resize_image(image, image_size)
        tensors.append(image_to_tensor(image, normalize=True))
    if not tensors:
        raise ValueError(f"No style images found for domain={domain!r}")
    return tensor_batch_to_numpy(torch.stack(tensors))


def load_label_maps(
    data_root: Path,
    image_names: list[str],
    image_size: int,
    split: str,
    mask_dir: Optional[Path],
) -> Optional[np.ndarray]:
    mask_dir_from_arg = mask_dir is not None
    if mask_dir is None and split == "test":
        mask_dir = data_root / "test" / "label_map"
    if mask_dir is None:
        return None
    if mask_dir_from_arg and not mask_dir.is_absolute():
        mask_dir = data_root / mask_dir
    if not mask_dir.exists():
        return None

    masks = []
    for image_name in image_names:
        mask_path = mask_dir / f"{Path(image_name).stem}.png"
        if not mask_path.exists():
            return None
        mask = Image.open(mask_path).convert("L")
        mask = mask.resize((image_size, image_size), Image.Resampling.NEAREST)
        masks.append(np.asarray(mask, dtype=np.float32) / 255.0)

    return np.stack(masks, axis=0)


def safe_metric(name: str, fn, metrics: dict) -> None:
    try:
        metrics[name] = fn()
    except Exception as exc:
        metrics[name] = None
        metrics[f"{name}_error"] = str(exc)


@torch.no_grad()
def evaluate(args: argparse.Namespace) -> dict:
    device = torch.device(args.device)
    generator = load_generator(args.checkpoint, device=device)
    dataset = CartoonGANEvalDataset(
        data_root=args.data_root,
        mode=args.split,
        image_size=args.image_size,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    output_dir = args.output_root / args.split
    generated_batches = []
    input_batches = []
    image_names = []

    for photos, meta in loader:
        photos = photos.to(device)
        generated = generator(photos)
        generated_batches.append(tensor_batch_to_numpy(generated))
        input_batches.append(tensor_batch_to_numpy(photos))

        names = meta["image_name"]
        image_names.extend(list(names))
        for image_tensor, name in zip(generated, names):
            save_image(image_tensor, output_dir / "image" / name)

    generated_np = np.concatenate(generated_batches, axis=0)
    input_np = np.concatenate(input_batches, axis=0)

    max_samples = min(args.max_metric_samples, generated_np.shape[0])
    style_np = load_style_images(
        data_root=args.data_root,
        domain=args.domain,
        image_size=args.image_size,
        max_samples=max_samples,
    )
    generated_for_distribution = generated_np[:max_samples]
    input_for_distribution = input_np[:max_samples]

    metrics = {
        "domain": args.domain,
        "split": args.split,
        "num_images": int(generated_np.shape[0]),
        "ssim": ssim_score(generated_np, input_np),
        "edge_similarity": edge_similarity(generated_np, input_np),
        "fid": fid_score(style_np, generated_for_distribution),
        "kid": kid_score(style_np, generated_for_distribution),
        "output_dir": str(output_dir),
        "checkpoint": str(args.checkpoint),
        "metric_reference": {
            "ssim": "input_photo_vs_generated",
            "lpips": "input_photo_vs_generated",
            "edge_similarity": "input_photo_vs_generated",
            "fid": "style_images_vs_generated",
            "kid": "style_images_vs_generated",
            "clip_style_score": "generated_vs_text_prompts",
            "mask_consistency": "test_label_map_before_vs_after",
            "boundary_consistency": "test_label_map_before_vs_after",
        },
    }

    safe_metric(
        "lpips",
        lambda: lpips_distance(generated_for_distribution, input_for_distribution, device=str(device)),
        metrics,
    )
    safe_metric(
        "clip_style_score",
        lambda: clip_style_score(generated_for_distribution, args.clip_prompts, device=str(device)),
        metrics,
    )

    label_maps = load_label_maps(
        data_root=args.data_root,
        image_names=image_names,
        image_size=args.image_size,
        split=args.split,
        mask_dir=args.mask_dir,
    )
    if label_maps is not None:
        metrics["mask_consistency"] = mask_consistency(label_maps, label_maps)
        boundary_f, boundary_precision, boundary_recall = boundary_consistency(label_maps, label_maps)
        metrics["boundary_consistency"] = boundary_f
        metrics["boundary_precision"] = boundary_precision
        metrics["boundary_recall"] = boundary_recall
        metrics["mask_source"] = str(args.mask_dir or (args.data_root / "test" / "label_map"))
    else:
        metrics["mask_consistency"] = None
        metrics["boundary_consistency"] = None
        metrics["boundary_precision"] = None
        metrics["boundary_recall"] = None
        metrics["mask_consistency_error"] = "No mask_dir found. Pass --mask-dir with masks named like input images."

    save_json(metrics, output_dir / "metrics.json")
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate CartoonGAN checkpoint on animeGAN val/test photos.")
    parser.add_argument("--data-root", type=Path, default=Path("data/animeGAN"))
    parser.add_argument("--domain", choices=["Hayao", "Shinkai"], default="Hayao")
    parser.add_argument("--split", choices=["val", "test"], default="test")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=Path("source/image_style_tranform/models/cartoon_gan/Hayao/eval_outputs"))
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-metric-samples", type=int, default=128)
    parser.add_argument("--mask-dir", type=Path, default=None)
    parser.add_argument(
        "--clip-prompts",
        nargs="+",
        default=["anime style", "cartoon style"],
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main() -> None:
    metrics = evaluate(parse_args())
    for key, value in metrics.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from source.image_style_tranform.metrics import edge_similarity, fid_score, kid_score, ssim_score
from source.image_style_tranform.models.cartoon_gan.utils import save_json


def load_generated_images(path, image_size=256):
    images = []
    paths = sorted(Path(path).glob("*_fake_B.png"))

    for p in paths:
        img = Image.open(p).convert("RGB")
        img = img.resize((image_size, image_size))
        img = np.array(img).astype(np.float32) / 255.0
        images.append(img)

    return np.stack(images)


def load_real_images(path, image_size=256):
    images = []
    paths = sorted(Path(path).glob("*.*"))

    for p in paths:
        img = Image.open(p).convert("RGB")
        img = img.resize((image_size, image_size))
        img = np.array(img).astype(np.float32) / 255.0
        images.append(img)

    return np.stack(images)


@torch.no_grad()
def evaluate(args):
    output_dir = args.output_root / args.domain / args.split
    output_dir.mkdir(parents=True, exist_ok=True)

    result_dir = ROOT / "pytorch-CycleGAN-and-pix2pix" / "results" / f"{args.domain.lower()}_cyclegan" / "test_latest" / "images"

    generated = load_generated_images(result_dir)
    reference = load_real_images(args.data_root / "test" / "label_map")
    real_style_distribution = load_real_images(args.data_root / args.domain / "style")

    metrics = {
        "ssim_generated_reference": ssim_score(generated, reference),
        "edge_similarity_generated_reference": edge_similarity(generated, reference),
        "fid_generated_style": fid_score(real_style_distribution, generated),
        "kid_generated_style": kid_score(real_style_distribution, generated),
    }

    save_json(metrics, output_dir / "metrics.json")
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate CycleGAN checkpoint on animeGAN val/test photos.")
    parser.add_argument("--data-root", type=Path, default=Path("data/animeGAN"))
    parser.add_argument("--domain", choices=["Hayao", "Shinkai"], default="Hayao")
    parser.add_argument("--split", choices=["val", "test"], default="test")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=Path("source/image_style_tranform/models/cycle_gan/eval_outputs"))
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-metric-samples", type=int, default=128)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main() -> None:
    metrics = evaluate(parse_args())
    for key, value in metrics.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()

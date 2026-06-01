from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from source.image_segmentation.dataset import IMAGENET_MEAN, IMAGENET_STD
from models.swinunet_swin_tiny import SwinUNetTiny


def preprocess(image_path: Path, image_size: int) -> torch.Tensor:
    image = Image.open(image_path).convert("RGB")
    image = image.resize((image_size, image_size), Image.Resampling.BILINEAR)
    image_np = np.asarray(image, dtype=np.float32) / 255.0
    image_np = (image_np - IMAGENET_MEAN) / IMAGENET_STD
    return torch.from_numpy(image_np.transpose(2, 0, 1)).float().unsqueeze(0)


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict a binary flower mask.")
    parser.add_argument("image", help="Path to input flower image.")
    parser.add_argument(
        "--config",
        default="source/image_segmentation/models/unet_efficientnet_b0/models/kaggle/config.yaml",
    )
    parser.add_argument(
        "--checkpoint",
        default="source/image_segmentation/models/unet_efficientnet_b0/models/kaggle/best_model.pth",
    )
    parser.add_argument("--output", default=None)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    with Path(args.config).open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    device = torch.device(args.device)
    model = SwinUNetTiny().to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    image_path = Path(args.image)
    image_tensor = preprocess(image_path, int(config["image_size"])).to(device)
    with torch.no_grad():
        prob = torch.sigmoid(model(image_tensor))[0, 0].cpu().numpy()

    mask = (prob > args.threshold).astype(np.uint8) * 255
    output = Path(args.output) if args.output else image_path.with_name(f"{image_path.stem}_mask.png")
    Image.fromarray(mask).save(output)
    print(output)


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from source.image_segmentation.dataset import IMAGENET_MEAN, IMAGENET_STD
from models.u2net_small import U2NetSmall


def preprocess(image_path: Path, image_size: int) -> torch.Tensor:
    image = Image.open(image_path).convert("RGB")
    image = image.resize((image_size, image_size), Image.Resampling.BILINEAR)
    image_np = np.asarray(image, dtype=np.float32) / 255.0
    image_np = (image_np - IMAGENET_MEAN) / IMAGENET_STD
    return torch.from_numpy(image_np.transpose(2, 0, 1)).float().unsqueeze(0)


def load_gt_mask(mask_path: Path, size: int) -> np.ndarray:
    mask = Image.open(mask_path).convert("RGB")
    mask = mask.resize((size, size), Image.Resampling.NEAREST)

    mask_np = np.asarray(mask)
    r = mask_np[:, :, 0].astype(np.int16)
    g = mask_np[:, :, 1].astype(np.int16)
    b = mask_np[:, :, 2].astype(np.int16)

    background = (b > 100) & (b > r + 25) & (b > g + 25)
    binary = (~background).astype(np.uint8)

    return binary


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict a binary flower mask.")
    parser.add_argument("image", help="Path to input flower image.")
    parser.add_argument(
        "--config",
        default="source/image_segmentation/models/u2net_small/models/config.yaml",
    )
    parser.add_argument(
        "--checkpoint",
        default="source/image_segmentation/models/u2net_small/models/best_model.pth",
    )
    parser.add_argument("--output", default=None)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    with Path(args.config).open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    device = torch.device(args.device)
    model = U2NetSmall().to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    image_path = Path(args.image)
    image_tensor = preprocess(image_path, int(config["image_size"])).to(device)
    with torch.no_grad():
        outputs = model(image_tensor)
        prob = torch.sigmoid(outputs[0])[0, 0].cpu().numpy()
    
    image_name = image_path.name
    image_id = image_name.replace("image_", "").replace(".jpg", "")
    gt_path = (image_path.parent.parent / "masks" / f"segmim_{image_id}.jpg")
    gt_mask = load_gt_mask(gt_path, int(config["image_size"]))

    pred_mask = (prob > args.threshold).astype(np.uint8)
    mask_vis = pred_mask * 255

    tp = (pred_mask == 1) & (gt_mask == 1)
    fp = (pred_mask == 1) & (gt_mask == 0)
    fn = (pred_mask == 0) & (gt_mask == 1)

    error_map = np.zeros((*pred_mask.shape, 3), dtype=np.uint8)
    error_map[tp] = [0, 255, 0]   # green = correct
    error_map[fp] = [255, 0, 0]   # red = over-prediction
    error_map[fn] = [0, 0, 255]   # blue = missing region

    out_dir = Path("source/image_segmentation/models/u2net_small/results") / image_id
    out_dir.mkdir(parents=True, exist_ok=True)

    stem = image_path.stem

    Image.open(image_path).save(out_dir / f"{stem}_input.png")
    Image.fromarray(gt_mask * 255).save(out_dir / f"{stem}_gt.png")
    Image.fromarray(pred_mask * 255).save(out_dir / f"{stem}_pred.png")
    Image.fromarray(error_map).save(out_dir / f"{stem}_error.png")


if __name__ == "__main__":
    main()

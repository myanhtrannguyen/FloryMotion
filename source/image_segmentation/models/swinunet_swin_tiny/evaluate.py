from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from source.image_segmentation.dataset import OxfordFlowersSegmentation
from source.image_segmentation.losses import BCEDiceLoss
from train import run_epoch
from models.swinunet_swin_tiny import SwinUNetTiny


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Swin-UNet checkpoint.")
    parser.add_argument(
        "--config",
        default="source/image_segmentation/swinunet_swin_tiny/config.yaml",
    )
    parser.add_argument(
        "--checkpoint",
        default="source/image_segmentation/swinunet_swin_tiny/best_model.pth",
    )
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    with Path(args.config).open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    device = torch.device(args.device)
    dataset = OxfordFlowersSegmentation(
        config["data_root"],
        split=args.split,
        image_size=int(config["image_size"]),
        augment=False,
    )
    loader = DataLoader(
        dataset,
        batch_size=int(config["batch_size"]),
        shuffle=False,
        num_workers=int(config["num_workers"]),
        pin_memory=device.type == "cuda",
    )

    model = SwinUNetTiny().to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    metrics = run_epoch(model, loader, BCEDiceLoss(float(config["dice_weight"])), device)
    output_path = Path(config["output_dir"]) / f"{args.split}_metrics.json"
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
from collections import defaultdict
import json
import sys
from pathlib import Path
from typing import Dict

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[3]
SOURCE_ROOT = Path(__file__).resolve().parents[2]
for path in (REPO_ROOT, SOURCE_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from dataset import OxfordFlowersSegmentation
from losses import BCEDiceLoss
from metric import compute_all_metrics, dice_score, hd95
from models.linknet_efficientnet_b0 import LinkNetEfficientNetB0


def evaluate_with_category_dice(
    model: torch.nn.Module,
    loader: DataLoader,
    criterion: torch.nn.Module,
    device: torch.device,
    top_k: int = 5,
) -> Dict[str, object]:
    model.eval()
    total_loss = 0.0
    metric_sums: Dict[str, float] = {}
    category_dice_sums: dict[str, float] = defaultdict(float)
    category_counts: dict[str, int] = defaultdict(int)
    sample_hd95_values: list[float] = []
    non_empty_gt_hd95_values: list[float] = []
    empty_gt_count = 0
    empty_pred_count = 0
    seen = 0

    with torch.no_grad():
        for images, masks, meta in loader:
            images = images.to(device)
            masks = masks.to(device)
            batch_size = images.size(0)

            logits = model(images)
            loss = criterion(logits, masks)
            probs = torch.sigmoid(logits).cpu()
            masks_cpu = masks.cpu()

            total_loss += float(loss.item()) * batch_size
            batch_metrics = compute_all_metrics(probs, masks_cpu)
            for name, value in batch_metrics.items():
                metric_sums[name] = metric_sums.get(name, 0.0) + value * batch_size
            seen += batch_size

            category_names = meta["name_cat"]
            for index, category_name in enumerate(category_names):
                pred_sample = probs[index : index + 1]
                target_sample = masks_cpu[index : index + 1]
                sample_dice = dice_score(pred_sample, target_sample)
                sample_hd95 = hd95(pred_sample, target_sample)
                target_is_empty = bool((target_sample > 0.5).sum().item() == 0)
                pred_is_empty = bool((pred_sample > 0.5).sum().item() == 0)

                sample_hd95_values.append(sample_hd95)
                if target_is_empty:
                    empty_gt_count += 1
                else:
                    non_empty_gt_hd95_values.append(sample_hd95)
                if pred_is_empty:
                    empty_pred_count += 1

                category_dice_sums[str(category_name)] += sample_dice
                category_counts[str(category_name)] += 1

    metrics: Dict[str, object] = {
        name: value / seen for name, value in metric_sums.items()
    }
    metrics["loss"] = total_loss / len(loader.dataset)
    metrics["hd95_median"] = float(np.median(sample_hd95_values))
    metrics["hd95_p95"] = float(np.percentile(sample_hd95_values, 95))
    metrics["hd95_non_empty_gt"] = (
        float(np.mean(non_empty_gt_hd95_values)) if non_empty_gt_hd95_values else None
    )
    metrics["empty_gt_count"] = empty_gt_count
    metrics["empty_pred_count"] = empty_pred_count

    category_summary = [
        {
            "name_cat": name,
            "mean_dice": category_dice_sums[name] / category_counts[name],
            "count": category_counts[name],
        }
        for name in category_dice_sums
    ]
    category_summary.sort(key=lambda item: (-float(item["mean_dice"]), str(item["name_cat"])))

    metrics["top5_categories_by_dice"] = category_summary[:top_k]
    metrics["per_category_dice"] = category_summary
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate LinkNet EfficientNet-B0 checkpoint.")
    parser.add_argument(
        "--config",
        default="source/image_segmentation/models/linknet_efficientnet_b0/models/kaggle/config.yaml",
    )
    parser.add_argument(
        "--checkpoint",
        default="source/image_segmentation/models/linknet_efficientnet_b0/models/kaggle/best_model.pth",
    )
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--top-k", type=int, default=5)
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
    print(
        f"Dataset {args.split}: {len(dataset)} samples "
        f"(dropped empty masks: {len(dataset.dropped_empty_masks)})",
        flush=True,
    )
    loader = DataLoader(
        dataset,
        batch_size=int(config["batch_size"]),
        shuffle=False,
        num_workers=int(config["num_workers"]),
        pin_memory=device.type == "cuda",
    )

    model = LinkNetEfficientNetB0(num_classes=1).to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    metrics = evaluate_with_category_dice(
        model,
        loader,
        BCEDiceLoss(float(config["dice_weight"])),
        device,
        top_k=args.top_k,
    )
    output_path = Path(config["output_dir"]) / f"{args.split}_metrics.json"
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()

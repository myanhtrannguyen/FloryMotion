from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Dict

import numpy as np
import torch
import yaml
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from source.image_segmentation.dataset import OxfordFlowersSegmentation
from source.image_segmentation.losses import BCEDiceLoss
from source.image_segmentation.metric import compute_all_metrics
from models.swinunet_swin_tiny import SwinUNetTiny


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_config(path: str | Path) -> Dict[str, object]:
    with Path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def run_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    criterion: torch.nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
    epoch: int | None = None,
    phase: str | None = None,
    log_interval: int = 25,
) -> Dict[str, float]:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    metric_sums: Dict[str, float] = {}
    seen = 0
    total_batches = len(loader)

    for batch_idx, (images, masks, _) in enumerate(loader, start=1):
        images = images.to(device)
        masks = masks.to(device)
        batch_size = images.size(0)

        with torch.set_grad_enabled(training):
            logits = model(images)
            loss = criterion(logits, masks)

            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

        total_loss += float(loss.item()) * batch_size
        probs = torch.sigmoid(logits).detach().cpu()
        batch_metrics = compute_all_metrics(probs, masks.detach().cpu())
        for name, value in batch_metrics.items():
            metric_sums[name] = metric_sums.get(name, 0.0) + value * batch_size
        seen += batch_size

        if log_interval > 0 and (
            batch_idx == 1 or batch_idx % log_interval == 0 or batch_idx == total_batches
        ):
            prefix = []
            if epoch is not None:
                prefix.append(f"epoch={epoch}")
            if phase is not None:
                prefix.append(f"phase={phase}")
            prefix.extend(
                [
                    f"batch={batch_idx}/{total_batches}",
                    f"loss={total_loss / seen:.4f}",
                    f"dice={metric_sums['dice'] / seen:.4f}",
                    f"iou={metric_sums['iou'] / seen:.4f}",
                ]
            )
            print(" | ".join(prefix), flush=True)

    metrics = {name: value / seen for name, value in metric_sums.items()}
    metrics["loss"] = total_loss / len(loader.dataset)
    return metrics


def save_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    metrics: Dict[str, float],
    config: Dict[str, object],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch,
            "metrics": metrics,
            "config": config,
        },
        path,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Swin-UNet with Swin-Tiny backbone.")
    parser.add_argument(
        "--config",
        default="source/image_segmentation/swinunet_swin_tiny/models/config.yaml",
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--resume", default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    set_seed(int(config["seed"]))

    output_dir = Path(str(config["output_dir"]))
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "config.yaml").open("w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, sort_keys=False)

    device = torch.device(args.device)
    print(f"Using device: {device}")

    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    train_dataset = OxfordFlowersSegmentation(
        config["data_root"],
        split="train",
        image_size=int(config["image_size"]),
        augment=True,
    )
    val_dataset = OxfordFlowersSegmentation(
        config["data_root"],
        split="val",
        image_size=int(config["image_size"]),
        augment=False,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=int(config["batch_size"]),
        shuffle=True,
        num_workers=int(config["num_workers"]),
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=int(config["batch_size"]),
        shuffle=False,
        num_workers=int(config["num_workers"]),
        pin_memory=device.type == "cuda",
    )

    model = SwinUNetTiny().to(device)

    criterion = BCEDiceLoss(dice_weight=float(config["dice_weight"]))
    optimizer = AdamW(
        model.parameters(),
        lr=float(config["learning_rate"]),
        weight_decay=float(config["weight_decay"]),
    )
    scheduler = ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=3)
    
    start_epoch = 1
    best_dice = -1.0
    epochs_without_improvement = 0
    history = []

    history_file = output_dir / "history.json"

    if history_file.exists():
        with history_file.open("r", encoding="utf-8") as f:
            history = json.load(f)

        last_history_epoch = history[-1]["epoch"]
    else:
        history = []
        last_history_epoch = 0

    if args.resume:
        if history_file.exists():
            with history_file.open("r", encoding="utf-8") as f:
                history = json.load(f)

        checkpoint = torch.load(args.resume, map_location=device)

        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        start_epoch = max(
            checkpoint["epoch"] + 1,
            last_history_epoch + 1,
        )
        best_dice = checkpoint["metrics"]["dice"]

        print(f"Resuming from epoch {start_epoch}")
        print(f"Best Dice so far: {best_dice:.4f}")

    print("Starting training...")
    for epoch in range(start_epoch, int(config["epochs"]) + 1):
        print(f"\nStarting epoch {epoch}/{int(config['epochs'])}", flush=True)
        train_metrics = run_epoch(
            model,
            train_loader,
            criterion,
            device,
            optimizer,
            epoch=epoch,
            phase="train",
            log_interval=int(config["log_interval"]),
        )
        val_metrics = run_epoch(
            model,
            val_loader,
            criterion,
            device,
            epoch=epoch,
            phase="val",
            log_interval=int(config["log_interval"]),
        )
        scheduler.step(val_metrics["dice"])

        record = {"epoch": epoch, "train": train_metrics, "val": val_metrics}
        history.append(record)
        print("Epoch summary:", flush=True)
        print(json.dumps(record, indent=2), flush=True)

        if val_metrics["dice"] > best_dice:
            best_dice = val_metrics["dice"]
            epochs_without_improvement = 0
            save_checkpoint(
                output_dir / "best_model.pth",
                model,
                optimizer,
                epoch,
                val_metrics,
                config,
            )
        else:
            epochs_without_improvement += 1

        with (output_dir / "history.json").open("w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)

        if epochs_without_improvement >= int(config["early_stopping_patience"]):
            break


if __name__ == "__main__":
    main()

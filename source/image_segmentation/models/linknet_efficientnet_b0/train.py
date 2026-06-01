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

REPO_ROOT = Path(__file__).resolve().parents[3]
SOURCE_ROOT = Path(__file__).resolve().parents[2]
for path in (REPO_ROOT, SOURCE_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from dataset import OxfordFlowersSegmentation
from losses import BCEDiceLoss
from metric import compute_all_metrics
from models.linknet_efficientnet_b0 import LinkNetEfficientNetB0


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
    scheduler: ReduceLROnPlateau | None = None,
) -> None:
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
        "metrics": metrics,
        "config": config,
    }
    if scheduler is not None:
        checkpoint["scheduler_state_dict"] = scheduler.state_dict()

    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, path)


def load_history(history_path: Path) -> list[Dict[str, object]]:
    if not history_path.exists():
        return []

    with history_path.open("r", encoding="utf-8") as f:
        history = json.load(f)

    if not isinstance(history, list):
        raise ValueError(f"Expected list history in {history_path}")
    return history


def get_best_dice(history: list[Dict[str, object]]) -> float:
    best_dice = -1.0
    for record in history:
        val_metrics = record.get("val", {})
        if isinstance(val_metrics, dict) and "dice" in val_metrics:
            best_dice = max(best_dice, float(val_metrics["dice"]))
    return best_dice


def get_epochs_without_improvement(history: list[Dict[str, object]]) -> int:
    best_dice = -1.0
    epochs_without_improvement = 0
    for record in history:
        val_metrics = record.get("val", {})
        if not isinstance(val_metrics, dict) or "dice" not in val_metrics:
            continue
        dice = float(val_metrics["dice"])
        if dice > best_dice:
            best_dice = dice
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
    return epochs_without_improvement


def resume_if_available(
    output_dir: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: ReduceLROnPlateau,
    device: torch.device,
) -> tuple[list[Dict[str, object]], int, float, int]:
    history_path = output_dir / "history.json"
    history = load_history(history_path)
    best_dice = get_best_dice(history)
    epochs_without_improvement = get_epochs_without_improvement(history)
    start_epoch = int(history[-1]["epoch"]) + 1 if history else 1

    checkpoint_path = output_dir / "last_model.pth"
    if not checkpoint_path.exists():
        checkpoint_path = output_dir / "best_model.pth"

    if history and checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        if "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if "scheduler_state_dict" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        checkpoint_epoch = int(checkpoint.get("epoch", start_epoch - 1))
        print(
            f"Resuming from {checkpoint_path} "
            f"(checkpoint epoch {checkpoint_epoch}, next epoch {start_epoch})",
            flush=True,
        )
    elif history:
        raise FileNotFoundError(
            f"Found {history_path} but no checkpoint in {output_dir}. "
            "Cannot resume safely without saved model weights."
        )

    return history, start_epoch, best_dice, epochs_without_improvement


def main() -> None:
    parser = argparse.ArgumentParser(description="Train LinkNet with EfficientNet-B0 backbone.")
    parser.add_argument(
        "--config",
        default="source/image_segmentation/models/linknet_efficientnet_b0/models/local/config.yaml",
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    config = load_config(args.config)
    set_seed(int(config["seed"]))

    output_dir = Path(str(config["output_dir"]))
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "config.yaml").open("w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, sort_keys=False)

    device = torch.device(args.device)
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
    print(
        f"Dataset train: {len(train_dataset)} samples "
        f"(dropped empty masks: {len(train_dataset.dropped_empty_masks)})",
        flush=True,
    )
    print(
        f"Dataset val: {len(val_dataset)} samples "
        f"(dropped empty masks: {len(val_dataset.dropped_empty_masks)})",
        flush=True,
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

    model = LinkNetEfficientNetB0(num_classes=1).to(device)
    encoder_weights = config.get("encoder_weights")
    if encoder_weights:
        model.load_encoder_weights(str(encoder_weights), strict=False)

    criterion = BCEDiceLoss(dice_weight=float(config["dice_weight"]))
    optimizer = AdamW(
        model.parameters(),
        lr=float(config["learning_rate"]),
        weight_decay=float(config["weight_decay"]),
    )
    scheduler = ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=3)

    history, start_epoch, best_dice, epochs_without_improvement = resume_if_available(
        output_dir,
        model,
        optimizer,
        scheduler,
        device,
    )

    if start_epoch > int(config["epochs"]):
        print(
            f"Training already reached epoch {start_epoch - 1}. "
            f"Increase epochs above {config['epochs']} to continue.",
            flush=True,
        )
        return

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

        save_checkpoint(
            output_dir / "last_model.pth",
            model,
            optimizer,
            epoch,
            val_metrics,
            config,
            scheduler,
        )

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
                scheduler,
            )
        else:
            epochs_without_improvement += 1

        with (output_dir / "history.json").open("w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)

        if epochs_without_improvement >= int(config["early_stopping_patience"]):
            break


if __name__ == "__main__":
    main()

import os
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from tqdm import tqdm

from model_resnet34_unet import ResNet34UNet
from dataset import OxfordFlowerSegmentationDataset
from loss import BCEDiceLoss
from metrics import compute_all_metrics


def train_one_epoch(model, dataloader, optimizer, criterion, device):
    model.train()

    running_loss = 0.0
    running_dice = 0.0
    running_iou = 0.0
    running_precision = 0.0

    for images, masks in dataloader:
        images = images.to(device)
        masks = masks.to(device)

        optimizer.zero_grad()

        logits = model(images)
        loss = criterion(logits, masks)

        loss.backward()
        optimizer.step()

        batch_metrics = compute_all_metrics(
            pred=logits,
            target=masks,
            threshold=0.5,
            boundary_tolerance=2,
        )

        running_loss += loss.item()
        running_dice += batch_metrics["dice"]
        running_iou += batch_metrics["iou"]
        running_precision += batch_metrics["precision"]

    n = len(dataloader)

    return {
        "loss": running_loss / n,
        "dice": running_dice / n,
        "iou": running_iou / n,
        "precision": running_precision / n,
    }


@torch.no_grad()
def validate(model, dataloader, criterion, device):
    model.eval()

    running_loss = 0.0
    running_dice = 0.0
    running_iou = 0.0
    running_precision = 0.0
    running_hd95 = 0.0
    running_boundary_f = 0.0
    running_boundary_precision = 0.0
    running_boundary_recall = 0.0

    for images, masks in dataloader:
        images = images.to(device)
        masks = masks.to(device)

        logits = model(images)
        loss = criterion(logits, masks)

        batch_metrics = compute_all_metrics(
            pred=logits,
            target=masks,
            threshold=0.5,
            boundary_tolerance=2,
        )

        running_loss += loss.item()
        running_dice += batch_metrics["dice"]
        running_iou += batch_metrics["iou"]
        running_precision += batch_metrics["precision"]
        running_hd95 += batch_metrics["hd95"]
        running_boundary_f += batch_metrics["boundary_f_score"]
        running_boundary_precision += batch_metrics["boundary_precision"]
        running_boundary_recall += batch_metrics["boundary_recall"]

    n = len(dataloader)

    return {
        "loss": running_loss / n,
        "dice": running_dice / n,
        "iou": running_iou / n,
        "precision": running_precision / n,
        "hd95": running_hd95 / n,
        "boundary_f_score": running_boundary_f / n,
        "boundary_precision": running_boundary_precision / n,
        "boundary_recall": running_boundary_recall / n,
    }


def main():
    root = "data/oxford_102_flower"

    image_size = 256
    batch_size = 8
    num_epochs = 50
    learning_rate = 1e-4
    num_workers = 4

    save_dir = "checkpoints"
    os.makedirs(save_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_dataset = OxfordFlowerSegmentationDataset(
        root=root,
        split="train",
        image_size=image_size
    )

    val_dataset = OxfordFlowerSegmentationDataset(
        root=root,
        split="val",
        image_size=image_size
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    model = ResNet34UNet(
        num_classes=1,
        pretrained=True
    ).to(device)

    criterion = BCEDiceLoss(
        bce_weight=0.5,
        dice_weight=0.5
    )

    optimizer = AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=1e-4
    )

    best_dice = 0.0

    for epoch in range(num_epochs):
        print(f"\nEpoch [{epoch + 1}/{num_epochs}]")

        train_metrics = train_one_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
        )

        val_metrics = validate(
            model=model,
            dataloader=val_loader,
            criterion=criterion,
            device=device,
        )

        print(
            f"Train | "
            f"Loss: {train_metrics['loss']:.4f} | "
            f"Dice: {train_metrics['dice']:.4f} | "
            f"IoU: {train_metrics['iou']:.4f} | "
            f"Precision: {train_metrics['precision']:.4f}"
        )

        print(
            f"Val   | "
            f"Loss: {val_metrics['loss']:.4f} | "
            f"Dice: {val_metrics['dice']:.4f} | "
            f"IoU: {val_metrics['iou']:.4f} | "
            f"HD95: {val_metrics['hd95']:.4f} | "
            f"Boundary-F: {val_metrics['boundary_f_score']:.4f} | "
            f"Precision: {val_metrics['precision']:.4f}"
        )

        if val_metrics["dice"] > best_dice:
            best_dice = val_metrics["dice"]

            save_path = os.path.join(save_dir, "best_resnet34_unet.pth")

            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_dice": best_dice,
                    "val_metrics": val_metrics,
                    "train_metrics": train_metrics,
                },
                save_path,
            )

            print(f"Saved best model with Dice: {best_dice:.4f}")


if __name__ == "__main__":
    main()
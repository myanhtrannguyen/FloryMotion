from pathlib import Path
import sys
from collections import defaultdict

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm


# =========================
# Project paths
# =========================

PROJECT_ROOT = Path("/Users/trannguyenmyanh/Documents/SleepingBeauty")

SEG_ROOT = PROJECT_ROOT / "source/image_segmentation"
MODEL_ROOT = PROJECT_ROOT / "source/image_segmentation/models/unet_resnet34"

sys.path.append(str(SEG_ROOT))
sys.path.append(str(MODEL_ROOT))


from dataset import OxfordFlowersSegmentation
from metrics import compute_all_metrics
from model_resnet34_unet import ResNet34UNet


# =========================
# Config
# =========================

DATA_ROOT = PROJECT_ROOT / "data/OxfordFlowers102"

CHECKPOINT_PATH = (
    PROJECT_ROOT
    / "source/image_segmentation/models/unet_resnet34/result/best_model.pth"
)

IMAGE_SIZE = 256
BATCH_SIZE = 32
NUM_WORKERS = 2
THRESHOLD = 0.5
BOUNDARY_TOLERANCE = 2


# =========================
# Utility
# =========================

def get_sample_dice(pred_mask, target_mask, eps=1e-7):
    pred_mask = pred_mask.astype(np.uint8)
    target_mask = target_mask.astype(np.uint8)

    intersection = np.logical_and(pred_mask, target_mask).sum()
    total = pred_mask.sum() + target_mask.sum()

    return float((2.0 * intersection + eps) / (total + eps))


def get_meta_value(metas, key, index):
    """
    DataLoader sẽ collate meta thành dict:
    metas["name_cat"] = list[str]
    metas["image_id"] = tensor/list
    """
    value = metas[key]

    if isinstance(value, torch.Tensor):
        return value[index].item()

    return value[index]


# =========================
# Load model
# =========================

from model_resnet34_unet import ResNet34UNet


def load_model(checkpoint_path: Path, device: torch.device):
    model = ResNet34UNet(
        num_classes=1,
        pretrained=False,
    )

    checkpoint = torch.load(checkpoint_path, map_location=device)

    if isinstance(checkpoint, dict):
        if "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        elif "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = checkpoint
    else:
        state_dict = checkpoint

    clean_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith("module."):
            key = key.replace("module.", "")
        clean_state_dict[key] = value

    model.load_state_dict(clean_state_dict, strict=True)
    model = model.to(device)
    model.eval()

    return model


# =========================
# Evaluate
# =========================

@torch.no_grad()
def evaluate(model, dataloader, device):
    running_dice = 0.0
    running_iou = 0.0
    running_hd95 = 0.0
    running_boundary_f = 0.0
    running_precision = 0.0

    num_batches = 0

    category_dice = defaultdict(list)

    for images, masks, metas in tqdm(dataloader, desc="Testing"):
        images = images.to(device)
        masks = masks.to(device)

        logits = model(images)

        batch_metrics = compute_all_metrics(
            pred=logits,
            target=masks,
            threshold=THRESHOLD,
            boundary_tolerance=BOUNDARY_TOLERANCE,
        )

        running_dice += batch_metrics["dice"]
        running_iou += batch_metrics["iou"]
        running_hd95 += batch_metrics["hd95"]
        running_boundary_f += batch_metrics["boundary_f_score"]
        running_precision += batch_metrics["precision"]

        num_batches += 1

        # Per-sample Dice để lấy top 5 category
        probs = torch.sigmoid(logits)
        preds = (probs > THRESHOLD).float()

        preds_np = preds.detach().cpu().numpy()
        masks_np = masks.detach().cpu().numpy()

        batch_size = preds_np.shape[0]

        for i in range(batch_size):
            pred_mask = preds_np[i, 0]
            gt_mask = masks_np[i, 0]

            sample_dice = get_sample_dice(pred_mask, gt_mask)
            name_cat = get_meta_value(metas, "name_cat", i)

            category_dice[name_cat].append(sample_dice)

    m_dice = running_dice / num_batches
    m_iou = running_iou / num_batches
    m_hd95 = running_hd95 / num_batches
    m_boundary_f = running_boundary_f / num_batches
    m_precision = running_precision / num_batches

    category_mean_dice = {
        category: float(np.mean(dice_list))
        for category, dice_list in category_dice.items()
    }

    top5_categories = sorted(
        category_mean_dice.items(),
        key=lambda x: x[1],
        reverse=True,
    )[:5]

    return {
        "mDice": m_dice,
        "mIoU": m_iou,
        "mHD95": m_hd95,
        "Boundary F-Score": m_boundary_f,
        "Precision": m_precision,
        "Top 5": top5_categories,
    }


# =========================
# Main
# =========================

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Using device: {device}")
    print(f"Data root: {DATA_ROOT}")
    print(f"Checkpoint: {CHECKPOINT_PATH}")

    if not DATA_ROOT.exists():
        raise FileNotFoundError(f"Data root not found: {DATA_ROOT}")

    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(f"Checkpoint not found: {CHECKPOINT_PATH}")

    test_dataset = OxfordFlowersSegmentation(
        data_root=DATA_ROOT,
        split="test",
        image_size=IMAGE_SIZE,
        augment=False,
        drop_empty_masks=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
    )

    print(f"Number of test samples: {len(test_dataset)}")
    print(f"Dropped empty masks: {len(test_dataset.dropped_empty_masks)}")

    model = load_model(CHECKPOINT_PATH, device)

    results = evaluate(model, test_loader, device)

    print("\n===== Test Performance =====")
    print(f"mDice            : {results['mDice']:.4f}")
    print(f"mIoU             : {results['mIoU']:.4f}")
    print(f"mHD95            : {results['mHD95']:.4f}")
    print(f"Boundary F-Score : {results['Boundary F-Score']:.4f}")
    print(f"Precision        : {results['Precision']:.4f}")

    print("\nTop 5 best performance categories based on Dice:")
    for rank, (category, dice) in enumerate(results["Top 5"], start=1):
        print(f"{rank}. {category}: {dice:.4f}")

    print("\n===== Table Row Format =====")
    top5_names = ", ".join([category for category, _ in results["Top 5"]])

    print(
        f"{results['mDice']:.4f}\t"
        f"{results['mIoU']:.4f}\t"
        f"{results['mHD95']:.4f}\t"
        f"{results['Boundary F-Score']:.4f}\t"
        f"{results['Precision']:.4f}\t"
        f"{top5_names}"
    )


if __name__ == "__main__":
    main()
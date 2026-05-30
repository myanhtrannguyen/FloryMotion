from __future__ import annotations

import csv
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from PIL import Image, ImageEnhance
from torch.utils.data import Dataset


IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


@dataclass(frozen=True)
class FlowerSample:
    image_id: int
    split: str
    class_label: int
    name_cat: str

    @property
    def image_name(self) -> str:
        return f"image_{self.image_id:05d}.jpg"

    @property
    def mask_name(self) -> str:
        return f"segmim_{self.image_id:05d}.jpg"


def read_split_csv(csv_path: Path, split: str) -> List[FlowerSample]:
    samples: List[FlowerSample] = []
    with csv_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["split"] != split:
                continue
            samples.append(
                FlowerSample(
                    image_id=int(row["image_id"]),
                    split=row["split"],
                    class_label=int(row["class_label"]),
                    name_cat=row["name_cat"],
                )
            )
    return samples


def convert_blue_background_mask(mask: Image.Image) -> np.ndarray:
    mask_np = np.asarray(mask.convert("RGB"))
    r = mask_np[:, :, 0].astype(np.int16)
    g = mask_np[:, :, 1].astype(np.int16)
    b = mask_np[:, :, 2].astype(np.int16)
    background = (b > 100) & (b > r + 25) & (b > g + 25)
    return (~background).astype(np.uint8)


class OxfordFlowersSegmentation(Dataset):
    def __init__(
        self,
        data_root: str | Path,
        split: str,
        image_size: int = 256,
        augment: bool = False,
    ) -> None:
        self.data_root = Path(data_root)
        self.split = split
        self.image_size = image_size
        self.augment = augment
        self.samples = read_split_csv(
            self.data_root / "oxford102_flower_segmentation.csv",
            split=split,
        )

        if not self.samples:
            raise ValueError(f"No samples found for split={split!r} in {self.data_root}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, object]]:
        sample = self.samples[index]
        image_path = self.data_root / self.split / "images" / sample.image_name
        mask_path = self.data_root / self.split / "masks" / sample.mask_name

        image = Image.open(image_path).convert("RGB")
        mask_image = Image.open(mask_path).convert("RGB")

        if self.augment:
            image, mask_image = self._augment(image, mask_image)

        image = image.resize((self.image_size, self.image_size), Image.Resampling.BILINEAR)
        mask_image = mask_image.resize((self.image_size, self.image_size), Image.Resampling.NEAREST)

        image_np = np.asarray(image, dtype=np.float32) / 255.0
        image_np = (image_np - IMAGENET_MEAN) / IMAGENET_STD
        image_tensor = torch.from_numpy(image_np.transpose(2, 0, 1)).float()

        mask_np = convert_blue_background_mask(mask_image)
        mask_tensor = torch.from_numpy(mask_np[None, :, :]).float()

        meta = {
            "image_id": sample.image_id,
            "class_label": sample.class_label,
            "name_cat": sample.name_cat,
            "image_path": str(image_path),
            "mask_path": str(mask_path),
        }
        return image_tensor, mask_tensor, meta

    def _augment(self, image: Image.Image, mask: Image.Image) -> Tuple[Image.Image, Image.Image]:
        if random.random() < 0.5:
            image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.Transpose.FLIP_LEFT_RIGHT)

        angle = random.uniform(-15.0, 15.0)
        image = image.rotate(angle, resample=Image.Resampling.BILINEAR)
        mask = mask.rotate(angle, resample=Image.Resampling.NEAREST)

        scale = random.uniform(0.85, 1.0)
        crop_w = max(1, int(image.width * scale))
        crop_h = max(1, int(image.height * scale))
        left = random.randint(0, image.width - crop_w)
        top = random.randint(0, image.height - crop_h)
        crop_box = (left, top, left + crop_w, top + crop_h)
        image = image.crop(crop_box)
        mask = mask.crop(crop_box)

        if random.random() < 0.5:
            image = ImageEnhance.Brightness(image).enhance(random.uniform(0.8, 1.2))
        if random.random() < 0.5:
            image = ImageEnhance.Contrast(image).enhance(random.uniform(0.8, 1.2))
        if random.random() < 0.5:
            image = ImageEnhance.Color(image).enhance(random.uniform(0.8, 1.2))

        return image, mask

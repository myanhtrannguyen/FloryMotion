import os
import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF


class OxfordFlowerSegmentationDataset(Dataset):
    def __init__(
        self,
        root,
        split="train",
        image_size=256,
        transform=None
    ):
        self.root = root
        self.split = split
        self.image_size = image_size
        self.transform = transform

        split_file = os.path.join(root, f"{split}.txt")

        with open(split_file, "r") as f:
            self.image_names = [line.strip() for line in f.readlines()]

        self.image_dir = os.path.join(root, "images")
        self.mask_dir = os.path.join(root, "masks")

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        image_name = self.image_names[idx]

        image_path = os.path.join(self.image_dir, image_name)

        mask_name = os.path.splitext(image_name)[0] + ".png"
        mask_path = os.path.join(self.mask_dir, mask_name)

        image = Image.open(image_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        image = TF.resize(image, [self.image_size, self.image_size])
        mask = TF.resize(
            mask,
            [self.image_size, self.image_size],
            interpolation=Image.NEAREST
        )

        image = TF.to_tensor(image)

        image = TF.normalize(
            image,
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )

        mask = np.array(mask).astype(np.float32)

        # Nếu mask đang là 0/255
        mask = (mask > 0).astype(np.float32)

        mask = torch.from_numpy(mask).unsqueeze(0)

        return image, mask

# from __future__ import annotations

# import csv
# import random
# from dataclasses import dataclass
# from pathlib import Path
# from typing import Dict, List, Tuple

# import numpy as np
# import torch
# from PIL import Image, ImageEnhance
# from torch.utils.data import Dataset


# IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
# IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


# @dataclass(frozen=True)
# class FlowerSample:
#     image_id: int
#     split: str
#     class_label: int
#     name_cat: str

#     @property
#     def image_name(self) -> str:
#         return f"image_{self.image_id:05d}.jpg"

#     @property
#     def mask_name(self) -> str:
#         return f"segmim_{self.image_id:05d}.jpg"


# def read_split_csv(csv_path: Path, split: str) -> List[FlowerSample]:
#     samples: List[FlowerSample] = []
#     with csv_path.open("r", newline="", encoding="utf-8") as f:
#         reader = csv.DictReader(f)
#         for row in reader:
#             if row["split"] != split:
#                 continue
#             samples.append(
#                 FlowerSample(
#                     image_id=int(row["image_id"]),
#                     split=row["split"],
#                     class_label=int(row["class_label"]),
#                     name_cat=row["name_cat"],
#                 )
#             )
#     return samples


# def convert_blue_background_mask(mask: Image.Image) -> np.ndarray:
#     mask_np = np.asarray(mask.convert("RGB"))
#     r = mask_np[:, :, 0].astype(np.int16)
#     g = mask_np[:, :, 1].astype(np.int16)
#     b = mask_np[:, :, 2].astype(np.int16)
#     background = (b > 100) & (b > r + 25) & (b > g + 25)
#     return (~background).astype(np.uint8)


# def mask_has_foreground(mask_path: Path) -> bool:
#     mask = Image.open(mask_path).convert("RGB")
#     binary_mask = convert_blue_background_mask(mask)
#     return bool(binary_mask.sum() > 0)


# class OxfordFlowersSegmentation(Dataset):
#     def __init__(
#         self,
#         data_root: str | Path,
#         split: str,
#         image_size: int = 256,
#         augment: bool = False,
#         drop_empty_masks: bool = True,
#     ) -> None:
#         self.data_root = Path(data_root)
#         self.split = split
#         self.image_size = image_size
#         self.augment = augment
#         self.drop_empty_masks = drop_empty_masks
#         self.samples = read_split_csv(
#             self.data_root / "oxford102_flower_segmentation.csv",
#             split=split,
#         )
#         self.dropped_empty_masks: List[FlowerSample] = []

#         if self.drop_empty_masks:
#             kept_samples: List[FlowerSample] = []
#             for sample in self.samples:
#                 mask_path = self.data_root / self.split / "masks" / sample.mask_name
#                 if mask_has_foreground(mask_path):
#                     kept_samples.append(sample)
#                 else:
#                     self.dropped_empty_masks.append(sample)
#             self.samples = kept_samples

#         if not self.samples:
#             raise ValueError(f"No samples found for split={split!r} in {self.data_root}")

#     def __len__(self) -> int:
#         return len(self.samples)

#     def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, object]]:
#         sample = self.samples[index]
#         image_path = self.data_root / self.split / "images" / sample.image_name
#         mask_path = self.data_root / self.split / "masks" / sample.mask_name

#         image = Image.open(image_path).convert("RGB")
#         mask_image = Image.open(mask_path).convert("RGB")

#         if self.augment:
#             image, mask_image = self._augment(image, mask_image)

#         image = image.resize((self.image_size, self.image_size), Image.Resampling.BILINEAR)
#         mask_image = mask_image.resize((self.image_size, self.image_size), Image.Resampling.NEAREST)

#         image_np = np.asarray(image, dtype=np.float32) / 255.0
#         image_np = (image_np - IMAGENET_MEAN) / IMAGENET_STD
#         image_tensor = torch.from_numpy(image_np.transpose(2, 0, 1)).float()

#         mask_np = convert_blue_background_mask(mask_image)
#         mask_tensor = torch.from_numpy(mask_np[None, :, :]).float()

#         meta = {
#             "image_id": sample.image_id,
#             "class_label": sample.class_label,
#             "name_cat": sample.name_cat,
#             "image_path": str(image_path),
#             "mask_path": str(mask_path),
#         }
#         return image_tensor, mask_tensor, meta

#     def _augment(self, image: Image.Image, mask: Image.Image) -> Tuple[Image.Image, Image.Image]:
#         if random.random() < 0.5:
#             image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
#             mask = mask.transpose(Image.Transpose.FLIP_LEFT_RIGHT)

#         angle = random.uniform(-15.0, 15.0)
#         image = image.rotate(angle, resample=Image.Resampling.BILINEAR)
#         mask = mask.rotate(angle, resample=Image.Resampling.NEAREST)

#         scale = random.uniform(0.85, 1.0)
#         crop_w = max(1, int(image.width * scale))
#         crop_h = max(1, int(image.height * scale))
#         left = random.randint(0, image.width - crop_w)
#         top = random.randint(0, image.height - crop_h)
#         crop_box = (left, top, left + crop_w, top + crop_h)
#         image = image.crop(crop_box)
#         mask = mask.crop(crop_box)

#         if random.random() < 0.5:
#             image = ImageEnhance.Brightness(image).enhance(random.uniform(0.8, 1.2))
#         if random.random() < 0.5:
#             image = ImageEnhance.Contrast(image).enhance(random.uniform(0.8, 1.2))
#         if random.random() < 0.5:
#             image = ImageEnhance.Color(image).enhance(random.uniform(0.8, 1.2))

#         return image, mask

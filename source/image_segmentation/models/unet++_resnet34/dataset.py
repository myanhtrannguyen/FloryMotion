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
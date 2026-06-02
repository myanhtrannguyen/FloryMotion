from __future__ import annotations

import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _natural_key(path: Path) -> List[object]:
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", path.name)]


def list_image_files(directory: str | Path) -> List[Path]:
    directory = Path(directory)
    if not directory.exists():
        raise FileNotFoundError(f"Directory not found: {directory}")

    files = [
        path
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    ]
    files.sort(key=_natural_key)
    return files


def image_to_tensor(image: Image.Image, normalize: bool = True) -> torch.Tensor:
    image_np = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    tensor = torch.from_numpy(image_np.transpose(2, 0, 1)).float()
    if normalize:
        tensor = tensor * 2.0 - 1.0
    return tensor


def resize_image(image: Image.Image, image_size: int | Tuple[int, int]) -> Image.Image:
    if isinstance(image_size, int):
        size = (image_size, image_size)
    else:
        size = image_size
    return image.resize(size, Image.Resampling.BICUBIC)


@dataclass(frozen=True)
class AnimeGANSample:
    domain: str
    smooth_path: Optional[Path] = None
    style_path: Optional[Path] = None
    image_path: Optional[Path] = None


class AnimeGANDataset(Dataset):
    """
    Dataset loader for the local animeGAN layout:

        data/animeGAN/
            Hayao/{smooth,style}
            Shinkai/{smooth,style}
            val/*.jpg
            test/test_photo/*.jpg

    Modes:
        train:
            returns unpaired or index-paired smooth/style images for a domain.
        val:
            returns validation photos only.
        test:
            returns test photos only.

    Returned tensors are RGB CHW float tensors. By default they are normalized
    to [-1, 1], which is the common GAN input range.
    """

    def __init__(
        self,
        data_root: str | Path,
        domain: str = "Hayao",
        mode: str = "train",
        image_size: int | Tuple[int, int] = 256,
        normalize: bool = True,
        unpaired: bool = True,
        transform: Optional[Callable[[Image.Image], torch.Tensor]] = None,
    ) -> None:
        self.data_root = Path(data_root)
        self.domain = domain
        self.mode = mode
        self.image_size = image_size
        self.normalize = normalize
        self.unpaired = unpaired
        self.transform = transform

        if self.mode == "train":
            domain_root = self.data_root / self.domain
            self.smooth_paths = list_image_files(domain_root / "smooth")
            self.style_paths = list_image_files(domain_root / "style")
            if not self.smooth_paths or not self.style_paths:
                raise ValueError(f"No train images found for domain={domain!r} in {domain_root}")
            self.image_paths: List[Path] = []
        elif self.mode == "val":
            self.image_paths = list_image_files(self.data_root / "val")
            self.smooth_paths = []
            self.style_paths = []
        elif self.mode == "test":
            self.image_paths = list_image_files(self.data_root / "test" / "test_photo")
            self.smooth_paths = []
            self.style_paths = []
        else:
            raise ValueError("mode must be one of: 'train', 'val', 'test'")

        if self.mode in {"val", "test"} and not self.image_paths:
            raise ValueError(f"No {self.mode} images found in {self.data_root}")

    def __len__(self) -> int:
        if self.mode == "train":
            return max(len(self.smooth_paths), len(self.style_paths))
        return len(self.image_paths)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, Dict[str, object]] | Tuple[torch.Tensor, torch.Tensor, Dict[str, object]]:
        if self.mode == "train":
            smooth_path = self.smooth_paths[index % len(self.smooth_paths)]
            if self.unpaired:
                style_path = random.choice(self.style_paths)
            else:
                style_path = self.style_paths[index % len(self.style_paths)]

            smooth_image = self._load_image(smooth_path)
            style_image = self._load_image(style_path)
            meta = {
                "domain": self.domain,
                "smooth_path": str(smooth_path),
                "style_path": str(style_path),
                "smooth_name": smooth_path.name,
                "style_name": style_path.name,
            }
            return smooth_image, style_image, meta

        image_path = self.image_paths[index]
        image = self._load_image(image_path)
        meta = {
            "domain": self.domain,
            "mode": self.mode,
            "image_path": str(image_path),
            "image_name": image_path.name,
        }
        return image, meta

    def _load_image(self, path: Path) -> torch.Tensor:
        image = Image.open(path).convert("RGB")
        image = resize_image(image, self.image_size)

        if self.transform is not None:
            return self.transform(image)

        return image_to_tensor(image, normalize=self.normalize)


def create_animegan_datasets(
    data_root: str | Path,
    domain: str = "Hayao",
    image_size: int | Tuple[int, int] = 256,
    normalize: bool = True,
) -> Dict[str, AnimeGANDataset]:
    return {
        "train": AnimeGANDataset(
            data_root=data_root,
            domain=domain,
            mode="train",
            image_size=image_size,
            normalize=normalize,
        ),
        "val": AnimeGANDataset(
            data_root=data_root,
            domain=domain,
            mode="val",
            image_size=image_size,
            normalize=normalize,
        ),
        "test": AnimeGANDataset(
            data_root=data_root,
            domain=domain,
            mode="test",
            image_size=image_size,
            normalize=normalize,
        ),
    }


class CartoonGANTrainDataset(Dataset):
    """
    Unpaired photo/style dataset for CartoonGAN.

    Photo images are read from `photo_dir`. Style images are read only from:

        data_root/{Hayao|Shinkai}/style

    The `{domain}/smooth` folder is intentionally not used.
    """

    def __init__(
        self,
        data_root: str | Path,
        domain: str,
        photo_dir: str | Path,
        image_size: int | Tuple[int, int] = 256,
        normalize: bool = True,
        unpaired: bool = True,
    ) -> None:
        self.data_root = Path(data_root)
        self.domain = domain
        self.photo_dir = Path(photo_dir)
        if not self.photo_dir.is_absolute():
            self.photo_dir = self.data_root / self.photo_dir
        self.style_dir = self.data_root / self.domain / "style"
        self.image_size = image_size
        self.normalize = normalize
        self.unpaired = unpaired

        self.photo_paths = list_image_files(self.photo_dir)
        self.style_paths = list_image_files(self.style_dir)
        if not self.photo_paths:
            raise ValueError(f"No photo images found in {self.photo_dir}")
        if not self.style_paths:
            raise ValueError(f"No style images found in {self.style_dir}")

    def __len__(self) -> int:
        return max(len(self.photo_paths), len(self.style_paths))

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, object]]:
        photo_path = self.photo_paths[index % len(self.photo_paths)]
        if self.unpaired:
            style_path = random.choice(self.style_paths)
        else:
            style_path = self.style_paths[index % len(self.style_paths)]

        photo = self._load_image(photo_path)
        style = self._load_image(style_path)
        meta = {
            "domain": self.domain,
            "photo_path": str(photo_path),
            "style_path": str(style_path),
            "photo_name": photo_path.name,
            "style_name": style_path.name,
        }
        return photo, style, meta

    def _load_image(self, path: Path) -> torch.Tensor:
        image = Image.open(path).convert("RGB")
        image = resize_image(image, self.image_size)
        return image_to_tensor(image, normalize=self.normalize)


class CartoonGANEvalDataset(Dataset):
    """
    Validation/test photo dataset for CartoonGAN.

    mode='val'  -> data_root/val
    mode='test' -> data_root/test/test_photo
    """

    def __init__(
        self,
        data_root: str | Path,
        mode: str,
        image_size: int | Tuple[int, int] = 256,
        normalize: bool = True,
    ) -> None:
        self.data_root = Path(data_root)
        self.mode = mode
        self.image_size = image_size
        self.normalize = normalize

        if self.mode == "val":
            image_dir = self.data_root / "val"
        elif self.mode == "test":
            image_dir = self.data_root / "test" / "test_photo"
        else:
            raise ValueError("mode must be one of: 'val', 'test'")

        self.image_dir = image_dir
        self.image_paths = list_image_files(image_dir)
        if not self.image_paths:
            raise ValueError(f"No images found in {image_dir}")

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, Dict[str, object]]:
        image_path = self.image_paths[index]
        image = Image.open(image_path).convert("RGB")
        image = resize_image(image, self.image_size)
        tensor = image_to_tensor(image, normalize=self.normalize)
        meta = {
            "mode": self.mode,
            "image_path": str(image_path),
            "image_name": image_path.name,
        }
        return tensor, meta

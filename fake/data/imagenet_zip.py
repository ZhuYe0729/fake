from __future__ import annotations

import csv
import io
import zipfile
from pathlib import Path
from typing import Any

from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import InterpolationMode


DEFAULT_IMAGENET_ROOT = Path("/data/home/scxj523/run/wja/data/datasets/imagenet_val")


class ImageNetZipDataset(Dataset[tuple[torch.Tensor, int]]):
    def __init__(
        self,
        dataset_root: str | Path = DEFAULT_IMAGENET_ROOT,
        csv_name: str = "val.csv",
        zip_name: str = "imagenet_val.zip",
        model_config: dict[str, Any] | None = None,
        transform: transforms.Compose | None = None,
    ) -> None:
        self.dataset_root = Path(dataset_root)
        self.csv_path = self.dataset_root / csv_name
        self.zip_path = self.dataset_root / zip_name
        self.samples = _read_samples(self.csv_path)
        self.transform = transform or build_imagenet_transform(model_config or {})
        self._zip: zipfile.ZipFile | None = None

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        image_path, target = self.samples[index]
        with self._zip_file().open(image_path) as f:
            image = Image.open(io.BytesIO(f.read())).convert("RGB")
        return self.transform(image), target

    def _zip_file(self) -> zipfile.ZipFile:
        if self._zip is None:
            self._zip = zipfile.ZipFile(self.zip_path)
        return self._zip


def build_imagenet_transform(model_config: dict[str, Any]) -> transforms.Compose:
    pretrained_cfg = model_config.get("pretrained_cfg", {})
    input_size = pretrained_cfg.get("input_size", [3, 224, 224])
    image_size = int(input_size[-1])
    crop_pct = float(pretrained_cfg.get("crop_pct", 0.95))
    resize_size = int(round(image_size / crop_pct))
    mean = pretrained_cfg.get("mean", [0.485, 0.456, 0.406])
    std = pretrained_cfg.get("std", [0.229, 0.224, 0.225])
    interpolation = _resolve_interpolation(pretrained_cfg.get("interpolation", "bicubic"))

    return transforms.Compose(
        [
            transforms.Resize(resize_size, interpolation=interpolation),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )


def _read_samples(csv_path: Path) -> list[tuple[str, int]]:
    samples: list[tuple[str, int]] = []
    with csv_path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        image_field = "image:FILE" if "image:FILE" in (reader.fieldnames or []) else "image"
        for row in reader:
            samples.append((row[image_field], int(row["category"])))
    return samples


def _resolve_interpolation(name: str) -> InterpolationMode:
    if name.lower() == "bicubic":
        return InterpolationMode.BICUBIC
    if name.lower() == "bilinear":
        return InterpolationMode.BILINEAR
    return InterpolationMode.BICUBIC

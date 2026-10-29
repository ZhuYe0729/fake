from __future__ import annotations

import io
import os
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tiff"}
LABELS = {"0_real": 0, "real": 0, "nature": 0,
          "1_fake": 1, "fake": 1, "ai": 1}


@dataclass(frozen=True)
class ImageRecord:
    path: str
    label: int
    source: str = "file"


def discover_labeled_folders(root: str | Path, limit_per_class: int | None = None) -> tuple[ImageRecord, ...]:
    records: list[ImageRecord] = []
    for current, _, files in os.walk(Path(root)):
        label = LABELS.get(Path(current).name.lower())
        if label is None:
            continue
        for name in sorted(files):
            path = Path(current) / name
            if path.suffix.lower() in IMAGE_EXTENSIONS:
                records.append(ImageRecord(str(path), label))
    return limit_records(records, limit_per_class)


def discover_genimage_zip(path: str | Path, limit_per_class: int | None = None) -> dict[str, tuple[ImageRecord, ...]]:
    grouped: dict[str, list[ImageRecord]] = {}
    with zipfile.ZipFile(path) as archive:
        for member in archive.namelist():
            parts = Path(member).parts
            if len(parts) < 5 or parts[0] != "GenImage" or parts[2] != "val":
                continue
            label = LABELS.get(parts[3].lower())
            if label is not None and Path(member).suffix.lower() in IMAGE_EXTENSIONS:
                grouped.setdefault(parts[1], []).append(ImageRecord(member, label, "zip"))
    return {name: limit_records(sorted(rows, key=lambda row: row.path), limit_per_class)
            for name, rows in sorted(grouped.items())}


def limit_records(records: Iterable[ImageRecord], limit: int | None) -> tuple[ImageRecord, ...]:
    rows = tuple(sorted(records, key=lambda row: row.path))
    if limit is None:
        return rows
    counts = {0: 0, 1: 0}
    selected = []
    for row in rows:
        if counts[row.label] < limit:
            selected.append(row)
            counts[row.label] += 1
    return tuple(selected)


class MirrorDataset(Dataset[tuple[torch.Tensor, int]]):
    def __init__(self, records: tuple[ImageRecord, ...], zip_path: str | Path | None = None) -> None:
        self.records = records
        self.zip_path = Path(zip_path) if zip_path else None
        self._archive: zipfile.ZipFile | None = None
        self.transform = transforms.Compose([
            transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(),
        ])

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        row = self.records[index]
        if row.source == "zip":
            if self._archive is None:
                self._archive = zipfile.ZipFile(self.zip_path)
            with self._archive.open(row.path) as handle:
                image = Image.open(io.BytesIO(handle.read())).convert("RGB")
        else:
            image = Image.open(row.path).convert("RGB")
        return self.transform(image), row.label


from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from PIL import Image
from torch.utils.data import Dataset


class FakeClueDataset(Dataset[dict[str, Any]]):
    def __init__(self, annotation: str | Path, image_root: str | Path,
                 processor: Any, limit: int | None = None) -> None:
        rows = json.loads(Path(annotation).read_text())
        self.rows = rows[:limit] if limit is not None else rows
        self.image_root = Path(image_root)
        self.processor = processor

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        image_path = self.image_root / row["image"]
        image = Image.open(image_path).convert("RGB")
        inputs = self.processor(
            text=row["conversations"][0]["value"], images=image,
            return_tensors="pt", padding="max_length", max_length=1024,
            truncation=True)
        return {"inputs": {key: value.squeeze(0) for key, value in inputs.items()},
                "label": int(row["label"]), "image_path": str(image_path)}


def classify_response(response: str) -> int | None:
    """Return FakeClue's historical label convention: real=1, fake=0."""
    for sentence in response.lower().split(".")[:2]:
        if "real" in sentence:
            return 1
        if "fake" in sentence:
            return 0
    return None


def score_predictions(labels: list[int], predictions: list[int | None]) -> dict[str, float | int]:
    if len(labels) != len(predictions):
        raise ValueError("labels and predictions must have equal length")
    total = len(labels)
    correct = sum(prediction == label for label, prediction in zip(labels, predictions))
    unknown = sum(prediction is None for prediction in predictions)
    real = [prediction == 1 for label, prediction in zip(labels, predictions) if label == 1]
    fake = [prediction == 0 for label, prediction in zip(labels, predictions) if label == 0]
    return {"num_samples": total, "accuracy": correct / total if total else 0.0,
            "real_accuracy": sum(real) / len(real) if real else 0.0,
            "fake_accuracy": sum(fake) / len(fake) if fake else 0.0,
            "unknown_responses": unknown}


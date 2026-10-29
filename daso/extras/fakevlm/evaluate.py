#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from extras.common.config import load_config, write_json
from extras.fakevlm.dataset import FakeClueDataset, classify_response, score_predictions
from extras.fakevlm.model import load_fakevlm_dense


def move(inputs: dict[str, torch.Tensor], device: str) -> dict[str, torch.Tensor]:
    return {key: value.to(device=device, dtype=torch.bfloat16 if key == "pixel_values" else None)
            for key, value in inputs.items()}


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a dense FakeVLM checkpoint on FakeClue.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    model, processor = load_fakevlm_dense(config["model"]["checkpoint"], args.device)
    data = FakeClueDataset(config["dataset"]["annotation"], config["dataset"]["image_root"],
                           processor, args.limit)
    loader = DataLoader(data, batch_size=args.batch_size, num_workers=args.workers)
    labels: list[int] = []
    predictions: list[int | None] = []
    samples = []
    for batch in loader:
        output = model.generate(**move(batch["inputs"], args.device),
                                max_new_tokens=args.max_new_tokens)
        decoded = processor.batch_decode(output, skip_special_tokens=True)
        for label, path, text in zip(batch["label"].tolist(), batch["image_path"], decoded):
            response = text.split("?")[-1]
            prediction = classify_response(response)
            labels.append(int(label)); predictions.append(prediction)
            samples.append({"image_path": path, "label": int(label),
                            "prediction": prediction, "response": response})
    write_json(args.output, {"metrics": score_predictions(labels, predictions),
                             "predictions": samples,
                             "config": json.loads(json.dumps(config))})


if __name__ == "__main__":
    main()

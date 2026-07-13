#!/usr/bin/env python
"""Create a fixed random CNN/DM test subset for quick PMPD-style testing."""

from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path

os.environ.setdefault("HF_DATASETS_CACHE", "/tmp/pmpd_eval_hf_datasets_cache")
os.environ.setdefault("HF_HOME", "/tmp/pmpd_eval_hf_home")

from datasets import DatasetDict, load_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a fixed CNN/DM test subset.")
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("/home/agent/wja/data/datasets/flaxquant"),
        help="Root directory containing dataset repos/subsets.",
    )
    parser.add_argument(
        "--output-name",
        default="cnn_dailymail_3.0.0_test_random1000_seed42",
        help="Output subdirectory name under --data-root.",
    )
    parser.add_argument("--sample-size", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite an existing subset directory.",
    )
    return parser.parse_args()


def load_cnn_dm_test(data_root: Path):
    repo_dir = data_root / "cnn_dailymail_repo" / "3.0.0"
    if repo_dir.exists():
        files = sorted(repo_dir.glob("test-*.parquet"))
        return load_dataset("parquet", data_files={"test": [str(path) for path in files]}, split="test")
    return load_dataset("cnn_dailymail", "3.0.0", split="test")


def main() -> None:
    args = parse_args()
    out_dir = args.data_root / args.output_name
    if out_dir.exists() and not args.overwrite:
        raise SystemExit(f"Output directory already exists: {out_dir}")

    dataset = load_cnn_dm_test(args.data_root)
    if args.sample_size > len(dataset):
        raise ValueError(f"sample-size={args.sample_size} exceeds dataset size={len(dataset)}")

    rng = random.Random(args.seed)
    indices = sorted(rng.sample(range(len(dataset)), args.sample_size))
    subset = dataset.select(indices)

    if out_dir.exists() and args.overwrite:
        import shutil

        shutil.rmtree(out_dir)

    DatasetDict({"test": subset}).save_to_disk(str(out_dir))

    metadata = {
        "source": "cnn_dailymail 3.0.0 test",
        "source_num_rows": len(dataset),
        "subset_split": "test",
        "subset_num_rows": len(subset),
        "sample_size": args.sample_size,
        "seed": args.seed,
        "sampling": "random.sample without replacement; indices sorted for stable source order",
        "columns": subset.column_names,
    }
    (out_dir / "subset_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (out_dir / "indices.json").write_text(
        json.dumps(indices, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print(f"[saved] {out_dir}")
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

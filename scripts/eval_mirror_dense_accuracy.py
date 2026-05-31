#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import datetime as dt
import io
import os
import random
import sys
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from PIL import Image, ImageFile


REPO_ROOT = Path(__file__).resolve().parents[1]
MIRROR_ROOT = REPO_ROOT / "third_party" / "MIRROR"


DEFAULT_CHAMELEON_ROOT = Path("/data/home/scxj523/run/wja/data/datasets/Chameleon/test")
DEFAULT_GENIMAGE_ROOT = Path("/data/home/scxj523/run/wja/data/datasets/genimage-validation")
DEFAULT_GENIMAGE_ZIP = DEFAULT_GENIMAGE_ROOT / "genimage-validation.zip"
DEFAULT_WEIGHT_ROOT = Path("/data/home/scxj523/run/wja/data/models/facebook/MIRROR/weight")
DEFAULT_MODEL_PATH = DEFAULT_WEIGHT_ROOT / "checkpoint-h-cur.pth"
DEFAULT_MEMORY_PATH = DEFAULT_WEIGHT_ROOT / "mirror_phase1.pth"
DEFAULT_BACKBONE_PATH = DEFAULT_WEIGHT_ROOT / "dinov3-huge"

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tiff"}
REAL_LABEL_NAMES = {"0_real", "real", "nature"}
FAKE_LABEL_NAMES = {"1_fake", "fake", "ai"}
GENIMAGE_GENERATORS = [
    "Midjourney",
    "stable_diffusion_v_1_4",
    "stable_diffusion_v_1_5",
    "ADM",
    "glide",
    "wukong",
    "VQDM",
    "BigGAN",
]

ImageFile.LOAD_TRUNCATED_IMAGES = True


@dataclass(frozen=True)
class ImageRecord:
    path: str
    label: int
    source: str


@dataclass(frozen=True)
class EvalSplit:
    benchmark: str
    dataset: str
    records: tuple[ImageRecord, ...]


def import_runtime_deps() -> None:
    global DataLoader
    global InterpolationMode
    global SequentialSampler
    global TF
    global average_precision_score
    global default_collate
    global np
    global roc_auc_score
    global torch
    global tqdm
    global transforms

    import numpy as np
    import torch
    from sklearn.metrics import average_precision_score, roc_auc_score
    from torch.utils.data import DataLoader, SequentialSampler
    from torch.utils.data._utils.collate import default_collate
    from torchvision import transforms
    from torchvision.transforms import InterpolationMode, functional as TF
    from tqdm import tqdm


class RandomScaleCropOrDirect224:
    def __init__(
        self,
        crop_size: int = 224,
        interpolation=None,
        antialias: bool = True,
        eval_resize_short: int = 512,
    ) -> None:
        self.crop_size = crop_size
        self.interp = interpolation if interpolation is not None else InterpolationMode.BICUBIC
        self.antialias = antialias
        self.eval_resize_short = eval_resize_short

    def __call__(self, img: Image.Image) -> Image.Image:
        width, height = img.size
        if min(width, height) > self.eval_resize_short:
            scale = self.eval_resize_short / min(width, height)
            new_width = int(round(width * scale))
            new_height = int(round(height * scale))
            img = TF.resize(
                img,
                [new_height, new_width],
                interpolation=self.interp,
                antialias=self.antialias,
            )
        return TF.center_crop(img, [self.crop_size, self.crop_size])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate MIRROR dense detector on forensic benchmarks.")
    parser.add_argument("--benchmarks", nargs="+", default=["Chameleon", "GenImage"])
    parser.add_argument("--chameleon-root", default=str(DEFAULT_CHAMELEON_ROOT))
    parser.add_argument("--genimage-root", default=str(DEFAULT_GENIMAGE_ROOT))
    parser.add_argument("--genimage-zip", default=str(DEFAULT_GENIMAGE_ZIP))
    parser.add_argument("--prefer-extracted-genimage", action="store_true")
    parser.add_argument("--model-path", default=str(DEFAULT_MODEL_PATH))
    parser.add_argument("--memory-path", default=str(DEFAULT_MEMORY_PATH))
    parser.add_argument("--backbone-path", default=str(DEFAULT_BACKBONE_PATH))
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--use-amp", action="store_true")
    parser.add_argument("--output", default="artifacts/results/mirror_dense/accuracy.csv")
    parser.add_argument("--log-interval", type=int, default=50)
    parser.add_argument("--limit-per-class", type=int, default=None)
    parser.add_argument("--discover-only", action="store_true")
    return parser.parse_args()


def seed_everything(seed: int = 0) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def label_from_name(name: str) -> int | None:
    lowered = name.lower()
    if lowered in REAL_LABEL_NAMES:
        return 0
    if lowered in FAKE_LABEL_NAMES:
        return 1
    return None


def is_image_path(path: str | Path) -> bool:
    return Path(path).suffix.lower() in IMAGE_EXTENSIONS


def limit_records_per_class(records: Iterable[ImageRecord], limit: int | None) -> tuple[ImageRecord, ...]:
    all_records = tuple(records)
    if limit is None:
        return all_records
    kept: list[ImageRecord] = []
    counts = {0: 0, 1: 0}
    for record in all_records:
        if counts[record.label] < limit:
            kept.append(record)
            counts[record.label] += 1
    return tuple(kept)


def discover_labeled_folder(root: Path, source: str, limit_per_class: int | None) -> tuple[ImageRecord, ...]:
    records: list[ImageRecord] = []
    if not root.exists():
        return ()
    for current_root, _, files in os.walk(root):
        label = label_from_name(Path(current_root).name)
        if label is None:
            continue
        for filename in sorted(files):
            path = Path(current_root) / filename
            if is_image_path(path):
                records.append(ImageRecord(str(path), label, source))
    records.sort(key=lambda item: item.path)
    return limit_records_per_class(records, limit_per_class)


def discover_chameleon(args: argparse.Namespace) -> list[EvalSplit]:
    root = Path(args.chameleon_root)
    records = discover_labeled_folder(root, "file", args.limit_per_class)
    return [EvalSplit("Chameleon", "ALL", records)]


def genimage_base(root: Path) -> Path:
    if (root / "GenImage").is_dir():
        return root / "GenImage"
    return root


def discover_genimage_from_folder(args: argparse.Namespace) -> list[EvalSplit]:
    base = genimage_base(Path(args.genimage_root))
    if not base.is_dir():
        return []
    splits: list[EvalSplit] = []
    generator_names = [name for name in GENIMAGE_GENERATORS if (base / name).is_dir()]
    if not generator_names:
        generator_names = sorted(path.name for path in base.iterdir() if path.is_dir())
    for generator in generator_names:
        records = discover_labeled_folder(base / generator / "val", "file", args.limit_per_class)
        if records:
            splits.append(EvalSplit("GenImage", generator, records))
    return splits


def discover_genimage_from_zip(args: argparse.Namespace) -> list[EvalSplit]:
    zip_path = Path(args.genimage_zip)
    if not zip_path.exists():
        return []
    grouped: dict[str, list[ImageRecord]] = {}
    with zipfile.ZipFile(zip_path) as zf:
        for member in zf.namelist():
            if member.endswith("/") or not is_image_path(member):
                continue
            parts = Path(member).parts
            if len(parts) < 5 or parts[0] != "GenImage" or parts[2] != "val":
                continue
            label = label_from_name(parts[3])
            if label is None:
                continue
            generator = parts[1]
            grouped.setdefault(generator, []).append(ImageRecord(member, label, "zip"))

    splits: list[EvalSplit] = []
    generator_names = [name for name in GENIMAGE_GENERATORS if name in grouped]
    generator_names.extend(sorted(name for name in grouped if name not in set(generator_names)))
    for generator in generator_names:
        records = sorted(grouped[generator], key=lambda item: item.path)
        splits.append(EvalSplit("GenImage", generator, limit_records_per_class(records, args.limit_per_class)))
    return splits


def discover_genimage(args: argparse.Namespace) -> list[EvalSplit]:
    if not args.prefer_extracted_genimage:
        zip_splits = discover_genimage_from_zip(args)
        if zip_splits:
            return zip_splits
    folder_splits = discover_genimage_from_folder(args)
    if folder_splits:
        return folder_splits
    return discover_genimage_from_zip(args)


def discover_splits(args: argparse.Namespace) -> list[EvalSplit]:
    splits: list[EvalSplit] = []
    for benchmark in args.benchmarks:
        normalized = benchmark.lower()
        if normalized == "chameleon":
            splits.extend(discover_chameleon(args))
        elif normalized in {"genimage", "genimage-validation", "genimage_validation"}:
            splits.extend(discover_genimage(args))
        else:
            raise ValueError(f"Unsupported benchmark: {benchmark}")
    return splits


def compress_image(img: Image.Image, quality: int = 96) -> Image.Image:
    buffer = io.BytesIO()
    img.convert("RGB").save(buffer, format="JPEG", quality=quality, optimize=True)
    buffer.seek(0)
    return Image.open(buffer).convert("RGB")


class MirrorDataset:
    def __init__(self, records: tuple[ImageRecord, ...], zip_path: str | None = None) -> None:
        self.records = records
        self.zip_path = zip_path
        self._zip_file: zipfile.ZipFile | None = None
        self.transform_pre = transforms.Compose([RandomScaleCropOrDirect224(), transforms.CenterCrop(224)])
        self.transform_norm = transforms.Compose([transforms.ToTensor()])

    def __len__(self) -> int:
        return len(self.records)

    def __getstate__(self) -> dict[str, object]:
        state = self.__dict__.copy()
        state["_zip_file"] = None
        return state

    def _zip(self) -> zipfile.ZipFile:
        if self.zip_path is None:
            raise RuntimeError("zip_path is required for zip records")
        if self._zip_file is None:
            self._zip_file = zipfile.ZipFile(self.zip_path)
        return self._zip_file

    def _open_image(self, record: ImageRecord) -> Image.Image:
        if record.source == "zip":
            with self._zip().open(record.path) as f:
                return Image.open(io.BytesIO(f.read())).convert("RGB")
        return Image.open(record.path).convert("RGB")

    def __getitem__(self, index: int):
        record = self.records[index]
        try:
            image = self._open_image(record)
            image = self.transform_pre(image)
            if Path(record.path).suffix.lower() in {".png", ".bmp", ".tiff"}:
                image = compress_image(image, quality=96)
            image = self.transform_norm(image)
            return image, torch.tensor(record.label, dtype=torch.long)
        except Exception as exc:
            print(f"[Warn] Skipping unreadable image: {record.path} ({exc})")
            return None


def collate_skip_invalid(batch: list[object]):
    valid = [item for item in batch if item is not None]
    if not valid:
        return None
    return default_collate(valid)


def summarize_records(records: tuple[ImageRecord, ...]) -> tuple[int, int, int]:
    real = sum(1 for record in records if record.label == 0)
    fake = sum(1 for record in records if record.label == 1)
    return len(records), real, fake


def evaluate(
    dataloader: DataLoader,
    model: torch.nn.Module,
    device: torch.device,
    use_amp: bool,
    log_interval: int,
) -> dict[str, float | int]:
    model.eval()
    all_preds: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []
    all_scores: list[np.ndarray] = []
    start = time.perf_counter()

    with torch.no_grad():
        for step, batch in enumerate(tqdm(dataloader, desc="   Computing", leave=False), start=1):
            if batch is None:
                continue
            samples, targets = batch
            samples = samples.to(device, non_blocking=True)
            with torch.amp.autocast("cuda", enabled=use_amp):
                logits, _, _ = model(samples)
            scores = torch.nn.functional.softmax(logits, dim=1)[:, 1]
            all_scores.append(scores.detach().cpu().numpy())
            all_preds.append(torch.argmax(logits, dim=1).detach().cpu().numpy())
            all_labels.append(targets.numpy())
            if log_interval > 0 and step % log_interval == 0:
                print(f"   processed_batches={step}")

    elapsed = time.perf_counter() - start
    labels = np.concatenate(all_labels) if all_labels else np.array([], dtype=np.int64)
    preds = np.concatenate(all_preds) if all_preds else np.array([], dtype=np.int64)
    scores = np.concatenate(all_scores) if all_scores else np.array([], dtype=np.float32)

    if labels.size == 0:
        raise RuntimeError("Cannot evaluate an empty dataset.")
    if len(np.unique(labels)) < 2:
        raise RuntimeError("Both real and fake samples are required for MIRROR detection metrics.")

    acc = float(np.mean(preds == labels))
    real_acc = float(np.mean(preds[labels == 0] == 0))
    fake_acc = float(np.mean(preds[labels == 1] == 1))
    bal_acc = (real_acc + fake_acc) / 2
    auc = float(roc_auc_score(labels, scores))
    ap = float(average_precision_score(labels, scores))
    return {
        "num_samples": int(labels.size),
        "real_samples": int(np.sum(labels == 0)),
        "fake_samples": int(np.sum(labels == 1)),
        "acc": acc,
        "real_acc": real_acc,
        "fake_acc": fake_acc,
        "bal_acc": bal_acc,
        "auc": auc,
        "ap": ap,
        "elapsed_sec": elapsed,
        "images_per_sec": float(labels.size / elapsed) if elapsed > 0 else 0.0,
    }


def append_rows(path: str | Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    exists = output_path.exists() and output_path.stat().st_size > 0
    with output_path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        writer.writerows(rows)


def load_model(args: argparse.Namespace, device: torch.device) -> torch.nn.Module:
    if str(MIRROR_ROOT) not in sys.path:
        sys.path.insert(0, str(MIRROR_ROOT))
    from models.mirror import build_mirror

    print(f">>> Loading MIRROR detector: {args.model_path}")
    model = build_mirror(memory_path=args.memory_path, backbone_path=args.backbone_path)
    checkpoint = torch.load(args.model_path, map_location="cpu", weights_only=False)
    state_dict = checkpoint.get("model", checkpoint.get("state_dict", checkpoint))
    incompatible = model.load_state_dict(state_dict, strict=False)
    print(
        ">>> Loaded checkpoint "
        f"missing_keys={len(incompatible.missing_keys)} unexpected_keys={len(incompatible.unexpected_keys)}"
    )
    return model.to(device)


def main() -> None:
    args = parse_args()
    splits = discover_splits(args)
    if not splits:
        raise RuntimeError("No benchmark data was discovered.")

    print("Discovered datasets:")
    for split in splits:
        total, real, fake = summarize_records(split.records)
        print(f"  {split.benchmark}/{split.dataset}: total={total} real={real} fake={fake}")

    if args.discover_only:
        return
    import_runtime_deps()
    seed_everything(0)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required. Submit this script to a GPU compute node.")

    device = torch.device(args.device)
    model = load_model(args, device)
    gpu_name = torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu"
    timestamp = dt.datetime.now().isoformat(timespec="seconds")
    rows: list[dict[str, object]] = []

    for split in splits:
        total, real, fake = summarize_records(split.records)
        if total == 0 or real == 0 or fake == 0:
            print(f"[Skip] {split.benchmark}/{split.dataset}: total={total} real={real} fake={fake}")
            continue

        zip_path = args.genimage_zip if split.records and split.records[0].source == "zip" else None
        dataset = MirrorDataset(split.records, zip_path=zip_path)
        dataloader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            sampler=SequentialSampler(dataset),
            num_workers=args.num_workers,
            pin_memory=device.type == "cuda",
            persistent_workers=args.num_workers > 0,
            collate_fn=collate_skip_invalid,
        )
        print(f"\n[{split.benchmark}/{split.dataset}] Evaluating {total} images ...")
        result = evaluate(dataloader, model, device, args.use_amp, args.log_interval)
        print(
            f"  Bal_Acc={result['bal_acc']:.4f} AUC={result['auc']:.4f} "
            f"images_per_sec={result['images_per_sec']:.3f}"
        )
        rows.append(
            {
                "timestamp": timestamp,
                "model": "MIRROR-DINOv3-Huge",
                "benchmark": split.benchmark,
                "dataset": split.dataset,
                "num_samples": result["num_samples"],
                "real_samples": result["real_samples"],
                "fake_samples": result["fake_samples"],
                "acc": f"{result['acc']:.6f}",
                "real_acc": f"{result['real_acc']:.6f}",
                "fake_acc": f"{result['fake_acc']:.6f}",
                "bal_acc": f"{result['bal_acc']:.6f}",
                "auc": f"{result['auc']:.6f}",
                "ap": f"{result['ap']:.6f}",
                "elapsed_sec": f"{result['elapsed_sec']:.3f}",
                "images_per_sec": f"{result['images_per_sec']:.3f}",
                "batch_size": args.batch_size,
                "num_workers": args.num_workers,
                "device": gpu_name,
                "use_amp": args.use_amp,
                "model_path": args.model_path,
                "memory_path": args.memory_path,
                "backbone_path": args.backbone_path,
            }
        )

    if not rows:
        raise RuntimeError("No metric rows were produced.")

    by_benchmark: dict[str, list[dict[str, object]]] = {}
    for row in rows:
        by_benchmark.setdefault(str(row["benchmark"]), []).append(row)
    for benchmark, benchmark_rows in by_benchmark.items():
        if len(benchmark_rows) < 2:
            continue
        metric_names = ["acc", "real_acc", "fake_acc", "bal_acc", "auc", "ap"]
        mean_row = dict(benchmark_rows[0])
        mean_row["dataset"] = "MEAN"
        mean_row["num_samples"] = sum(int(row["num_samples"]) for row in benchmark_rows)
        mean_row["real_samples"] = sum(int(row["real_samples"]) for row in benchmark_rows)
        mean_row["fake_samples"] = sum(int(row["fake_samples"]) for row in benchmark_rows)
        for metric in metric_names:
            mean_row[metric] = f"{np.mean([float(row[metric]) for row in benchmark_rows]):.6f}"
        mean_row["elapsed_sec"] = f"{sum(float(row['elapsed_sec']) for row in benchmark_rows):.3f}"
        mean_row["images_per_sec"] = ""
        rows.append(mean_row)

    fieldnames = list(rows[0].keys())
    append_rows(args.output, fieldnames, rows)
    print(f"\nSaved results -> {args.output}")


if __name__ == "__main__":
    main()

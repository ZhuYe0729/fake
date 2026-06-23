#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import math
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from common_fakevlm_pareto import (
    DEBUG_ROOT,
    DEFAULT_IMAGE_ROOT,
    DEFAULT_MODEL_PATH,
    DEFAULT_TEST_JSON,
    append_csv,
    local_cuda_index,
    read_csv,
    read_json,
    write_json,
)
from fakevlm_policy_runtime import apply_policy_runtime


IGNORE_INDEX = -100
MAX_LENGTH = 1024
LOSS_DEFINITION = "assistant_answer_token_nll_v2_active_prefix_aligned"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Measure FakeVLM policy teacher-forcing NLL on FakeClue.")
    parser.add_argument("--output-root", type=Path, default=DEBUG_ROOT)
    parser.add_argument("--policies", choices=["stratified", "validation"], default="stratified")
    parser.add_argument("--points", default="all", help="For validation mode: all or comma-separated batch:point pairs, e.g. 16:0,16:5")
    parser.add_argument("--policy-indices", default=None, help="For stratified mode: comma-separated policy indices to run.")
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--test-json-file", default=DEFAULT_TEST_JSON)
    parser.add_argument("--image-root", default=DEFAULT_IMAGE_ROOT)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--sample-limit", type=int, default=None)
    parser.add_argument("--calib-samples", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--calib-batch-size", type=int, default=1)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--max-policies", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    from eval_fakevlm_uniform_accuracy import FakeVLMDataset as AccuracyDataset
    from run_fakevlm_prefill_speed import load_fakevlm
    from transformers import AutoProcessor

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = torch.device(f"cuda:{local_cuda_index(args.gpu)}")
    torch.cuda.set_device(device)

    policies = select_policies(args)
    if args.max_policies is not None:
        policies = policies[: args.max_policies]
    if not policies:
        raise RuntimeError("no policies selected")

    processor = AutoProcessor.from_pretrained(args.model_path, local_files_only=True)
    processor.vision_feature_select_strategy = None
    dataset = FakeClueLossDataset(
        model_path=args.model_path,
        test_json_file=args.test_json_file,
        image_root=args.image_root,
        sample_limit=args.sample_limit,
        processor=processor,
    )
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True)
    calib_dataset = AccuracyDataset(
        model_path=args.model_path,
        test_json_file=args.test_json_file,
        image_root=args.image_root,
        sample_limit=max(args.calib_samples, 1),
    )
    calib_loader = DataLoader(calib_dataset, batch_size=args.calib_batch_size, shuffle=False, num_workers=args.workers, pin_memory=True)

    out_path = args.output_root / "quality" / f"{args.policies}_loss.csv"
    if out_path.exists() and args.overwrite:
        out_path.unlink()
    done = existing_keys(out_path)
    dense_nll = existing_dense_nll(out_path)
    rows = []
    for item in policies:
        key = item["key"]
        if key in done:
            print(f"[skip] existing loss row {key}")
            continue
        policy = read_json(Path(item["policy_json"]))
        model = load_fakevlm(args.model_path, device)
        report = apply_policy_runtime(model, policy, calib_loader=calib_loader, device=device, calib_samples=args.calib_samples)
        result = evaluate_nll(model, dataloader, device)
        if dense_nll is None and is_dense_policy(item):
            dense_nll = result["nll"]
        if dense_nll is None:
            raise RuntimeError("dense baseline NLL is required before non-dense policy rows")
        row = {
            **item,
            "nll": f"{result['nll']:.10f}",
            "nll_delta_vs_dense": f"{result['nll'] - dense_nll:.10f}",
            "ppl": f"{math.exp(result['nll']):.10f}" if result["nll"] < 50 else "",
            "loss_tokens": result["loss_tokens"],
            "loss_sum": f"{result['loss_sum']:.6f}",
            "dense_nll": f"{dense_nll:.10f}",
            "replaced_linear_count": report.replaced_linear_count,
            "skipped_linear_count": report.skipped_linear_count,
            "backend_counts": report.backend_counts,
            "runtime_skipped": report.skipped[:20],
            "sample_limit": args.sample_limit if args.sample_limit is not None else "",
            "batch_size": args.batch_size,
            "calib_samples": args.calib_samples,
            "loss_definition": LOSS_DEFINITION,
        }
        rows.append(row)
        append_csv(out_path, [row])
        point_dir = args.output_root / "quality" / f"{args.policies}_loss_points"
        write_json(point_dir / f"{key}.json", {"row": row})
        print(f"[done] {key} nll={result['nll']:.6f} delta={result['nll'] - dense_nll:.6f}")
        del model
        gc.collect()
        torch.cuda.empty_cache()
    if rows:
        write_json(args.output_root / "quality" / f"{args.policies}_loss_metadata.json", {"rows_written": len(rows), "policies": len(policies)})
    print(f"measured loss for {len(rows)} policies")


class FakeClueLossDataset(Dataset):
    def __init__(
        self,
        *,
        model_path: str,
        test_json_file: str,
        image_root: str,
        sample_limit: int | None,
        processor: Any,
    ) -> None:
        del model_path
        super().__init__()
        self.image_root = Path(image_root)
        self.processor = processor
        data = read_json(Path(test_json_file))
        self.data = data[:sample_limit] if sample_limit is not None else data
        pad_id = getattr(self.processor.tokenizer, "pad_token_id", None)
        self.pad_token_id = 0 if pad_id is None else int(pad_id)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        item = self.data[idx]
        prompt = item["conversations"][0]["value"]
        answer = item["conversations"][1]["value"]
        image = Image.open(self.image_root / item["image"]).convert("RGB")
        prompt_inputs = self.processor(
            text=prompt,
            images=image,
            return_tensors="pt",
            padding="max_length",
            max_length=MAX_LENGTH,
            truncation=True,
        )
        full_inputs = self.processor(
            text=prompt + answer,
            images=image,
            return_tensors="pt",
            padding="max_length",
            max_length=MAX_LENGTH,
            truncation=True,
        )
        input_ids = full_inputs["input_ids"].squeeze(0)
        attention_mask = full_inputs["attention_mask"].squeeze(0)
        labels, answer_token_count = build_answer_labels(
            input_ids=input_ids,
            attention_mask=attention_mask,
            prompt_input_ids=prompt_inputs["input_ids"].squeeze(0),
            prompt_attention_mask=prompt_inputs["attention_mask"].squeeze(0),
            eos_token_id=self.processor.tokenizer.eos_token_id,
        )
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "pixel_values": full_inputs["pixel_values"].squeeze(0),
            "labels": labels,
            "loss_token_count": torch.tensor(answer_token_count, dtype=torch.long),
        }


def build_answer_labels(
    *,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    prompt_input_ids: torch.Tensor,
    prompt_attention_mask: torch.Tensor,
    eos_token_id: int | None,
) -> tuple[torch.Tensor, int]:
    full_positions = torch.nonzero(attention_mask, as_tuple=False).flatten()
    prompt_positions = torch.nonzero(prompt_attention_mask, as_tuple=False).flatten()
    full_active = input_ids[full_positions]
    prompt_active = prompt_input_ids[prompt_positions]
    if eos_token_id is not None and prompt_active.numel() and int(prompt_active[-1]) == int(eos_token_id):
        prompt_prefix = prompt_active[:-1]
    else:
        prompt_prefix = prompt_active
    prefix_len = int(prompt_prefix.numel())
    if prefix_len == 0 or full_active.numel() <= prefix_len:
        raise RuntimeError("no answer tokens remain after prompt alignment")
    if not torch.equal(full_active[:prefix_len], prompt_prefix):
        raise RuntimeError("prompt tokens are not a prefix of full prompt-plus-answer tokens")
    answer_positions = full_positions[prefix_len:]
    labels = torch.full_like(input_ids, IGNORE_INDEX)
    labels[answer_positions] = input_ids[answer_positions]
    return labels, int(answer_positions.numel())


@torch.inference_mode()
def evaluate_nll(model: torch.nn.Module, dataloader: DataLoader, device: torch.device) -> dict[str, float]:
    loss_sum = 0.0
    loss_tokens = 0
    for batch in dataloader:
        labels = batch.pop("labels").to(device=device, non_blocking=True)
        token_count = int(batch.pop("loss_token_count").sum().item())
        if token_count == 0:
            continue
        inputs = move_inputs(batch, device)
        outputs = model(**inputs, labels=labels, use_cache=False)
        loss = float(outputs.loss.detach().float().item())
        loss_sum += loss * token_count
        loss_tokens += token_count
        del outputs
    if loss_tokens == 0:
        raise RuntimeError("no target tokens were available for loss measurement")
    return {"nll": loss_sum / loss_tokens, "loss_sum": loss_sum, "loss_tokens": loss_tokens}


def move_inputs(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    moved = {}
    for key, value in batch.items():
        if key == "pixel_values":
            moved[key] = value.to(device=device, dtype=torch.bfloat16, non_blocking=True)
        else:
            moved[key] = value.to(device=device, non_blocking=True)
    return moved


def select_policies(args: argparse.Namespace) -> list[dict[str, Any]]:
    if args.policies == "stratified":
        rows = read_csv(args.output_root / "stratified" / "quality_policies.csv")
        if args.policy_indices:
            wanted = {int(item) for item in args.policy_indices.split(",") if item.strip()}
            rows = [row for row in rows if int(float(row["policy_index"])) in wanted]
        return [
            {
                "key": f"policy_{int(float(row['policy_index'])):03d}",
                "policy_index": int(float(row["policy_index"])),
                "batch_size_for_policy": "",
                "point_index": "",
                "policy_json": row["policy_json"],
                "label": row.get("label", ""),
            }
            for row in rows
        ]
    selected = read_csv(args.output_root / "validation" / "selected_pareto_points.csv")
    wanted = None
    if args.points != "all":
        wanted = {tuple(int(part) for part in spec.split(":", 1)) for spec in args.points.split(",") if spec.strip()}
    out = []
    for row in selected:
        batch = int(float(row["batch_size"]))
        point = int(float(row["point_index"]))
        if wanted is not None and (batch, point) not in wanted:
            continue
        out.append(
            {
                "key": f"batch_{batch}_point_{point:03d}",
                "policy_index": "",
                "batch_size_for_policy": batch,
                "point_index": point,
                "policy_json": row["policy_json"],
                "label": row.get("selection_reason", ""),
            }
        )
    return out


def existing_keys(path: Path) -> set[str]:
    if not path.exists():
        return set()
    return {row["key"] for row in read_csv(path)}


def existing_dense_nll(path: Path) -> float | None:
    if not path.exists():
        return None
    for row in read_csv(path):
        if is_dense_policy(row):
            return float(row["nll"])
    return None


def is_dense_policy(row: dict[str, Any]) -> bool:
    policy_index = str(row.get("policy_index", ""))
    point_index = str(row.get("point_index", ""))
    return policy_index in {"0", "0.0"} or point_index in {"0", "0.0"} or row.get("label") == "dense"


if __name__ == "__main__":
    main()

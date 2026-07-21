#!/usr/bin/env python3
"""Compare deterministic vLLM generation records from two KV-cache modes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def load(path: Path) -> dict[str, dict]:
    records: dict[str, dict] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        record = json.loads(line)
        records[str(record["question_id"])] = record
    return records


def answer(record: dict) -> str:
    return record["choices"][0]["turns"][0]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    reference, candidate = load(args.reference), load(args.candidate)
    common = sorted(set(reference) & set(candidate))
    changed = [qid for qid in common if answer(reference[qid]) != answer(candidate[qid])]
    details = [{
        "question_id": qid,
        "reference_answer": answer(reference[qid]),
        "candidate_answer": answer(candidate[qid]),
    } for qid in changed]
    result = {
        "reference": str(args.reference),
        "candidate": str(args.candidate),
        "reference_records": len(reference),
        "candidate_records": len(candidate),
        "common_records": len(common),
        "exact_match_records": len(common) - len(changed),
        "changed_records": len(changed),
        "changed_fraction": len(changed) / len(common) if common else None,
        "changed_examples": details[:5],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n",
                           encoding="utf-8")


if __name__ == "__main__":
    main()

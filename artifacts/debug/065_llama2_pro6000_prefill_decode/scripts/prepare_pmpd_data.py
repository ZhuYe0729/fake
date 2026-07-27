#!/usr/bin/env python3
"""Audit the fixed local PMPD datasets and write an artifact-local manifest."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pmpd_eval
from common import BERTSCORE_MODEL, BUNDLE, IWSLT_FILTER_TOKENIZER, PMPD, PMPD_DATA_ROOT, sha256, write_json


def main() -> None:
    rows = []
    for dataset, expected in PMPD["datasets"].items():
        args = argparse.Namespace(dataset=dataset, split="test", data_root=PMPD_DATA_ROOT,
                                  iwslt_filter_tokenizer=str(IWSLT_FILTER_TOKENIZER),
                                  question_begin=None, question_end=None)
        questions = pmpd_eval.build_questions(args)
        if len(questions) != expected:
            raise RuntimeError(f"{dataset}: expected {expected}, found {len(questions)}")
        rows.append({"dataset": dataset, "samples": len(questions),
                     "first_question_id": questions[0]["question_id"],
                     "last_question_id": questions[-1]["question_id"]})
    files = [PMPD_DATA_ROOT / "cnn_dailymail_3.0.0_test_random1000_seed42/subset_metadata.json",
             PMPD_DATA_ROOT / "dialogsum_repo/test.csv",
             PMPD_DATA_ROOT / "iwslt2017_en_fr_repo/data/2017-01-trnted/texts/en/fr/en-fr.zip"]
    for path in files:
        if not path.is_file():
            raise FileNotFoundError(path)
    write_json(BUNDLE / "cache/task_data_manifest.json", {
        "complete": True, "protocol": PMPD, "datasets": rows,
        "payload_files": [{"path": str(path.resolve()), "bytes": path.stat().st_size,
                           "sha256": sha256(path)} for path in files],
        "bertscore_model": str(BERTSCORE_MODEL),
        "iwslt_filter_tokenizer": str(IWSLT_FILTER_TOKENIZER)})
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()

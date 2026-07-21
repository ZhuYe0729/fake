#!/usr/bin/env python3
"""Create a clean run tree using only immutable designs and sample tensors."""

from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

from common import SRC054, SRC056, env_path, sha256, write_json


def copy_file(source: Path, target: Path, *, resume: bool) -> dict[str, object]:
    if not source.is_file():
        raise SystemExit(f"required bootstrap input is missing: {source}")
    source_hash = sha256(source)
    if target.exists():
        if not resume:
            raise SystemExit(f"target exists (use --resume to hash-check it): {target}")
        if sha256(target) != source_hash:
            raise SystemExit(f"existing target differs from source: {target}")
    else:
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    return {"source": str(source), "target": str(target), "sha256": source_hash, "bytes": source.stat().st_size}


def write_generated(source: Path, target: Path, text: str, *, resume: bool) -> dict[str, object]:
    if target.exists() and not resume:
        raise SystemExit(f"target exists (use --resume to update generated manifest): {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text)
    return {"source": str(source), "target": str(target), "source_sha256": sha256(source),
            "target_sha256": sha256(target), "bytes": target.stat().st_size, "generated": True}


def copy_policy_tree(source: Path, target: Path, *, resume: bool) -> list[dict[str, object]]:
    records = []
    for source_file in sorted(source.glob("*.json")):
        if source_file.name == "manifest.json":
            continue
        records.append(copy_file(source_file, target / source_file.name, resume=resume))
    policies = [p for p in target.glob("p[0-9][0-9].json")]
    if len(policies) != 72:
        raise SystemExit(f"expected 72 policy JSON files under {target}, found {len(policies)}")
    source_manifest = source / "manifest.json"
    manifest = json.loads(source_manifest.read_text())
    for row in manifest:
        row["path"] = str((target / f"{row['policy_id']}.json").resolve())
    records.append(write_generated(source_manifest, target / "manifest.json",
                                   json.dumps(manifest, indent=2) + "\n", resume=resume))
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, default=env_path("COSPAQ_RUN_ROOT"))
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    run_root = args.run_root.resolve()
    records: list[dict[str, object]] = []

    records += copy_policy_tree(
        SRC054 / "llama2_7b_chat/policies/prefill_only",
        run_root / "prefill_only/policies/prefill_only",
        resume=args.resume,
    )
    records += copy_policy_tree(
        SRC056 / "llama2_7b_chat/policies/prefill_decode",
        run_root / "prefill_decode/policies/prefill_decode",
        resume=args.resume,
    )
    records.append(copy_file(
        SRC054 / "llama2_7b_chat/samples/wikitext_2048_targets.pt",
        run_root / "prefill_only/samples/wikitext_2048_targets.pt",
        resume=args.resume,
    ))
    records.append(copy_file(
        SRC056 / "llama2_7b_chat/samples/wikitext_2048_64.pt",
        run_root / "prefill_decode/samples/wikitext_2048_64.pt",
        resume=args.resume,
    ))
    metadata = SRC056 / "llama2_7b_chat/samples/wikitext_2048_64_metadata.json"
    if metadata.exists():
        records.append(copy_file(metadata, run_root / "prefill_decode/samples" / metadata.name, resume=args.resume))
    architecture = SRC056 / "llama2_7b_chat/architecture_manifest.json"
    records.append(copy_file(architecture, run_root / "prefill_decode/architecture_manifest.json", resume=args.resume))

    provenance = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "clean reproduction bootstrap; no checkpoints, measurements, or fitted coefficients copied",
        "source_054": str(SRC054),
        "source_056": str(SRC056),
        "run_root": str(run_root),
        "files": records,
    }
    write_json(run_root / "bootstrap_provenance.json", provenance)
    print(json.dumps({"run_root": str(run_root), "copied_or_verified": len(records)}, indent=2))


if __name__ == "__main__":
    main()

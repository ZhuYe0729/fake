#!/usr/bin/env python3
"""Archive large raw evidence without deleting original files."""
from __future__ import annotations

import concurrent.futures
import gzip
import hashlib
import json
import os
from pathlib import Path
import tarfile
import tempfile

REPO = Path(__file__).resolve().parents[3]
ROOT = REPO / "artifacts/rebuttal"
DEST = ROOT / "portability/raw_archives"
SKIP = {
    "__pycache__", ".git", "cache", "compile_cache", "compiler_cache",
    "torch_compile_cache", "triton_cache", "hf_cache", "huggingface_cache",
    "extensions", "build", "torch_extensions", "runtime_tmp", ".pytest_cache",
    "checkpoints", "exports", "likelihood_datasets", "likelihood_auxiliary_datasets",
    "raw_archives",
}
BINARY = {
    ".safetensors", ".pt", ".pth", ".bin", ".cubin", ".o", ".so", ".a",
    ".pyc", ".llir", ".ptx", ".ttir", ".ttgir", ".source", ".arrow",
}
SOURCE = {".py", ".sh", ".md", ".csv", ".cu", ".cpp", ".h", ".cuh", ".tex"}


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(2**20), b""):
            h.update(chunk)
    return h.hexdigest()


def selected_paths():
    """Yield source/evidence files, excluding weights and reproducible caches."""
    for directory, dirs, names in os.walk(ROOT, followlinks=False):
        dirs[:] = sorted(
            n for n in dirs
            if n not in SKIP and not n.startswith("extensions_")
        )
        for name in sorted(names):
            path = Path(directory) / name
            token_data = (
                path.suffix == ".pt" and "data" in path.parts
                and any(s in name for s in ("tokens", "calibration_", "evaluation_test_"))
                and path.exists() and path.stat().st_size < 32 * 2**20
            )
            if path.suffix in BINARY and not token_data:
                continue
            if path.suffix in {".pid", ".lock"} or name in {".msc", ".DS_Store"}:
                continue
            yield path


def archive_candidate(path: Path) -> bool:
    if path.is_symlink() or path.suffix in SOURCE:
        return False
    return (
        path.name.endswith(".phase_trace.json")
        or (path.suffix == ".jsonl" and "results" in path.parts)
        or (path.suffix == ".json" and path.stat().st_size > 2**20)
        or (path.suffix != ".pt" and path.stat().st_size > 8 * 2**20)
    )


def group_name(path: Path) -> str:
    rel = path.relative_to(ROOT).parts
    return "run-" + rel[1] if rel[0] == "runs" else rel[0]


def make_archive(item):
    name, paths = item
    print(f"Archiving {name}: {len(paths)} files", flush=True)
    entries = []
    with tempfile.TemporaryDirectory(prefix="rebuttal183_") as temporary:
        archive = Path(temporary) / (name + ".tar.gz")
        with tarfile.open(archive, "w:gz", compresslevel=1, dereference=True) as tar:
            for path in paths:
                entries.append({
                    "path": path.relative_to(REPO).as_posix(),
                    "bytes": path.stat().st_size,
                    "sha256": digest(path),
                })
                tar.add(path, arcname=entries[-1]["path"], recursive=False)
        parts = []
        with archive.open("rb") as stream:
            index = 0
            while chunk := stream.read(40 * 2**20):
                part = DEST / f"{name}.tar.gz.part{index:03d}"
                part.write_bytes(chunk)
                parts.append({
                    "path": part.relative_to(REPO).as_posix(),
                    "bytes": len(chunk), "sha256": hashlib.sha256(chunk).hexdigest(),
                })
                index += 1
    manifest = DEST / (name + ".files.jsonl.gz")
    with gzip.open(manifest, "wt", encoding="utf-8", compresslevel=6) as stream:
        for entry in entries:
            stream.write(json.dumps(entry, ensure_ascii=False) + "\n")
    result = {
        "group": name, "file_count": len(entries),
        "original_bytes": sum(e["bytes"] for e in entries),
        "archive_bytes": sum(p["bytes"] for p in parts), "parts": parts,
        "file_manifest": manifest.relative_to(REPO).as_posix(),
        "manifest_sha256": digest(manifest),
    }
    print(f"Finished {name}: {result['archive_bytes'] / 2**20:.1f} MiB", flush=True)
    return result


def main():
    DEST.mkdir(parents=True, exist_ok=True)
    assert not (DEST / "manifest.json").exists(), "Archive already prepared."
    groups = {}
    for path in selected_paths():
        if archive_candidate(path):
            groups.setdefault(group_name(path), []).append(path)
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        records = list(pool.map(make_archive, sorted(groups.items())))
    output = {"format": 1, "originals_deleted": False, "groups": records}
    (DEST / "manifest.json").write_text(
        json.dumps(output, ensure_ascii=False, indent=2) + "\n"
    )
    extra = [
        "/" + p.relative_to(ROOT).as_posix()
        for paths in groups.values() for p in paths
        if not p.name.endswith(".phase_trace.json")
        and not (p.suffix == ".jsonl" and "results" in p.parts)
    ]
    (ROOT / ".gitignore").write_text(
        "# Generated raw evidence is stored losslessly in portability/raw_archives.\n"
        "!*.log\n"
        "**/cache/\n**/compile_cache/\n**/compiler_cache/\n"
        "**/torch_compile_cache/\n**/triton_cache/\n**/hf_cache/\n"
        "**/huggingface_cache/\n**/extensions/\n**/extensions_*/\n"
        "**/build/\n**/torch_extensions/\n**/runtime_tmp/\n"
        "**/.pytest_cache/\n**/likelihood_datasets/\n"
        "**/likelihood_auxiliary_datasets/\n"
        "*.pid\n*.lock\n*.pyc\n*.so\n*.o\n*.cubin\n*.llir\n"
        "*.ptx\n*.ttir\n*.ttgir\n*.source\n*.arrow\n"
        "*.phase_trace.json\n**/results/**/*.jsonl\n"
        "!**/data/wikitext_*_tokens.pt\n!**/data/calibration_*.pt\n"
        "!**/data/evaluation_test_*.pt\n"
        + "\n".join(sorted(extra)) + "\n"
    )
    print(json.dumps({
        "groups": len(records),
        "files": sum(r["file_count"] for r in records),
        "original_GiB": sum(r["original_bytes"] for r in records) / 2**30,
        "archive_MiB": sum(r["archive_bytes"] for r in records) / 2**20,
    }), flush=True)


if __name__ == "__main__":
    main()

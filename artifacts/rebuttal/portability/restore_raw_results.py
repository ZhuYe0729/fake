#!/usr/bin/env python3
"""Restore selected raw evidence and verify its original SHA-256."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import tarfile
import tempfile

REPO = Path(__file__).resolve().parents[3]
ROOT = Path(__file__).resolve().parent / "raw_archives"


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(2**20), b""):
            h.update(chunk)
    return h.hexdigest()


def restore(record):
    source = REPO / record["file_manifest"]
    assert digest(source) == record["manifest_sha256"]
    with gzip.open(source, "rt", encoding="utf-8") as stream:
        entries = {entry["path"]: entry for entry in map(json.loads, stream)}
    with tempfile.TemporaryFile() as combined:
        for entry in record["parts"]:
            part = REPO / entry["path"]
            assert digest(part) == entry["sha256"], part
            with part.open("rb") as stream:
                shutil.copyfileobj(stream, combined)
        combined.seek(0)
        seen = set()
        with tarfile.open(fileobj=combined, mode="r:gz") as archive:
            for member in archive:
                relative = PurePosixPath(member.name)
                assert not relative.is_absolute() and ".." not in relative.parts
                assert member.isfile() and member.name in entries
                destination = REPO / member.name
                assert destination.resolve().is_relative_to(REPO)
                expected = entries[member.name]
                if destination.exists():
                    assert digest(destination) == expected["sha256"], destination
                else:
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    with archive.extractfile(member) as src, destination.open("wb") as dst:
                        shutil.copyfileobj(src, dst)
                    assert digest(destination) == expected["sha256"], destination
                seen.add(member.name)
        assert seen == set(entries)
    print(f"Verified {record['group']}: {len(seen)} files")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--group", action="append", default=[])
    parser.add_argument("--all", action="store_true")
    args = parser.parse_args()
    records = json.loads((ROOT / "manifest.json").read_text())["groups"]
    known = {r["group"] for r in records}
    assert set(args.group) <= known, "Unknown group."
    if not args.all and not args.group:
        for r in records:
            print(f"{r['group']}: {r['file_count']} files, "
                  f"{r['archive_bytes'] / 2**20:.1f} MiB packed")
        return
    for record in records:
        if args.all or record["group"] in args.group:
            restore(record)


if __name__ == "__main__":
    main()

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_manifest(path: str | Path) -> list[str]:
    source = Path(path).resolve()
    root = source.parent
    manifest: dict[str, Any] = json.loads(source.read_text())
    errors: list[str] = []
    listed: set[str] = set()
    for item in manifest.get("files", []):
        listed.add(item["path"])
        target = root / item["path"]
        if not target.is_file():
            errors.append(f"missing: {item['path']}")
        elif item.get("sha256") and sha256(target) != item["sha256"]:
            errors.append(f"hash mismatch: {item['path']}")
    current = {str(path.relative_to(root)) for path in root.rglob("*")
               if path.is_file() and path != source}
    for relative in sorted(current - listed):
        errors.append(f"untracked by manifest: {relative}")
    return errors

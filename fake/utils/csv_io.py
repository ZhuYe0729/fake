from __future__ import annotations

import csv
import fcntl
from pathlib import Path
from typing import Mapping, Sequence


def append_csv_row(path: str | Path, fieldnames: Sequence[str], row: Mapping[str, object]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = output_path.with_suffix(output_path.suffix + ".lock")
    with lock_path.open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        _append_csv_row_locked(output_path, fieldnames, row)


def _append_csv_row_locked(output_path: Path, fieldnames: Sequence[str], row: Mapping[str, object]) -> None:
    requested_fieldnames = list(fieldnames)
    if not output_path.exists() or output_path.stat().st_size == 0:
        with output_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=requested_fieldnames)
            writer.writeheader()
            writer.writerow({name: row.get(name, "") for name in requested_fieldnames})
        return

    with output_path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        existing_fieldnames = list(reader.fieldnames or [])
        existing_rows = list(reader)

    merged_fieldnames = existing_fieldnames + [
        name for name in requested_fieldnames if name not in existing_fieldnames
    ]
    if not existing_fieldnames:
        merged_fieldnames = requested_fieldnames

    needs_rewrite = merged_fieldnames != existing_fieldnames
    open_mode = "w" if needs_rewrite else "a"
    with output_path.open(open_mode, newline="") as f:
        writer = csv.DictWriter(f, fieldnames=merged_fieldnames)
        if needs_rewrite:
            writer.writeheader()
            for existing_row in existing_rows:
                writer.writerow({name: existing_row.get(name, "") for name in merged_fieldnames})
        writer.writerow({name: row.get(name, "") for name in merged_fieldnames})

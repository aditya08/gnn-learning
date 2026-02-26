#!/usr/bin/env python3
import argparse
import csv
from pathlib import Path
from typing import Dict, List, Tuple


DEFAULT_KEY_COLUMNS = ("run_id", "sync_every", "epoch")
NUMERIC_SORT_COLUMNS = {"sync_every", "epoch"}


def read_csv_rows(path: Path) -> Tuple[List[str], List[Dict[str, str]]]:
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"{path} has no header")
        return reader.fieldnames, list(reader)


def sort_key(row: Dict[str, str], fieldnames: List[str]) -> Tuple:
    out = []
    for col in fieldnames:
        value = row.get(col, "")
        if col in NUMERIC_SORT_COLUMNS:
            try:
                out.append((0, int(value)))
            except ValueError:
                out.append((1, value))
        else:
            out.append((0, value))
    return tuple(out)


def dedup_key(row: Dict[str, str], key_columns: Tuple[str, ...]) -> Tuple[str, ...]:
    return tuple(row.get(k, "") for k in key_columns)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Merge multiple metrics.csv files into one deterministic CSV."
    )
    parser.add_argument(
        "--input-root",
        default="logs",
        help="Root directory containing run subdirectories with metrics.csv",
    )
    parser.add_argument(
        "--glob",
        default="*/metrics.csv",
        help="Glob relative to --input-root for selecting input CSVs",
    )
    parser.add_argument(
        "--output",
        default="results/metrics.csv",
        help="Merged output CSV path",
    )
    parser.add_argument(
        "--key-columns",
        nargs="+",
        default=list(DEFAULT_KEY_COLUMNS),
        help="Columns used to de-duplicate rows. Last seen row wins.",
    )
    args = parser.parse_args()

    input_root = Path(args.input_root)
    input_paths = sorted(input_root.glob(args.glob))
    if not input_paths:
        raise SystemExit(f"no input files matched: {input_root / args.glob}")

    merged_rows: Dict[Tuple[str, ...], Dict[str, str]] = {}
    canonical_fieldnames: List[str] = []

    for path in input_paths:
        fieldnames, rows = read_csv_rows(path)
        if not canonical_fieldnames:
            canonical_fieldnames = fieldnames
        elif fieldnames != canonical_fieldnames:
            raise SystemExit(
                f"header mismatch in {path}; expected {canonical_fieldnames}, got {fieldnames}"
            )

        for row in rows:
            merged_rows[dedup_key(row, tuple(args.key_columns))] = row

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    final_rows = sorted(merged_rows.values(), key=lambda r: sort_key(r, canonical_fieldnames))
    with output_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=canonical_fieldnames)
        writer.writeheader()
        writer.writerows(final_rows)

    print(f"merged {len(input_paths)} files, wrote {len(final_rows)} rows to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

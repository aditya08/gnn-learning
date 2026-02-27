#!/usr/bin/env python3
import argparse
import csv
from pathlib import Path
from typing import Dict, List


def read_rows(path: Path) -> List[Dict[str, str]]:
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise SystemExit(f"{path} has no header")
        return list(reader)


def to_float(row: Dict[str, str], key: str) -> float:
    try:
        return float(row[key])
    except (KeyError, ValueError):
        raise SystemExit(f"invalid numeric value for '{key}' in row: {row}")


def pareto_frontier(rows: List[Dict[str, str]], comm_key: str, acc_key: str) -> List[Dict[str, str]]:
    # Minimize communication (x), maximize accuracy (y).
    ordered = sorted(rows, key=lambda r: (to_float(r, comm_key), -to_float(r, acc_key)))
    frontier: List[Dict[str, str]] = []
    best_acc = float("-inf")
    for row in ordered:
        acc = to_float(row, acc_key)
        if acc > best_acc:
            frontier.append(row)
            best_acc = acc
    return frontier


def write_rows(path: Path, fieldnames: List[str], rows: List[Dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate Pareto frontier CSV from metrics.")
    parser.add_argument("--input", default="results/metrics.csv", help="Input metrics CSV")
    parser.add_argument("--all-output", default="results/pareto_all_fp32.csv", help="All points CSV")
    parser.add_argument(
        "--frontier-output",
        default="results/pareto_frontier_fp32.csv",
        help="Pareto frontier CSV",
    )
    parser.add_argument(
        "--comm-column",
        default="commMB_cum_fp32",
        help="Communication column (minimize)",
    )
    parser.add_argument(
        "--acc-column",
        default="best_microF1",
        help="Accuracy column (maximize)",
    )
    args = parser.parse_args()

    in_path = Path(args.input)
    rows = read_rows(in_path)
    if not rows:
        raise SystemExit(f"no rows in {in_path}")

    all_fields = ["sync_every", "epoch", args.comm_column, args.acc_column]
    all_rows = [
        {
            "sync_every": row.get("sync_every", ""),
            "epoch": row.get("epoch", ""),
            args.comm_column: f"{to_float(row, args.comm_column):.6f}",
            args.acc_column: f"{to_float(row, args.acc_column):.6f}",
        }
        for row in rows
    ]

    frontier = pareto_frontier(rows, args.comm_column, args.acc_column)
    frontier_rows = [
        {
            "sync_every": row.get("sync_every", ""),
            "epoch": row.get("epoch", ""),
            args.comm_column: f"{to_float(row, args.comm_column):.6f}",
            args.acc_column: f"{to_float(row, args.acc_column):.6f}",
        }
        for row in frontier
    ]

    write_rows(Path(args.all_output), all_fields, all_rows)
    write_rows(Path(args.frontier_output), all_fields, frontier_rows)
    print(
        f"wrote {len(all_rows)} points to {args.all_output}; "
        f"{len(frontier_rows)} Pareto points to {args.frontier_output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

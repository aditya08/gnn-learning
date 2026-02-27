#!/usr/bin/env python3

import argparse
import csv
import random
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List


LOG_PREFIX = "LOG,"


CSV_COLUMNS = [
    "run_id",
    "sync_every",
    "epoch",
    "loss",
    "microF1",
    "macroF1",
    "best_microF1",
    "best_macroF1",
    "local_steps",
    "epoch_steps",
    "effective_global_steps",
    "seed_nodes_epoch_global",
    "seed_nodes_cum_global",
    "target_global_steps",
    "world_size",
    "time",
    "syncs",
    "commMB_per_step_fp16",
    "commMB_per_step_fp32",
    "commMB_per_step_fp64",
    "commMB_epoch_fp16",
    "commMB_cum_fp16",
    "commMB_epoch_fp32",
    "commMB_cum_fp32",
    "commMB_epoch_fp64",
    "commMB_cum_fp64",
]


def pick_port() -> int:
    return random.randint(20000, 50000)


def parse_log_line(line: str) -> Dict[str, str]:
    if not line.startswith(LOG_PREFIX):
        return {}
    parts = line.strip().split(",")[1:]
    kv = {}
    for p in parts:
        if "=" in p:
            k, v = p.split("=", 1)
            kv[k] = v
    return kv


def write_csv_header(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(CSV_COLUMNS)


def append_csv_rows(path: Path, run_id: str, sync_every: int, lines: List[str]):
    with path.open("a", newline="") as f:
        writer = csv.writer(f)
        for line in lines:
            if not line.startswith(LOG_PREFIX):
                continue
            kv = parse_log_line(line)
            if not kv or "epoch" not in kv:
                continue

            row = []
            for col in CSV_COLUMNS:
                if col == "run_id":
                    row.append(run_id)
                elif col == "sync_every":
                    row.append(sync_every)
                else:
                    row.append(kv.get(col, ""))
            writer.writerow(row)


def run_experiment(args, sync_every: int, raw_log_path: Path):
    port = pick_port()

    cmd = [
        "torchrun",
        "--standalone",
        f"--nproc_per_node={args.nproc_per_node}",
        f"--master_port={port}",
        args.train_py,
        "--max_steps", str(args.max_steps),
        "--epochs", str(args.epochs),
        "--batch_size", str(args.batch_size),
        "--num_neighbors", *map(str, args.num_neighbors),
        "--hidden_channels", str(args.hidden_channels),
        "--num_layers", str(args.num_layers),
        "--lr", str(args.lr),
        "--weight_decay", str(args.weight_decay),
        "--dropout", str(args.dropout),
        "--seed", str(args.seed),
        "--sync_every", str(sync_every),
    ]

    print(f"\n=== sync_every={sync_every} | port={port} ===")
    print("CMD:", " ".join(cmd))

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    lines = []
    with raw_log_path.open("w") as f:
        for line in proc.stdout:
            lines.append(line)
            f.write(line)
            sys.stdout.write(line)
            sys.stdout.flush()

    proc.wait()
    return lines


def main():
    parser = argparse.ArgumentParser()

    # Sweep parameters
    parser.add_argument("--sync_values", type=int, nargs="+",
                        default=[1, 2, 4, 8])

    # torchrun
    parser.add_argument("--nproc_per_node", type=int, default=4)

    # train.py arguments
    parser.add_argument("--train_py", type=str, default="src/train.py")
    parser.add_argument("--max_steps", type=int, default=2000)
    parser.add_argument("--epochs", type=int, default=999)
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--num_neighbors", type=int, nargs="+", default=[25, 10])
    parser.add_argument("--hidden_channels", type=int, default=256)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=1234)

    # Logging
    parser.add_argument("--out_dir", type=str, default="logs")
    parser.add_argument("--run_id", type=str, default=None)

    args = parser.parse_args()

    if args.run_id is None:
        args.run_id = f"sync_sweep_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    out_dir = Path(args.out_dir) / args.run_id
    raw_dir = out_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    csv_path = out_dir / "metrics.csv"
    write_csv_header(csv_path)

    print("Run ID:", args.run_id)
    print("Output directory:", out_dir)

    for sync_every in args.sync_values:
        raw_log_path = raw_dir / f"sync_every_{sync_every}.log"
        lines = run_experiment(args, sync_every, raw_log_path)
        append_csv_rows(csv_path, args.run_id, sync_every, lines)

    print("\nSweep complete.")
    print("CSV:", csv_path)


if __name__ == "__main__":
    main()

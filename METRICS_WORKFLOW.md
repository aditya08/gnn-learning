# Metrics and Figure Workflow

This repository now supports a split workflow:
- run training on the cluster
- sync `logs/` to local machine
- merge all run CSVs into one deterministic file
- render PGFPlots figures locally

## 1) Remote (cluster): run experiments

Your current sweep script already writes:
- per-run folder: `logs/<run_id>/`
- per-run metrics: `logs/<run_id>/metrics.csv`
- raw logs: `logs/<run_id>/raw/*.log`

## 2) Local: sync metrics from cluster

Use rsync via:

```bash
make sync-metrics REMOTE_HOST=user@cluster REMOTE_LOGS_DIR=/path/to/gnn-learning/logs
```

Equivalent direct call:

```bash
REMOTE_HOST=user@cluster REMOTE_DIR=/path/to/gnn-learning/logs ./scripts/sync_metrics.sh
```

## 3) Local: merge per-run CSV files

```bash
make merge-metrics
```

This writes:
- merged file: `results/metrics.csv`
- de-dup key: `run_id,sync_every,epoch` (last row wins)
- deterministic row order for clean diffs

## 4) Local: build figures with LaTeX/PGFPlots

```bash
make figures
```

This builds from:
- source plot: `plots/sync_ablation.tex`
- input CSV: `results/metrics.csv`
- output directory: `plots/build/`

## 5) Recommended Git hygiene

- Commit code/config changes first.
- Run experiment tied to that commit.
- Sync logs locally and run merge.
- Commit metrics update separately from code.
- Keep LaTeX intermediates and built plot files out of Git.

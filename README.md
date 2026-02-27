# GNN Learning: Training, SLURM, and Plotting Workflow

This repo contains:
- Distributed GraphSAGE training (`src/train.py`)
- Sync strategy abstraction (`src/distributed/sync_schemes.py`)
- Metrics collection and merge utilities (`scripts/`)
- PGFPlots figure generation (`plots/`)
- Perlmutter SLURM script generator for strong scaling (`scripts/generate_slurm_strong_scaling.py`)

## Repository Layout

```text
src/
  train.py
  models/         # model definitions
  metrics/        # accuracy + parameter metrics helpers
  distributed/    # sync strategy abstractions
  runtime/        # launch/runtime helpers (DDP setup)
  experiments/    # local experiment runners
  data/           # data utilities
scripts/          # workflow utilities (sync/merge/plot/slurm generation)
plots/            # PGFPlots sources
results/          # merged metrics and derived CSV outputs
logs/             # per-run raw logs/metrics
```

## 1) Training

### Single-node local run

```bash
torchrun --standalone --nproc_per_node=4 src/train.py \
  --max_steps 4000 \
  --batch_size 1024 \
  --num_neighbors 25 10 \
  --sync_every 1 \
  --sync_scheme deferred_allreduce
```

Key training flags:
- `--max_steps`: total global optimizer steps target
- `--sync_every`: communication period
- `--sync_scheme`: currently `deferred_allreduce` (and scaffold `local_sgd`)
- `--dataset_root`: dataset location (defaults from `PYG_DATA_ROOT`/`$SCRATCH`)

## 2) Generate Perlmutter SLURM Scripts

Use the generator to create one `.sbatch` script per node count:

```bash
python3 scripts/generate_slurm_strong_scaling.py \
  --nodes-list 1 2 4 8 \
  --allocation m1234 \
  --mail-user you@org.edu \
  --mail-type END,FAIL \
  --job-name-prefix reddit-strong \
  --output-dir slurm \
  --setup-line "module load pytorch" \
  -- --max_steps 4000 --batch_size 1024 --sync_every 1 --sync_scheme deferred_allreduce
```

Important options:
- Metadata:
  - `--allocation` (`#SBATCH --account`)
  - `--mail-user`
  - `--mail-type` (e.g. `END,FAIL`)
- SLURM resources:
  - `--nodes-list`
  - `--time`
  - `--partition`, `--qos`, `--constraint`
  - `--gpus-per-node`, `--cpus-per-task`
- Runtime:
  - `--setup-line` (repeatable)
  - `--env KEY=VALUE` (repeatable)
  - trailing args after `--` are passed to `src/train.py`

Submit generated scripts:

```bash
sbatch slurm/reddit-strong_n1.sbatch
sbatch slurm/reddit-strong_n2.sbatch
sbatch slurm/reddit-strong_n4.sbatch
sbatch slurm/reddit-strong_n8.sbatch
```

## 3) Metrics Sync + Merge

This repo is set up for:
- Remote cluster execution writes logs under `logs/<run_id>/`
- Local machine syncs those logs and merges CSVs

Sync from cluster:

```bash
make sync-metrics REMOTE_HOST=user@perlmutter REMOTE_LOGS_DIR=/path/to/gnn-learning/logs
```

Merge all run CSVs into deterministic merged output:

```bash
make merge-metrics
```

Output:
- `results/metrics.csv`

## 4) Plotting

Build a specific PGF figure:

```bash
./scripts/build_figures.sh plots/sync_ablation.tex plots/build
```

Or via Makefile defaults:

```bash
make figures TEX_FILE=plots/sync_ablation.tex
```

Pareto frontier (accuracy vs communication):

```bash
make pareto
```

Outputs:
- `results/pareto_all_fp32.csv`
- `results/pareto_frontier_fp32.csv`
- `plots/build/pareto_frontier_fp32.pdf`

## 5) Remote Cluster + Local Machine Workflow

Recommended split workflow:

1. Local machine:
   - Edit code and commit.
   - Generate SLURM scripts with metadata (allocation/email/time/resources).
   - Push code to remote cluster filesystem (or use shared filesystem).

2. Remote cluster (Perlmutter):
   - Submit generated `.sbatch` scripts.
   - Jobs run distributed training with `torch.distributed.run`.
   - Each experiment writes logs/metrics under `logs/<run_id>/`.

3. Local machine:
   - Pull back logs with `make sync-metrics`.
   - Merge into one analysis file with `make merge-metrics`.
   - Build figures locally (LaTeX/PGFPlots) with `make figures` / `make pareto`.

Why this split is useful:
- Cluster time is used for GPU compute only.
- Plotting/LaTeX and iterative analysis stay local and fast.
- Reproducibility improves because merged metrics and plots are regenerated from synced raw outputs.

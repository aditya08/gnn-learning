SHELL := /bin/bash

# Cluster source (override on command line or env)
REMOTE_HOST ?=
REMOTE_LOGS_DIR ?=
LOCAL_LOGS_DIR ?= logs

MERGED_CSV ?= results/metrics.csv
TEX_FILE ?= plots/sync_ablation.tex
PLOT_BUILD_DIR ?= plots/build

.PHONY: sync-metrics merge-metrics figures all

all: merge-metrics figures

sync-metrics:
	REMOTE_HOST="$(REMOTE_HOST)" REMOTE_DIR="$(REMOTE_LOGS_DIR)" LOCAL_DIR="$(LOCAL_LOGS_DIR)" ./scripts/sync_metrics.sh

merge-metrics:
	python3 ./scripts/merge_metrics.py --input-root "$(LOCAL_LOGS_DIR)" --output "$(MERGED_CSV)"

figures:
	./scripts/build_figures.sh "$(TEX_FILE)" "$(PLOT_BUILD_DIR)"

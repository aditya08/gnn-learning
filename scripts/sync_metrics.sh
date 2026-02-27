#!/usr/bin/env bash
set -euo pipefail

# Sync experiment outputs from a remote cluster to local logs/.
# Usage:
#   REMOTE_HOST=user@cluster REMOTE_DIR=/path/to/repo/logs ./scripts/sync_metrics.sh
# Optional:
#   LOCAL_DIR=logs ./scripts/sync_metrics.sh

REMOTE_HOST="${REMOTE_HOST:-}"
REMOTE_DIR="${REMOTE_DIR:-}"
LOCAL_DIR="${LOCAL_DIR:-logs}"

if [[ -z "$REMOTE_HOST" || -z "$REMOTE_DIR" ]]; then
  echo "error: set REMOTE_HOST and REMOTE_DIR"
  echo "example: REMOTE_HOST=user@login REMOTE_DIR=/scratch/me/gnn-learning/logs $0"
  exit 1
fi

mkdir -p "$LOCAL_DIR"
rsync -avP --delete-delay \
  "${REMOTE_HOST}:${REMOTE_DIR%/}/" \
  "${LOCAL_DIR%/}/"

echo "synced remote logs to ${LOCAL_DIR}"

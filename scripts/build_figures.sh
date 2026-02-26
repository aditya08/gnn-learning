#!/usr/bin/env bash
set -euo pipefail

TEX_FILE="${1:-plots/sync_ablation.tex}"
OUT_DIR="${2:-plots/build}"

mkdir -p "$OUT_DIR"

if command -v latexmk >/dev/null 2>&1; then
  latexmk -pdf -interaction=nonstopmode -halt-on-error -output-directory="$OUT_DIR" "$TEX_FILE"
elif command -v pdflatex >/dev/null 2>&1; then
  pdflatex -interaction=nonstopmode -halt-on-error -output-directory="$OUT_DIR" "$TEX_FILE"
else
  echo "error: no LaTeX engine found (need latexmk or pdflatex on local machine)"
  exit 1
fi

echo "figure build complete: $OUT_DIR"

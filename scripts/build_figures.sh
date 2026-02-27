#!/usr/bin/env bash
set -euo pipefail

TEX_FILE="${1:-plots/sync_ablation.tex}"
OUT_DIR="${2:-plots/build}"

mkdir -p "$OUT_DIR"
OUT_DIR_ABS="$(cd "$OUT_DIR" && pwd)"
TEX_DIR="$(cd "$(dirname "$TEX_FILE")" && pwd)"
TEX_BASENAME="$(basename "$TEX_FILE")"

if command -v latexmk >/dev/null 2>&1; then
  latexmk -cd -pdf -interaction=nonstopmode -halt-on-error -output-directory="$OUT_DIR_ABS" "$TEX_FILE"
elif command -v pdflatex >/dev/null 2>&1; then
  (
    cd "$TEX_DIR"
    pdflatex -interaction=nonstopmode -halt-on-error -output-directory="$OUT_DIR_ABS" "$TEX_BASENAME"
  )
else
  echo "error: no LaTeX engine found (need latexmk or pdflatex on local machine)"
  exit 1
fi

echo "figure build complete: $OUT_DIR_ABS"

#!/usr/bin/env bash
# Build Upstream workbench tour PDF (compact article layout; no forced blank pages).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"
SRC="docs/N2S-Upstream-Workbench-Tour.md"
OUT="docs/N2S-Upstream-Workbench-Tour.pdf"
need() { command -v "$1" >/dev/null 2>&1 || { echo "Missing: $1" >&2; exit 1; }; }
need pandoc
[[ -f "$SRC" ]] || { echo "Missing $SRC" >&2; exit 1; }

PDF_ENGINE=""
for eng in lualatex xelatex pdflatex; do
  if command -v "$eng" >/dev/null 2>&1; then PDF_ENGINE="$eng"; break; fi
done
[[ -n "$PDF_ENGINE" ]] || { echo "Need a LaTeX engine" >&2; exit 1; }

echo "Building $OUT with $PDF_ENGINE …"
pandoc "$SRC" -o "$OUT" --pdf-engine="$PDF_ENGINE" \
  --resource-path=docs \
  -V documentclass=article \
  -V classoption=oneside \
  -V mainfont="DejaVu Sans" -V monofont="DejaVu Sans Mono" \
  -V geometry:margin=0.8in -V fontsize=10pt \
  -V colorlinks=true \
  -V linkcolor=black -V urlcolor=blue -V toccolor=black \
  --toc --toc-depth=2 \
  -V toc-title="Contents"
ls -lh "$OUT"
pages=$(pdfinfo "$OUT" 2>/dev/null | awk '/^Pages:/ {print $2}')
echo "Done: $ROOT/$OUT (${pages:-?} pages)"

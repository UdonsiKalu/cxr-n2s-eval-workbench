#!/usr/bin/env bash
# Build newcomer tour PDF (theory + screenshots). Does not touch CLI walkthrough PDF.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"
SRC="docs/N2S-Workbench-Newcomer-Tour.md"
OUT="docs/N2S-Workbench-Newcomer-Tour.pdf"
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
  -V mainfont="DejaVu Sans" -V monofont="DejaVu Sans Mono" \
  -V geometry:margin=0.7in -V fontsize=10pt \
  --toc --toc-depth=2
ls -lh "$OUT"
echo "Done: $ROOT/$OUT"

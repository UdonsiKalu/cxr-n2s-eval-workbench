#!/usr/bin/env bash
# Build CLI walkthrough PDF — same code-block styling as cxr-mi-repeng-grounding.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"
OUT="docs/N2S-CLI-Walkthrough.pdf"
COMBINED="docs/_walkthrough-combined.md"
need() { command -v "$1" >/dev/null 2>&1 || { echo "Missing: $1" >&2; exit 1; }; }
need pandoc
PDF_ENGINE=""
for eng in lualatex xelatex pdflatex; do
  if command -v "$eng" >/dev/null 2>&1; then PDF_ENGINE="$eng"; break; fi
done
[[ -n "$PDF_ENGINE" ]] || { echo "Need a LaTeX engine" >&2; exit 1; }

FILES=(
  walkthrough/README.md
  walkthrough/ROADMAP.md
  walkthrough/lessons/W00-welcome.md
  walkthrough/lessons/W01-atoms-traps.md
  walkthrough/lessons/W02-evaluate-contra.md
  walkthrough/lessons/W03-evaluate-planned.md
  walkthrough/lessons/W04-evaluate-temporal-review.md
  walkthrough/lessons/W05-deepdive-label-rule.md
  walkthrough/lessons/W06-intervene-control.md
  walkthrough/lessons/W07-intervene-flip.md
  walkthrough/lessons/W08-claim-hygiene.md
)

cleanup() {
  rm -f "$COMBINED" \
    docs/_walkthrough-combined.aux docs/_walkthrough-combined.log \
    docs/_walkthrough-combined.out docs/_walkthrough-combined.toc \
    docs/_walkthrough-combined.tex docs/_walkthrough-combined.synctex.gz \
    2>/dev/null || true
}
trap cleanup EXIT

{
  cat <<'EOF'
---
title: "N2S Eval — CLI Walkthrough"
subtitle: "Implement sketches + optional demo runner (no browser)"
author: "cxr-n2s-eval-workbench/walkthrough"
date: "September 2026"
documentclass: extarticle
fontsize: 9pt
geometry: margin=0.65in
linestretch: 1.05
header-includes:
  - |
    ```{=latex}
    \usepackage{etoolbox}
    \AtBeginEnvironment{Shaded}{\footnotesize}
    \usepackage{titlesec}
    \titlespacing*{\section}{0pt}{1.4ex plus .2ex}{0.7ex plus .1ex}
    \titlespacing*{\subsection}{0pt}{1.1ex plus .2ex}{0.5ex plus .1ex}
    ```
---

\newpage

# How to use this PDF

**Goal:** Learn the N2S Eval stack **without the GUI** — same idea as the grounding PDF’s Intuition → Minimal pattern → Send-back.

| Section | Purpose |
|---------|---------|
| **Minimal pattern** | Short **Python** you should be able to rewrite / whiteboard |
| **Demo runner** | Optional `walk_n2s.py` one-liners that call the same APIs |

**This PDF** is the CLI walkthrough (`cxr-n2s-eval-workbench/walkthrough/`).

**Not this PDF:** `docs/N2S-Workbench-Newcomer-Tour.pdf` (theory + screenshots + SAE UI), `cxr-mi-repeng-grounding/` (foundations ladder), or the portfolio curriculum PDF.

**Ladder:**

```text
W00–W01   Path setup + symbolic evaluate_rule (no LLM)
W02–W04   evaluate_fast / Dual / N2S mismatch
W05       run_forensics_live (observe L20)
W06–W07   run_intervene_live + HF steer hook
W08       Claim hygiene + run_workbench_sae sketch
```

**Demo runner only** (optional):

```bash
cd cxr-n2s-eval-workbench
python3 walk_n2s.py list
python3 walk_n2s.py run W02 --model qwen2.5-coder:32b
python3 walk_n2s.py run W06 --alpha 8
```

Prefer HF phases with `../cxrlabs/faiss_gpu1/bin/python`.

**Send-backs:** `W00 done` … `W08 done`.

\newpage

EOF
  first=1
  for f in "${FILES[@]}"; do
    [[ -f "$f" ]] || { echo "Missing $f" >&2; exit 1; }
    if [[ "$first" -eq 0 ]]; then printf '\n\\newpage\n\n'; fi
    first=0
    cat "$f"
    printf '\n'
  done
} > "$COMBINED"

echo "Building $OUT with pandoc + $PDF_ENGINE …"

PANDOC_ARGS=(
  "$COMBINED"
  -o "$OUT"
  --pdf-engine="$PDF_ENGINE"
  --toc
  --toc-depth=2
  -V documentclass=extarticle
  -V fontsize=9pt
  -V geometry:margin=0.65in
  -V linestretch=1.05
  -V colorlinks=true
  -V linkcolor=blue
  -V urlcolor=blue
  -V toccolor=black
  --highlight-style=tango
  -f markdown
)

if [[ "$PDF_ENGINE" == "pdflatex" ]]; then
  python3 - <<'PY'
from pathlib import Path
p = Path("docs/_walkthrough-combined.md")
t = p.read_text(encoding="utf-8")
repl = {
    "→": "->", "←": "<-", "↔": "<->", "—": "--", "–": "-",
    "“": '"', "”": '"', "‘": "'", "’": "'", "…": "...",
    "≈": "~=", "×": "x", "§": "Section ", "≠": "!=",
    "·": "-", "α": "alpha",
}
for a, b in repl.items():
    t = t.replace(a, b)
p.write_text(t, encoding="utf-8")
PY
  pandoc "${PANDOC_ARGS[@]}"
else
  pandoc "${PANDOC_ARGS[@]}" \
    -V mainfont="DejaVu Sans" \
    -V monofont="DejaVu Sans Mono"
fi

ls -lh "$OUT"
echo "Done: $ROOT/$OUT"

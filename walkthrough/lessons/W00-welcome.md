# W00 — Welcome

**Track:** N2S Eval CLI walkthrough · **Next:** W01

## Intuition

Someone asks: *walk me through the project* **and** *could you implement this without the GUI?*

- **Demo runner** `walk_n2s.py` — one command that exercises the real stack (visitor path).  
- **Minimal pattern** (from W01 on) — short Python you should be able to rewrite: Dual / gates, L20 forensics, `h ← h + α·v`, SAE top-k.

This tree is the **demo + implementer** walkthrough. It is not the long MI foundations ladder (`cxr-mi-repeng-grounding/`) and not the portfolio PDF (`cxr-repeng-curriculum/`).

```text
Evaluate  →  Deep dive  →  Intervene (alpha)  [+ SAE in tour / W08 sketch]
Track A      observe L20     causal steer + controls
```

## Demo runner (optional)

```bash
cd cxr-n2s-eval-workbench
python3 walk_n2s.py list
python3 walk_n2s.py show W00
python3 walk_n2s.py run W02 --model qwen2.5-coder:32b
python3 walk_n2s.py run W06 --alpha 8
```

## Minimal pattern — path setup (every later snippet)

Run from `cxr-n2s-eval-workbench/`. Lab code lives one directory up.

```python
import sys
from pathlib import Path

ROOT = Path(".").resolve()                 # cxr-n2s-eval-workbench
LAB = ROOT.parent / "cxr-evidence-grounding-lab"
sys.path[:0] = [str(ROOT), str(LAB)]
```

Prefer HF GPU phases with `../cxrlabs/faiss_gpu1/bin/python` (not bare system Python).

## Companion reading

- Theory + screenshots + SAE UI: `docs/N2S-Workbench-Newcomer-Tour.pdf`
- Frozen α=8 claim: `../cxr-evidence-grounding-lab/docs/TRACKB-ALPHA8-FREEZE.md`

## Send-back

**W00 done** — one sentence: demo runner vs Minimal pattern (where the learning is).

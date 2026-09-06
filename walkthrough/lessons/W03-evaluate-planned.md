# W03 — Evaluate: planned only

**Prerequisite:** W02 · **Next:** W04 · **Action:** `evaluate` · **Note:** `EX_PLANNED`

## Intuition

Therapy planned next week, none given yet → atom **B** false → **NOT_SATISFIED**.

Same `evaluate_fast` entry as W02; only the note changes. Deep dive later marks representation–commitment **NOT EVALUATED** when commitment is NOT_SATISFIED (W05).

## Demo runner (optional)

```bash
python3 walk_n2s.py run W03
```

## Minimal pattern — same API, planned note

```python
import sys
from pathlib import Path

ROOT = Path(".").resolve()
sys.path[:0] = [str(ROOT), str(ROOT.parent / "cxr-evidence-grounding-lab")]
import eval_api

NOTE = (
    "Newly diagnosed metastatic colon cancer. Multidisciplinary plan is to start "
    "first-line FOLFOX next week. No systemic therapy has been administered yet."
)

fast = eval_api.evaluate_fast(
    {
        "note": NOTE,
        "gold": "NOT_SATISFIED",
        "case_id": "SKETCH_PLANNED",
        "model": "qwen2.5-coder:32b",
        "unload_after": True,
    }
)
print(fast["disposition"], fast["verdict"])
# Typical: AUTO NOT_SATISFIED  (B false after ground → evaluate_rule)
```

## Send-back

**W03 done** — paste disposition + verdict; name which atom fails.

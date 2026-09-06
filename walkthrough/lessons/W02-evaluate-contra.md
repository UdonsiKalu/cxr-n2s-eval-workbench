# W02 — Evaluate: true contradiction

**Prerequisite:** W01 · **Next:** W03 · **Action:** `evaluate` · **Note:** `EX_CONTRA`

## Intuition

Same-day conflict should AUTO as CONTRADICTION when Dual and grounding agree.

Track A wiring (what you implement / call):

```text
note → analyze → path C extract + path D extract
    → ground / verify / evidence → dual_path_verdict
    → N2S mismatch (neural X vs grounded X) → apply_gates → AUTO|REVIEW
```

## Demo runner (optional)

```bash
python3 walk_n2s.py run W02 --model qwen2.5-coder:32b
# dry wiring:  python3 walk_n2s.py run W02 --model mock
```

## Minimal pattern — Evaluate without the GUI

```python
import sys
from pathlib import Path

ROOT = Path(".").resolve()  # cxr-n2s-eval-workbench
sys.path[:0] = [str(ROOT), str(ROOT.parent / "cxr-evidence-grounding-lab")]

import eval_api

NOTE = (
    "Note A: Patient failed first-line FOLFOX after four cycles. "
    "Addendum same day: Disease remains responsive to FOLFOX; continue current regimen."
)

fast = eval_api.evaluate_fast(
    {
        "note": NOTE,
        "gold": "CONTRADICTION",
        "case_id": "SKETCH_CONTRA",
        "model": "qwen2.5-coder:32b",  # or "mock"
        "unload_after": True,
    }
)

print(fast["disposition"], fast["verdict"])
print("mismatch agree:", (fast.get("mismatch") or {}).get("agree"))
print("paths C/D:", fast["paths"]["C_full"]["verdict"], fast["paths"]["D_full"]["verdict"])
```

Expect roughly:

```text
AUTO CONTRADICTION
mismatch agree: True
```

## Core idea — Dual + mismatch (whiteboard)

```python
from n2s_lab.roundtrip import dual_path_verdict

# After both paths have a symbolic verdict string:
dual = dual_path_verdict(verdict_c, verdict_d)
# dual["disposition"] is AUTO only if C and D agree on a non-REVIEW verdict

# N2S mismatch (simplified — see eval_api._mismatch):
neural_x = extraction["contradiction"]["present"]   # extractor claim
grounded_x = grounding["contradiction"]             # after ground()
if neural_x is True and grounded_x is False:
    disposition = "REVIEW"  # classic temporal false-X containment
```

## Send-back

**W02 done** — paste disposition + verdict, and name the two functions Dual vs mismatch use.

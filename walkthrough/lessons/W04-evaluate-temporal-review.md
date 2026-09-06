# W04 — Evaluate: temporal containment

**Prerequisite:** W03 · **Next:** W05 · **Action:** `evaluate` · **Note:** `EX_TEMPORAL_FOLFOX`

## Intuition

Clinically: response then later progression → gold **SATISFIED**, not contradiction.

Track A often **REVIEW**s when the extractor sets neural X=true while `ground` clears X as temporal-change (**N2S mismatch**). That is containment, not product failure.

## Demo runner (optional)

```bash
python3 walk_n2s.py run W04
```

## Minimal pattern — read mismatch explicitly

```python
import sys
from pathlib import Path

ROOT = Path(".").resolve()
sys.path[:0] = [str(ROOT), str(ROOT.parent / "cxr-evidence-grounding-lab")]
import eval_api

NOTE = (
    "Metastatic colorectal cancer. First-line FOLFOX produced a partial response. "
    "At follow-up four months later, imaging demonstrated new hepatic lesions "
    "consistent with progression. FOLFOX was discontinued for treatment failure; "
    "second-line therapy discussed."
)

fast = eval_api.evaluate_fast(
    {
        "note": NOTE,
        "gold": "SATISFIED",
        "case_id": "SKETCH_TEMPORAL",
        "model": "qwen2.5-coder:32b",
        "unload_after": True,
    }
)

mm = fast.get("mismatch") or {}
print(fast["disposition"], fast["verdict"])
print("neural_x", mm.get("neural_contradiction"), "grounded_X", mm.get("grounded_X"))
print("agree", mm.get("agree"), mm.get("reasons"))
# Soft expect: REVIEW + agree False  OR  AUTO SATISFIED if X already clean
```

## Teaching point

Do not patch every paste to force AUTO. Investigate Dual / mismatch; REVIEW is OK.

## Send-back

**W04 done** — paste headline + neural_x / grounded_X / agree.

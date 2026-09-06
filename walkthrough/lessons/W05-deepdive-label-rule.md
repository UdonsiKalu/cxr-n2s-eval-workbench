# W05 — Deep dive: observe L20

**Prerequisite:** W04 · **Next:** W06 · **Action:** `deep_dive` · **Note:** `EX_PLANNED`

## Intuition

Deep dive **observes**. Round-trip checks + HF Qwen 7B layer scores (L20 temporal-change vs contradiction). It does **not** change the model.

**Label rule:** agree/mismatch comparison only when commitment is **CONTRADICTION**.  
For **NOT_SATISFIED** / **SATISFIED** → **NOT EVALUATED**.

Needs ~14 GiB free VRAM (unload Ollama first).

## Demo runner (optional)

```bash
python3 walk_n2s.py run W05
```

## Minimal pattern — forensics live (HF hooks under the hood)

```python
import sys
from pathlib import Path

ROOT = Path(".").resolve()
sys.path[:0] = [str(ROOT), str(ROOT.parent / "cxr-evidence-grounding-lab")]

import eval_api
from n2s_lab.n2s_forensics import format_forensics_text, run_forensics_live
from n2s_lab.ollama_client import unload_model

NOTE = (
    "Newly diagnosed metastatic colon cancer. Multidisciplinary plan is to start "
    "first-line FOLFOX next week. No systemic therapy has been administered yet."
)

# 1) Track A first (optional but matches UI)
fast = eval_api.evaluate_fast(
    {
        "note": NOTE,
        "gold": "NOT_SATISFIED",
        "case_id": "SKETCH_DD",
        "model": "qwen2.5-coder:32b",
        "unload_after": True,
    }
)
print(fast["disposition"], fast["verdict"])

# 2) Free Ollama VRAM, then HF L20 readout
unload_model("qwen2.5-coder:32b")

panel = run_forensics_live(
    NOTE,
    case_id="SKETCH_DD",
    gold="NOT_SATISFIED",
    final_commitment=fast["verdict"],
    unload_after=True,
)
print(panel.get("status") or panel.get("ok"))
print(format_forensics_text(panel) if panel.get("ok") else panel.get("message"))
# Look for L20 soft scores + "NOT EVALUATED" when commitment is NOT_SATISFIED
```

Workbench UI path is the same idea via `eval_api._deep_dive_work(fast)` (round-trip + forensics).

## Send-back

**W05 done** — paste L20 soft scores (or deferred_gpu) + comparison line.

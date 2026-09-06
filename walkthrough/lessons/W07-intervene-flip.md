# W07 — Intervene: selective flip

**Prerequisite:** W06 · **Next:** W08 · **Action:** `intervene` · **Note:** `EX_TEMPORAL_FOLFOX` · **alpha:** 8

## Intuition

On some temporal false-X notes, baseline X=true → full_vector X=false while gaussian stays true and reverse does not help.

That is the live demo of the limited expand editor. Soft: if baseline already X=false, pick a harder expand fixture (e.g. TX_E14).

## Demo runner (optional)

```bash
python3 walk_n2s.py run W07 --alpha 8
```

## Minimal pattern — same API, temporal note + controls

```python
import sys
from pathlib import Path

ROOT = Path(".").resolve()
sys.path[:0] = [str(ROOT), str(ROOT.parent / "cxr-evidence-grounding-lab")]

from n2s_lab.n2s_intervene import format_intervene_text, run_intervene_live

NOTE = (
    "Metastatic colorectal cancer. First-line FOLFOX produced a partial response. "
    "At follow-up four months later, imaging demonstrated new hepatic lesions "
    "consistent with progression. FOLFOX was discontinued for treatment failure; "
    "second-line therapy discussed."
)

panel = run_intervene_live(
    NOTE,
    case_id="SKETCH_TEMPORAL",
    gold="SATISFIED",
    alpha=8.0,
    include_controls=True,  # gaussian_unit + reverse_vector
    unload_after=True,
)

print(format_intervene_text(panel) if panel.get("ok") else panel)
by = {a["label"]: a for a in (panel.get("arms") or [])}
for name in ("baseline", "full_vector", "gaussian_unit", "reverse_vector"):
    a = by.get(name) or {}
    print(f"{name:16} X={a.get('repair_final_x')}  margin={a.get('margin')}")

# Credible story: baseline true → full_vector false; gaussian true; reverse not helping
```

## Send-back

**W07 done** — paste arms table; say whether flip happened and what the controls did.

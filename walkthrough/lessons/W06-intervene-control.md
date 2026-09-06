# W06 — Intervene: control (no flip)

**Prerequisite:** W05 · **Next:** W07 · **Action:** `intervene` · **Note:** `EX_CONTRA` · **alpha:** 8

## Intuition

Steer `h ← h + alpha · v` at L20 on the `contradiction.present` commit token. True contradictions must **stay** X=true.

Arms: baseline, full_vector, gaussian_unit, reverse_vector.

## Demo runner (optional)

```bash
python3 walk_n2s.py run W06 --alpha 8
```

## Minimal pattern — expand editor API

```python
import sys
from pathlib import Path

ROOT = Path(".").resolve()
sys.path[:0] = [str(ROOT), str(ROOT.parent / "cxr-evidence-grounding-lab")]

from n2s_lab.n2s_intervene import format_intervene_text, run_intervene_live

NOTE = (
    "Note A: Patient failed first-line FOLFOX after four cycles. "
    "Addendum same day: Disease remains responsive to FOLFOX; continue current regimen."
)

panel = run_intervene_live(
    NOTE,
    case_id="SKETCH_CONTRA",
    gold="CONTRADICTION",
    alpha=8.0,
    include_controls=True,
    unload_after=True,
)
print(format_intervene_text(panel) if panel.get("ok") else panel)
# Expect: baseline X=true and full_vector X=true (no true→false flip)
for arm in panel.get("arms") or []:
    print(arm["label"], "X=", arm.get("repair_final_x"), "margin=", arm.get("margin"))
```

## Core idea — HF forward hook (what the library does)

```python
# Sketch of n2s_lab/hf_intervene.py activation_steer (Qwen: model.model.layers[i])
def steer_hook(_module, _inp, out, *, vec, alpha):
    hs = out[0] if isinstance(out, tuple) else out
    modified = hs.clone()
    # last position = commit token when steer_active
    modified[0, -1, :] = modified[0, -1, :] + alpha * vec
    return (modified,) + out[1:] if isinstance(out, tuple) else modified

# handle = layers[20].register_forward_hook(...)
# v = unit(mean_A − mean_B) from artifacts/n2s-forensics-directions-qwen7b.json
```

Frozen limited claim @ α=8 — see lab `TRACKB-ALPHA8-FREEZE.md`.

## Send-back

**W06 done** — paste four-arm X / margin; state the hook equation in one line.

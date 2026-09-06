# W08 — Claim hygiene (+ SAE sketch)

**Prerequisite:** W07

## Say

- Track A contains uncertain transforms via REVIEW / Dual / mismatch.  
- Track B shows a **partial** L20 α=8 editor with controls on expand-style temporal false-X.  
- You can demo with `walk_n2s.py` **or** call `evaluate_fast` / `run_forensics_live` / `run_intervene_live` directly.

## Do not say

- Steering fixed healthcare / all temporal false-X.  
- α=16/32 is the freeze.  
- 14B MI editor transfer (behavioral transfer only; cluster did not persist).  
- Sealed `temporal-family-test.json` was used for fitting.  
- SAE ranking = “we found the temporality neuron.”

## What is next (optional)

- **SAE features** — Chanin L20 decompose **v**; workbench `:8257` or sketch below.  
- Upstream prefill + thinner circuits — lab HF hooks.  
- Foundations ladder: `cxr-mi-repeng-grounding/`.  
- Portfolio spine: `cxr-repeng-curriculum/`.

## Demo runner (optional)

```bash
python3 walk_n2s.py show W08
python3 walk_n2s.py list
```

## Minimal pattern — SAE without the GUI

```python
import sys
from pathlib import Path

ROOT = Path(".").resolve()
sys.path[:0] = [str(ROOT), str(ROOT.parent / "cxr-evidence-grounding-lab")]

from n2s_lab.n2s_sae_pilot import run_workbench_sae

NOTE = (
    "Metastatic colorectal cancer. First-line FOLFOX produced a partial response. "
    "At follow-up four months later, imaging demonstrated new hepatic lesions "
    "consistent with progression. FOLFOX was discontinued for treatment failure."
)

# Prefer: ../cxrlabs/faiss_gpu1/bin/python
panel = run_workbench_sae(
    note=NOTE,
    case_id="SKETCH_SAE",
    gold="SATISFIED",
    top_k=3,
    alpha=8.0,
    include_contra_control=True,  # EX_CONTRA must not flip true→false
)
# Ranked feats: Δ(A−B), cos_to_v, track X/margin — then top-k steers
for row in (panel.get("candidates") or [])[:5]:
    print(row.get("feat_id"), row.get("delta_A_minus_B"), row.get("cos_to_v"))
```

Lab docs: `docs/TRACKB-SAE-PILOT.md`. Ranking ≠ causal proof; read the arms table.

## Send-back

**W08 done** — one sentence claim you would tell a visitor (Track A + limited α=8 + SAE optional).

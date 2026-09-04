# N2S Interactive Eval Workbench

Paste a clinical note → **fast Track A evaluation** (extract → ground → verify → Dual → AUTO/REVIEW).  
Optional **Deep dive** expands round-trip forensics — not full HF MI on every note.

**Port:** **8257**  
**Companion lab:** clone/use sibling `../cxr-evidence-grounding-lab/`

## Run

```bash
# Needs Ollama for live models (or pick Mock in the UI)
cd cxr-n2s-eval-workbench
python3 server.py
# → http://127.0.0.1:8257/
```

## What it does

| Action | Path |
|--------|------|
| **Evaluate** | C_full + D_full + Dual_full + extract-vs-ground X mismatch → AUTO or REVIEW |
| **Deep dive** | Round-trip + **N2S forensics** (HF layer scores: temporal-change vs contradiction). If GPU is busy (~15 GiB needed), shows a frozen **example** panel and defers live capture. |

Forensics directions: `../cxr-evidence-grounding-lab/artifacts/n2s-forensics-directions-qwen7b.json` (L20 Class A/B from expand).

Predicate: first-line therapy failed (same as the evidence-grounding lab).

## Related

- Lab UI `:8253` · RepEng Workbench `:8256`
- Public demos: [grounding](https://udonsikalu.github.io/cxr-evidence-grounding-lab/) · [workbench](https://udonsikalu.github.io/cxr-repeng-workbench/)

# N2S Eval Workbench

Interactive Track A evaluation on port **8257**.

Sibling lab: `../cxr-evidence-grounding-lab` (`n2s_lab`, artifacts).

## Run

```bash
cd cxr-n2s-eval-workbench
python3 server.py   # http://127.0.0.1:8257/
```

## Flow

1. **Evaluate** — Ollama extract → ground → Dual → AUTO/REVIEW (then verified Ollama unload)
2. **Deep dive** — round-trip + HF L20 forensics (when ~14 GiB free)
3. **Intervene α** — HF 7B expand L20 steer: baseline + full_vector @ α (default **8**) + optional gaussian/reverse controls

Intervene uses the frozen expand direction (`n2s-forensics-directions-qwen7b.json`), **not** the Phase-10 BC_E* vector on `:8256`. Limited α=8 claim — partial editor, not family-wide.

## Links

- Lab UI `:8253` · RepEng Workbench `:8256`

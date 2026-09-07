# N2S Eval Workbench

Interactive Track A evaluation on port **8257**.

Sibling lab: `../cxr-evidence-grounding-lab` (`n2s_lab`, artifacts).

## Run (browser UI)

```bash
cd cxr-n2s-eval-workbench
python3 server.py   # http://127.0.0.1:8257/
```

## Run (CLI walkthrough — no browser)

Visitor / project walkthrough in the terminal (same Evaluate → Deep dive → Intervene stack):

```bash
cd cxr-n2s-eval-workbench
python3 walk_n2s.py list
python3 walk_n2s.py run W02 --model qwen2.5-coder:32b
python3 walk_n2s.py run W06 --alpha 8
python3 walk_n2s.py tour --through W04
```

Lessons live in `walkthrough/`. Build PDFs:

- `./build-walkthrough-pdf.sh` → `docs/N2S-CLI-Walkthrough.pdf`
- `./build-newcomer-tour-pdf.sh` → `docs/N2S-Workbench-Newcomer-Tour.pdf` (Downstream :8257 theory + screenshots; includes SAE)
- `./build-upstream-tour-pdf.sh` → `docs/N2S-Upstream-Workbench-Tour.pdf` (Upstream :8258 background + UI + code sketches)

## Flow

1. **Evaluate** — Ollama extract → ground → Dual → AUTO/REVIEW (then verified Ollama unload)
2. **Deep dive** — round-trip + HF L20 forensics (when ~14 GiB free)
3. **Intervene α** — HF 7B expand L20 steer: baseline + full_vector @ α (default **8**) + optional gaussian/reverse controls
4. **SAE features** — Chanin L20 SAE: rank vs frozen `v`, then top-k feature steers on the pasted note (+ EX_CONTRA control)

**Upstream** (formation map / live score) is a **separate** workbench on **8258** — not on this port:

```bash
../cxrlabs/faiss_gpu1/bin/python upstream_server.py   # http://127.0.0.1:8258/
```

SAE uses the same lab pilot as `../cxr-evidence-grounding-lab` (`n2s_sae_pilot`). Prefer running the server with **faiss_gpu1**:

```bash
../cxrlabs/faiss_gpu1/bin/python server.py
```

## Links

- Lab UI `:8253` · RepEng Workbench `:8256` · Upstream `:8258`
- Upstream program: `../cxr-evidence-grounding-lab/docs/TRACKB-UPSTREAM-PROGRAM.md`

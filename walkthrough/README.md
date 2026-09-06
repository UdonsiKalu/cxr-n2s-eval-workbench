# N2S Eval — CLI walkthrough

**Visitor walkthrough of the project — terminal only, no browser.**

Same phases as the UI on port 8257 (CLI covers through Intervene; SAE is in the newcomer tour / lab):

```text
Evaluate  →  Deep dive  →  Intervene (alpha)
Track A      observe L20     causal steer + controls
```

## Quick start

```bash
cd cxr-n2s-eval-workbench

python3 walk_n2s.py list
python3 walk_n2s.py show W00
python3 walk_n2s.py run W02 --model qwen2.5-coder:32b
python3 walk_n2s.py run W06 --alpha 8
python3 walk_n2s.py tour --through W04
```

Mock Evaluate (no Ollama):

```bash
python3 walk_n2s.py run W02 --model mock
```

## Layout

| Path | Role |
|------|------|
| `walk_n2s.py` | CLI entry |
| `walkthrough/lessons/W00…W08.md` | Lesson text: Intuition → Minimal pattern (Python) → optional Demo runner |
| `walkthrough/lessons.json` | Machine-readable actions / expects |
| `walkthrough/notes.json` | Bound clinical notes |
| `docs/N2S-Workbench-Newcomer-Tour.pdf` | Background theory + screenshots |
| `./build-walkthrough-pdf.sh` | Build CLI walkthrough PDF (tango / Shaded like grounding) |
| `./build-newcomer-tour-pdf.sh` | Build theory + screenshots PDF (incl. SAE) |

## Requirements

| Lessons | Needs |
|---------|--------|
| W00–W01, W08 | Nothing |
| W02–W04 | Ollama (or `--model mock`) |
| W05 | Ollama + ~14 GiB free for HF 7B forensics |
| W06–W07 | ~14 GiB free for HF 7B intervene |

Sibling lab must sit at `../cxr-evidence-grounding-lab`.

## Not this track

- `cxr-mi-repeng-grounding/` — long foundations ladder  
- `cxr-repeng-curriculum/` — portfolio / interview PDF  
- Browser UI (`python3 server.py`) — optional; CLI is the walkthrough path

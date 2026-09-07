---
title: "N2S Upstream Workbench — Complete Tour"
subtitle: "Background · experiments · :8258 UI · code · claim hygiene"
author: "CXR N2S lab notes (port 8258)"
date: "September 2026"
geometry: margin=0.8in
fontsize: 10pt
mainfont: DejaVu Sans
monofont: DejaVu Sans Mono
abstract: |
  Complete Upstream write-up: background concepts, every experiment (U0→U-A→gen→causal-d→:8258),
  UI controls, and code — with a full decode of $d = \mathrm{unit}(\mu_T-\mu_C)$.
  Twin Downstream tour (port 8257): N2S-Workbench-Newcomer-Tour.pdf.
  Lab-scale only — not clinical validation.
header-includes:
  - \usepackage{titling}
  - \setlength{\droptitle}{-3.5em}
  - \pretitle{\begin{center}\Large\bfseries}
  - \posttitle{\par\end{center}\vspace{-0.6em}}
  - \preauthor{\begin{center}\small}
  - \postauthor{\end{center}}
  - \predate{\begin{center}\small}
  - \postdate{\end{center}\vspace{-0.4em}}
  - \usepackage{titlesec}
  - \titlespacing*{\section}{0pt}{1.1em}{0.45em}
  - \titlespacing*{\subsection}{0pt}{0.85em}{0.35em}
  - \setlength{\parskip}{0.35em}
  - \pagestyle{plain}
  - \renewcommand{\abstractname}{About this document}
---

# Part I — Background (keep these concepts clear)

## 1. The clinical question

We care about one formal predicate:

> Does the note satisfy **`FIRST_LINE_THERAPY_FAILED`**?

Notes can be:

| Kind | Idea | Example shape |
|------|------|----------------|
| **Temporal-change** | Response / benefit **then later** progression | FOLFOX helps → months later grows → stop |
| **Contradiction** | Same-time conflict | “Failed” and “responsive” in one visit |
| **Clear fail / nofail** | Explicit progression or ongoing response | Straightforward E-notes |

A common failure mode: the model marks **`contradiction.present = true` (X=true)** on a *temporal* note that is **not** a true same-time contradiction (**false X**).

## 2. Three pieces of the N2S system

```text
Doctor's note
      ↓  UPSTREAM — formation of internal reps (THIS DOCUMENT · :8258)
Tokenization → layer-wise compute → residual stream
      ↓  DOWNSTREAM — utilization → structured LLM output (:8257)
Generation / commit (e.g. contradiction.present)
════════════════════════════
   NEURAL → SYMBOLIC BOUNDARY  (Track A · hard cut)
════════════════════════════
CXR atoms / Dual / gates → AUTO | REVIEW
```

| Piece | Question | Port |
|-------|----------|------|
| **Upstream** | How does the representation **form** in prefill? | **8258** |
| **Downstream** | How is it **used** at commit / can we steer it? | **8257** |
| **Boundary** | Can we contain bad transforms as REVIEW? | **8257** Evaluate |

Upstream vs downstream is an **experimental partition**, not a hard cut inside the transformer. The only hard product cut is neural → symbolic.

## 3. Prefill vs commit (must not mix)

| Stage | When | What we measure |
|-------|------|-----------------|
| **Prefill** | Model reads the whole prompt (note + instructions) once | Residuals at note-body tokens × layers → **formation** |
| **Commit** | During generation, when writing `contradiction.present: true/false` | Logits / margin / X → **decision** |

Upstream Score projects **prefill** activations onto **d**.  
Downstream Intervene adds α·**v** at **commit**.  
Same model; different question.

## 4. Activations, layers, residual stream

A transformer layer outputs a high-dimensional vector per token (hidden size ≈ 3584 for Qwen2.5-7B).

- **Layer Ln** = after block *n* (we use absolute indices: L8, L12, L16, L20, L24).  
- **Residual stream** = the running representation those blocks write into.  
- **Attn / MLP** = two writers inside a block (attention and feed-forward).  

We capture these with HF **forward hooks** (callbacks on module outputs). We do **not** require TransformerLens for this work.

## 5. Directions in activation space (RepEng idea)

If a concept lives as a **direction**, you can:

1. **Read** — project activations onto that direction (Score / Deep dive)  
2. **Write** — add a multiple of that direction (Intervene)  
3. **Ablate** — zero or replace a component write (U-B, circuit C1)

Reading ≠ writing. A strong Score does **not** prove an editor.

## 6. Glossary — how to read `d = unit(μ_T − μ_C)`

This is the Upstream formation direction. Read it left to right:

| Symbol | Plain English |
|--------|----------------|
| **T** | Temporal-change discovery note (`EX_TEMPORAL_FOLFOX`) — response then later failure |
| **C** | Same-time contradiction discovery note (`EX_CONTRA`) |
| **h** | Residual vector at one token, one layer (length D) |
| **μ_T** | **Mean** of h over **all note-body tokens** of the temporal note at that layer |
| **μ_C** | Same mean for the contradiction note |
| **μ_T − μ_C** | Difference vector: points toward “more like temporal mean / less like contra mean” |
| **unit(·)** | Divide by Euclidean length → length **1**. Direction only; removes overall scale |
| **d** | That unit difference, **per layer** (so there is a d@L8, d@L12, … d@L24) |

**Bake (once, then freeze):**

```text
for each layer L:
    μ_T = mean over tokens of residual_T[L]
    μ_C = mean over tokens of residual_C[L]
    d[L] = (μ_T − μ_C) / ||μ_T − μ_C||
```

Cached as `artifacts/n2s-upstream-ua-directions.pt`.

**Score a new note:**

```text
for each note-body token t at layer L:
    score = residual[t,L] · d[L]     # dot product
best layer ≈ argmax mean |score|
```

| Quantity on UI | Meaning |
|----------------|---------|
| **\|μ·d\|** | Mean absolute token·d at that layer (how strongly the note aligns with d) |
| **Best layer** | Layer with largest \|μ·d\| (usually **L24** for temporal-ish notes) |
| **cos(d, v)** | Angle between Upstream **d** and Downstream freeze-**v** (~0.03 → almost unrelated) |
| **tok / score_d** | Which token pieces light up along **d** (correlational hotspots) |
| **cos_v** | That token’s cosine to Downstream **v** (usually near 0) |

**Forbidden reading:** “These top tokens *are* the temporality circuit.” They are **correlational** sites.

## 7. Glossary — Downstream sibling `v = unit(μ_A − μ_B)`

Do not confuse with **d**.

| Symbol | Upstream **d** | Downstream **v** |
|--------|----------------|------------------|
| Built from | Prefill note means (temporal vs contra) | Commit hiddens Class A vs B |
| Strongest layer | **L24** | **L20** |
| Used for | Formation Score / map | Steer / SAE / circuit |
| Causal editor? | **No** (ablate + α·d null) | **Partial** (α=8) + L20 MLP site |

```text
v = unit( mean(h_A) − mean(h_B) )   # expand fit @ L20 commit
h ← h + α · v                         # Intervene at contradiction.present
```

## 8. X, margin, soft gates

| Term | Meaning |
|------|---------|
| **X** | `contradiction.present` boolean in the JSON extract |
| **Margin** | logit(true) − logit(false) at the commit token |
| **Soft gate** | Pre-registered pass/fail rule for a panel (not fishing after the fact) |
| **Null / weak** | Soft gate failed — no honest causal claim |

# Part II — Everything we did (Upstream science)

## Timeline at a glance

| Step | Name | Result | Status |
|------|------|--------|--------|
| U0 | Prefill cue × layer · freeze-v | Weak cue·v | Done |
| U1 / U1b | Prefill ±α·v at cue / L16 sites | **No X flip** | Done · stop α-chase |
| **U-A** | Multi-token class-mean **d** map | Strongest **L24**; cos(d,v)≈0.03 | Done |
| **U-B** | Zero-ablate attn/mlp/resid @ L24 sites | **Null** | Done · paused |
| **U-B2** | Mean-ablate @ L20 sites | **Null** | Done · paused |
| **U-A gen** | Held-out paraphrases vs frozen **d** | **soft+strong YES** @ L24 | **FROZEN** |
| **U-A causal-d** | Steer α·d @ L24 (commit + prefill) | **NULL/weak** | Done |
| **`:8258`** | Live Score / Frozen replay / Custom pair | Correlational UI | Live |
| U-C / U-D | Upstream SAE / circuits | **Not run** — locked off nulls | Locked |

## U0 — Prefill trace

**Quest:** Do freeze-**v** (commit editor) sit on obvious note cues during prefill?

**Method:** One prefill forward; cue-token residuals at layers {8,12,16,20,24}; project onto freeze-**v**.

**Result:** Weak. Commit-fitted **v** is **not** obviously “sitting on” recipe cues in prefill.

**Artifact / doc:** `TRACKB-UPSTREAM-PREFILL.md` · `n2s_upstream_prefill.py`

## U1 / U1b — Prefill position steer

**Quest:** If we add ±α·v at cue positions on prefill, does commit X flip?

**Method:** `prefill_position_steer` in `hf_intervene.py`; α=8 at recipe / L16 outcome sites.

**Result:** **Null** — no X flip; margins flat. Contra stays true.

**Decision:** Stop α-chase on freeze-**v** × cue. Formation ≠ automatic commit editor.

## U-A — Formation map (the core Upstream positive)

**Quest:** Where do temporal vs contradiction notes separate in prefill residual space?

**Method:**

1. Capture note-body residuals for discovery pair.  
2. Per layer: `d = unit(μ_T − μ_C)`.  
3. Rank layers by ‖μ_T − μ_C‖ / mean |token·d|.  
4. List top tokens by |score|.

**Result:**

| Finding | Detail |
|---------|--------|
| Best layer | **L24** |
| cos(d, freeze-v) | ≈ **0.03** |
| Story | Late-layer formation map; **not** the same object as Downstream **v** |

**Artifacts:** `n2s-upstream-ua-map.json` · `n2s-upstream-ua-directions.pt` · `n2s-upstream-ua-readout.json`  
**CLI:** `./scripts/run_upstream_ua.sh map`

**Formation hypothesis (allowed):**  
Treatment-course vs contradiction geometry becomes visible late in prefill (esp. L24); it is correlationally distinct from the L20 commit editor **v**.

## U-B / U-B2 — Ablation (causal test #1) → NULL

**Quest:** Are U-A top sites **necessary** for commit X?

| Variant | Layer | Mode | Result |
|---------|-------|------|--------|
| U-B | L24 | Zero attn / mlp / resid at top sites | **No X flip** |
| U-B2 | L20 | Mean-ablate component writes | **No X flip** |

**Claim:** Localization alone ≠ control. Ablation family **paused**.  
**Do not** open U-C SAE / U-D from these nulls.

**Artifacts:** `n2s-upstream-ub-patch.json` · `n2s-upstream-ub2-patch.json` (+ readouts)

## U-A gen — Paraphrase generalization → FROZEN YES

**Quest:** Does frozen **d** (baked only on discovery pair) separate **held-out** near/far paraphrases?

**Gates:**

| Gate | Meaning | Result |
|------|---------|--------|
| Soft | mean_T > mean_C @ L24 | **YES** (~58.4 > ~23.5) |
| Strong | min_T > max_C @ L24 | **YES** |

**Claim:** Correlational **lexical generalization** of the formation map.  
**Not** a causal editor.

**Freeze:** `TRACKB-UPSTREAM-UA-GEN-FREEZE.md`  
**Data:** `data/heldout-ua-paraphrase.json`

## U-A causal-d — Steer along frozen d → NULL/weak

**Quest:** Does α·d edit X / margin dose-dependently?

**Method:** Commit + prefill steer @ L24; α ∈ {1,2,4,8}; ±d + gaussian; FOLFOX + CONTRA.

**Result:** **NULL/weak** both sites — no X flip; margin noise; contra held.  
(Note: FOLFOX baseline already X=false on some runs — dull for false-X clear, still no dose editor.)

**Distinct from U-B:** direction steer vs top-site ablate; **both** null families.

**Doc:** `TRACKB-UPSTREAM-UA-CAUSAL.md`

## Portfolio plain English

> I traced where a treatment-failure representation becomes visible inside a transformer, then tested whether the strongest locations (or the direction itself) controlled the model’s decision. They did not. Localization alone was insufficient for causality. The map still **generalizes** to held-out paraphrases and lights up real temporal notes on `:8258`.

# Part III — Workbench UI (:8258)

## Run

```bash
cd cxr-n2s-eval-workbench
../cxrlabs/faiss_gpu1/bin/python upstream_server.py
# → http://127.0.0.1:8258/
```

Needs ~14 GiB free GPU for Score / pair.

## Left panel — every control

| Control | What it does | What it does **not** |
|---------|--------------|----------------------|
| **Case id** | Labels the JSON artifact | Change the model |
| **Optional gold** | Metadata only | Drive Score math |
| **Clinical note** | Input for **Score** | Used by Frozen replay |
| **Score vs frozen d** | Project this note onto frozen **d** | Give AUTO / SATISFIED |
| **Frozen U-A/U-B replay** | Show lab discovery map + null ablations | Score *your* paste |
| **Temporal / Contra boxes** | Only for **Map custom pair** | Affect Score |
| **Map custom pair** | Bake a **new** d′ from two pastes | Replace frozen discovery d |
| **Example notes** | Load short EX_TEMPORAL / EX_CONTRA | |

## Right panel — how to read Score

1. **Best layer** (green) — usually L24  
2. **Bar chart** — mean |token·d| by layer (ramp L8→L24 is expected for temporal notes)  
3. **Top sites** — tokenizer pieces with largest |score_d|  
4. **Artifact path** — JSON under `artifacts/n2s-upstream-live-runs/`

## Worked example — long FOLFOX note

Paste: metastatic CRC, FOLFOX response → later hepatic progression → discontinue (hedged with neuropathy).

**Live Score (`LIVE_3c8c30b5`):**

| Signal | Value |
|--------|-------|
| Tokens | ~737 note tokens · dirs cached |
| Best | **L24** \|μ·d\| ≈ **54** |
| Ramp | L8≈5 → L20≈29 → L24≈54 |
| cos(d,v) | ~0.03 |
| Top toks | `atic` / `metast` / `pulmonary` / … (correlational) |

**Verdict:** Score **worked**. Formation alignment is strong.  
**Not** a clinical Dual decision — that is `:8257` Evaluate.

**Frozen replay after this?** **No** — it ignores the paste.

## Custom pair map

Builds `d′ = unit(μ_T′ − μ_C′)` from *your* two notes. Heavier (two HF captures). Still correlational. Use for teaching “map on my pair,” not as the frozen discovery **d**.

# Part IV — Code path (what powers the UI)

```text
browser  static/upstream.html
    │  GET  /api/meta  /api/frozen  /api/job
    │  POST /api/score  /api/pair
    ▼
upstream_server.py          # ThreadingHTTPServer :8258
    ▼
upstream_api.py             # jobs, JSON shaping, claim_hygiene
    ▼
lab n2s_lab/
  n2s_upstream_live.py      # ensure_ua_directions, score_note_live, run_custom_pair_map
  n2s_upstream_prefill.py   # DEFAULT_NOTES, capture helpers
  hf_intervene.py           # prefill steers / ablate (lab U1/U-B; not Score buttons)
artifacts/
  n2s-upstream-ua-directions.pt
  n2s-upstream-ua-map.json / ub*.json
  n2s-upstream-live-runs/*.json
```

## Bake / load d

```python
# n2s_upstream_live.ensure_ua_directions
# If directions.pt exists → load cache.
# Else capture EX_TEMPORAL_FOLFOX + EX_CONTRA, then:
for L in (8, 12, 16, 20, 24):
    d = unit(mean(h_T[L]) - mean(h_C[L]))
torch.save(blob, "artifacts/n2s-upstream-ua-directions.pt")
```

## Live Score

```python
dirs = ensure_ua_directions(...)
captured = _capture_note(note)          # note-body residuals only
for layer, d in dirs.items():
    scores = residuals[layer] @ d       # each token
    # mean |scores|, top-k tokens, cos(d, freeze_v)
best_layer = argmax mean_abs_score
```

## API sketch

```python
# POST /api/score  → background job → score_note_live(...)
# GET  /api/job    → poll until result
# GET  /api/frozen → slim U-A map + U-B/U-B2 null tables (no HF)
# POST /api/pair   → run_custom_pair_map(temporal, contra)
```

## Browser sketch

```javascript
fetch('/api/score', { method:'POST', body: JSON.stringify({
  evidence: note.value, case_id, gold
})});
// poll /api/job → renderScore(bars + top sites)
```

# Part V — How Upstream sits next to Downstream (do not mix claims)

| | Upstream (:8258 / lab) | Downstream (:8257 / lab) |
|--|------------------------|---------------------------|
| Object | **d** @ L24 | **v** @ L20 |
| Read | Score ·d | Deep dive |
| Write | α·d **null** | α·v **partial** (α=8) |
| Ablate | U-B null | Circuit C1 L20 MLP **causal site** |
| Sparse | U-C **locked off** | SAE pilot on **v** |
| Boundary | — | Track A Dual wrong_AUTO=0 |

**Circuit C0–C2 (Downstream, frozen):** top write L20/mlp along **v**; C1 soft_gate YES; C2 LOCAL_SUFFICIENT.  
That does **not** reopen Upstream U-C/U-D.

# Part VI — Claim hygiene

## Say

- U-A formation map; strongest correlational sep @ **L24**  
- cos(d, v) ≈ 0.03 — formation ≠ commit editor object  
- Held-out paraphrase gen **soft+strong YES** (correlational)  
- U-B / U-B2 / α·d **null** — no upstream editor  
- `:8258` Score is a **correlational** live demo  
- Track A contains uncertain transforms via REVIEW  
- Downstream: α=8 partial editor + L20 MLP causal **site** (not full circuit)

## Do not say

- Found the temporality neuron / full circuit  
- Upstream editor / production clinical safety  
- Score top tokens = mechanism  
- Frozen replay scored my note  
- Open U-C/U-D from nulls or from Downstream C0–C2  
- α-chase expand-v; sealed-test redesign from these panels  

# Part VII — Reproduce / pointers

```bash
# UI
cd cxr-n2s-eval-workbench
../cxrlabs/faiss_gpu1/bin/python upstream_server.py

# Lab science
cd cxr-evidence-grounding-lab
./scripts/run_upstream_ua.sh map
./scripts/run_upstream_ub.sh
./scripts/run_upstream_ua_gen.sh panel
./scripts/run_upstream_ua_causal.sh panel --site commit

# Rebuild this PDF
cd cxr-n2s-eval-workbench
./build-upstream-tour-pdf.sh
```

| Doc | Role |
|-----|------|
| This file | Complete Upstream tour |
| `TRACKB-UPSTREAM-PORTFOLIO.md` | One-page portfolio |
| `TRACKB-UPSTREAM-PROGRAM.md` | Program + hard stop |
| `TRACKB-UPSTREAM-UA-GEN-FREEZE.md` | Gen freeze |
| `TRACKB-UPSTREAM-UA-CAUSAL.md` | α·d null |
| `TRACKB-UPSTREAM-PREFILL.md` | U0/U1 |
| `N2S-SYSTEM-CLAIM.md` | System snapshot |
| `N2S-Workbench-Newcomer-Tour.md` | Downstream twin |
| `TRACKB-CIRCUIT-C01-FREEZE.md` | Downstream circuit freeze |

## Closing line

**Upstream:** we can **see** and **generalize** a late-layer formation direction **d**, and Score real notes against it on `:8258` — but we could **not** turn that map into a causal prefill editor.

**Downstream:** we can **partially edit** commit utilization with **v** @ L20 and point to an L20 MLP **site**.

Two ports. Two objects (**d** vs **v**). Keep the math and the claims separate — and always decode `unit(μ_T − μ_C)` as: *unit difference of mean note-body residuals between temporal and contradiction discovery notes.*

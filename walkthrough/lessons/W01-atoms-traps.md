# W01 — Atoms and two traps

**Prerequisite:** W00 · **Next:** W02

## Intuition

`FIRST_LINE_THERAPY_FAILED` — did first-line therapy fail?

| Atom | Meaning |
|------|---------|
| A | First-line therapy identified |
| B | Actually given (not only planned) |
| C | Failure / progression event |
| D | Failure of that first-line |
| X | Same-time incompatible claims → CONTRADICTION |

**Two traps**

1. **True contradiction** — same visit: failed *and* still responding.  
2. **Temporal change** — responded earlier, progressed later. Gold often **SATISFIED**, not contradiction. Models may wrongly set **X=true**.

**N2S in one line:** neural extract → symbolic `ground` → Dual paths C/D → gates → AUTO or REVIEW.

The **neural** side fills atoms. The **symbolic** rule alone decides the verdict — no LLM required for this lesson.

## Demo runner (optional)

```bash
python3 walk_n2s.py show W01
```

## Minimal pattern — symbolic rule (no LLM)

```python
import sys
from pathlib import Path

LAB = Path("../cxr-evidence-grounding-lab").resolve()
sys.path.insert(0, str(LAB))

from n2s_lab.types import Atom, Grounding
from n2s_lab.predicate import evaluate_rule

# Planned only → B false → NOT_SATISFIED
planned = Grounding(
    first_line_identified=Atom.TRUE,
    first_line_administered=Atom.FALSE,  # B
    failure_event=Atom.UNKNOWN,
    failure_of_first_line=Atom.UNKNOWN,
    contradiction=False,
)
print(evaluate_rule(planned).verdict)  # NOT_SATISFIED

# Same-day conflict → X → CONTRADICTION (short-circuit)
contra = Grounding(
    first_line_identified=Atom.TRUE,
    first_line_administered=Atom.TRUE,
    failure_event=Atom.TRUE,
    failure_of_first_line=Atom.TRUE,
    contradiction=True,  # X
)
print(evaluate_rule(contra).verdict)  # CONTRADICTION

# Temporal success story atoms (no X) → SATISFIED
temporal = Grounding(
    first_line_identified=Atom.TRUE,
    first_line_administered=Atom.TRUE,
    failure_event=Atom.TRUE,
    failure_of_first_line=Atom.TRUE,
    contradiction=False,
)
print(evaluate_rule(temporal).verdict)  # SATISFIED
```

Or: `python3 -c "from n2s_lab.predicate import selftest; selftest()"` from the lab root.

## Send-back

**W01 done** — why temporal change is not a contradiction, and what `evaluate_rule` returns for B=false vs X=true.

#!/usr/bin/env python3
"""N2S Eval CLI walkthrough — optional demo runner (no browser).

Learning lives in walkthrough/lessons/*.md Minimal pattern Python sketches.
This script only exercises the same APIs for a visitor one-liner.

Usage (from cxr-n2s-eval-workbench/):
  python3 walk_n2s.py list
  python3 walk_n2s.py show W02
  python3 walk_n2s.py run W02 [--model ...]
  python3 walk_n2s.py run W06 --alpha 8
  python3 walk_n2s.py tour [--through W04] [--model ...]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
WT = ROOT / "walkthrough"
LESSONS_DIR = WT / "lessons"
sys.path.insert(0, str(ROOT))


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _catalog() -> dict[str, Any]:
    return _load_json(WT / "lessons.json")


def _notes() -> dict[str, Any]:
    return _load_json(WT / "notes.json")


def _lesson_by_id(lid: str) -> dict[str, Any]:
    cat = _catalog()
    for lesson in cat["lessons"]:
        if lesson["id"].upper() == lid.upper():
            return lesson
    raise SystemExit(f"Unknown lesson {lid!r}. Try: python3 walk_n2s.py list")


def _print_md(path: Path) -> None:
    print(path.read_text(encoding="utf-8"))


def cmd_list(_: argparse.Namespace) -> None:
    cat = _catalog()
    print(f"Track: {cat['track']}")
    print(f"Companion: {cat.get('companion_pdf')}")
    print()
    print(f"{'ID':<5} {'Phase':<12} {'Action':<12} {'GPU':<4} {'Title'}")
    print("-" * 78)
    for L in cat["lessons"]:
        gpu = "yes" if L.get("requires_gpu") else "no"
        print(
            f"{L['id']:<5} {L['phase']:<12} {L.get('action','none'):<12} "
            f"{gpu:<4} {L['title']}"
        )
    print()
    print("Run:  python3 walk_n2s.py run W02")
    print("Show: python3 walk_n2s.py show W02")


def cmd_show(args: argparse.Namespace) -> None:
    L = _lesson_by_id(args.lesson_id)
    path = LESSONS_DIR / L["file"]
    print("=" * 72)
    print(f"{L['id']}: {L['title']}")
    print("=" * 72)
    _print_md(path)
    if L.get("note_id"):
        note = _notes().get(L["note_id"]) or {}
        print("-" * 72)
        print(f"Bound note {L['note_id']}  gold={note.get('gold')}")
        print(note.get("evidence", "")[:500])
        print("-" * 72)


def _fmt_evaluate(result: dict[str, Any]) -> str:
    mm = result.get("mismatch") or {}
    s = result.get("summary") or {}
    lines = [
        "=== EVALUATE ===",
        f"case_id:      {result.get('case_id')}",
        f"model:        {result.get('model')}  live={result.get('live')}",
        f"headline:     {s.get('headline')}",
        f"disposition:  {result.get('disposition')}",
        f"verdict:      {result.get('verdict')}",
        f"C / D / Dual: {s.get('c_verdict')} / {s.get('d_verdict')} / {s.get('dual_verdict')}",
        f"N2S mismatch: agree={mm.get('agree')}  neural_X={mm.get('neural_contradiction')}  "
        f"grounded_X={mm.get('grounded_X')}",
        f"gates:        {result.get('gates')}",
        f"artifact:     {result.get('artifact')}",
    ]
    return "\n".join(lines)


def _fmt_deep(report: dict[str, Any]) -> str:
    lines = ["=== DEEP DIVE ===", f"status/keys: {list(report.keys())[:12]}"]
    forensics = report.get("forensics") or {}
    ascii_block = forensics.get("ascii") or (report.get("forensics_example") or {}).get("ascii")
    if ascii_block:
        lines.append(ascii_block)
    else:
        lines.append(json.dumps({
            "forensics_status": forensics.get("status") or forensics.get("ok"),
            "message": forensics.get("message"),
            "commitment": forensics.get("final_commitment") or report.get("commitment"),
        }, indent=2))
    if report.get("artifact"):
        lines.append(f"artifact: {report['artifact']}")
    return "\n".join(lines)


def _arm_x(arm: dict[str, Any]) -> Any:
    return arm.get("repair_final_x", arm.get("raw_x"))


def _fmt_intervene(panel: dict[str, Any]) -> str:
    from n2s_lab.n2s_intervene import format_intervene_text  # type: ignore

    if panel.get("ascii"):
        body = panel["ascii"]
    elif panel.get("ok"):
        body = format_intervene_text(panel)
    else:
        body = json.dumps(panel, indent=2)
    lines = ["=== INTERVENE ===", body]
    arms = panel.get("arms") or []
    if arms:
        lines.append("")
        lines.append(f"{'arm':<16} {'X':<8} {'margin'}")
        lines.append("-" * 40)
        for a in arms:
            lines.append(
                f"{str(a.get('label') or a.get('intervention')):<16} "
                f"{str(_arm_x(a)):<8} {a.get('margin')}"
            )
    if panel.get("artifact"):
        lines.append(f"artifact: {panel.get('artifact')}")
    return "\n".join(lines)


def _check_expect(action: str, payload: dict[str, Any], expect: dict[str, Any] | None) -> None:
    if not expect:
        return
    if expect.get("soft"):
        print(f"[expect] soft: {expect.get('note')}")
        return
    if action == "evaluate":
        ok_d = payload.get("disposition") == expect.get("disposition")
        ok_v = payload.get("verdict") == expect.get("verdict")
        print(f"[expect] disposition {'OK' if ok_d else 'DIFF'} "
              f"(got {payload.get('disposition')}, want {expect.get('disposition')})")
        print(f"[expect] verdict     {'OK' if ok_v else 'DIFF'} "
              f"(got {payload.get('verdict')}, want {expect.get('verdict')})")
    elif action == "intervene":
        by = {a.get("label") or a.get("intervention"): a for a in (payload.get("arms") or [])}
        # labels from _summarize_arm use label field
        def find(*names: str) -> dict[str, Any] | None:
            for n in names:
                for k, v in by.items():
                    if k and n in str(k):
                        return v
            return None

        base = find("baseline") or (payload.get("arms") or [None])[0]
        full = find("full_vector", "full")
        if base is not None and "baseline_x" in expect:
            got = _arm_x(base)
            want = expect["baseline_x"]
            print(f"[expect] baseline X {'OK' if got == want else 'DIFF'} (got {got}, want {want})")
        if full is not None and "full_vector_x" in expect:
            got = _arm_x(full)
            want = expect["full_vector_x"]
            print(f"[expect] full_vector X {'OK' if got == want else 'DIFF'} (got {got}, want {want})")


def _run_evaluate(note: dict[str, Any], model: str) -> dict[str, Any]:
    import eval_api

    return eval_api.evaluate_fast(
        {
            "note": note["evidence"],
            "gold": note.get("gold") or "UNKNOWN",
            "case_id": f"WALK_{note['id']}",
            "model": model,
            "unload_after": True,
        }
    )


def _run_deep_dive(note: dict[str, Any], model: str) -> tuple[dict[str, Any], dict[str, Any]]:
    import eval_api

    fast = _run_evaluate(note, model)
    print(_fmt_evaluate(fast))
    print()
    print("[walk] deep dive (sync) …")
    deep = eval_api._deep_dive_work(fast)
    return fast, deep


def _run_intervene(note: dict[str, Any], alpha: float, model_hint: str) -> dict[str, Any]:
    # Free Ollama if possible before HF
    try:
        from n2s_lab.ollama_client import unload_model

        if model_hint and model_hint != "mock":
            freed = unload_model(model_hint)
            print(
                f"[walk] pre-intervene ollama unload {model_hint}: "
                f"verified={freed.get('verified')}"
            )
    except Exception as exc:  # noqa: BLE001
        print(f"[walk] ollama unload skipped: {exc}")

    from n2s_lab.n2s_intervene import format_intervene_text, run_intervene_live
    from n2s_lab.paths import ARTIFACTS_DIR
    import uuid
    from datetime import datetime, timezone

    panel = run_intervene_live(
        note["evidence"],
        case_id=f"WALK_{note['id']}",
        gold=note.get("gold") or "",
        alpha=alpha,
        include_controls=True,
        unload_after=True,
    )
    if panel.get("ok"):
        panel["ascii"] = format_intervene_text(panel)
    runs = ARTIFACTS_DIR / "n2s-eval-runs"
    runs.mkdir(parents=True, exist_ok=True)
    path = runs / f"walk-intervene-{note['id']}-{uuid.uuid4().hex[:6]}.json"
    path.write_text(
        json.dumps({**panel, "timestamp": datetime.now(timezone.utc).isoformat()}, indent=2),
        encoding="utf-8",
    )
    panel["artifact"] = str(path)
    return panel


def cmd_run(args: argparse.Namespace) -> None:
    L = _lesson_by_id(args.lesson_id)
    print("=" * 72)
    print(f"RUN {L['id']}: {L['title']}")
    print("=" * 72)
    brief = LESSONS_DIR / L["file"]
    # print first ~25 lines of lesson for context
    lines = brief.read_text(encoding="utf-8").splitlines()
    for line in lines[:28]:
        print(line)
    print("…")
    print()

    action = L.get("action") or "none"
    if action == "none":
        print("[walk] theory lesson — no model call.")
        print(f"Full text: python3 walk_n2s.py show {L['id']}")
        return

    note_id = L.get("note_id")
    if not note_id:
        raise SystemExit(f"{L['id']} has action={action} but no note_id")
    note = _notes()[note_id]
    model = args.model
    alpha = float(args.alpha if args.alpha is not None else L.get("alpha") or 8.0)

    if action == "evaluate":
        print(f"[walk] evaluate note={note_id} model={model}")
        result = _run_evaluate(note, model)
        print(_fmt_evaluate(result))
        _check_expect(action, result, L.get("expect"))
    elif action == "deep_dive":
        print(f"[walk] deep_dive note={note_id} model={model}")
        _fast, deep = _run_deep_dive(note, model)
        print(_fmt_deep(deep))
        _check_expect(action, deep, L.get("expect"))
    elif action == "intervene":
        print(f"[walk] intervene note={note_id} alpha={alpha}")
        panel = _run_intervene(note, alpha, model)
        print(_fmt_intervene(panel))
        _check_expect(action, panel, L.get("expect"))
    else:
        raise SystemExit(f"Unknown action {action}")


def cmd_tour(args: argparse.Namespace) -> None:
    """Run consecutive lessons up to --through (inclusive)."""
    cat = _catalog()
    stop = (args.through or "W04").upper()
    ids = [L["id"] for L in cat["lessons"]]
    if stop not in ids:
        raise SystemExit(f"--through {stop} not in {ids}")
    for L in cat["lessons"]:
        ns = argparse.Namespace(
            lesson_id=L["id"],
            model=args.model,
            alpha=args.alpha,
        )
        cmd_run(ns)
        print()
        if L["id"] == stop:
            break
    print(f"[tour] stopped after {stop}")


def main() -> None:
    p = argparse.ArgumentParser(
        description="N2S Eval CLI walkthrough (Evaluate → Deep dive → Intervene). No browser."
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    sp = sub.add_parser("list", help="List lessons")
    sp.set_defaults(func=cmd_list)

    sp = sub.add_parser("show", help="Print lesson markdown + bound note")
    sp.add_argument("lesson_id")
    sp.set_defaults(func=cmd_show)

    sp = sub.add_parser("run", help="Execute one lesson (prints results)")
    sp.add_argument("lesson_id")
    sp.add_argument(
        "--model",
        default="qwen2.5-coder:32b",
        help="Ollama model for Evaluate/Deep dive (use mock for dry wiring)",
    )
    sp.add_argument("--alpha", type=float, default=None, help="Intervene strength (default 8)")
    sp.set_defaults(func=cmd_run)

    sp = sub.add_parser("tour", help="Run W00… through a lesson id")
    sp.add_argument("--through", default="W04", help="Stop after this lesson (default W04)")
    sp.add_argument("--model", default="qwen2.5-coder:32b")
    sp.add_argument("--alpha", type=float, default=None)
    sp.set_defaults(func=cmd_tour)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()

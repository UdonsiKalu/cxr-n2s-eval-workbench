"""Interactive N2S eval — fast Track A path + on-demand deep dive.

Sibling lab: ../cxr-evidence-grounding-lab (n2s_lab, artifacts).
"""

from __future__ import annotations

import json
import sys
import threading
import traceback
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
LAB_ROOT = ROOT.parent / "cxr-evidence-grounding-lab"
if str(LAB_ROOT) not in sys.path:
    sys.path.insert(0, str(LAB_ROOT))

from n2s_lab.auto_contract import apply_gates, classify_case  # noqa: E402
from n2s_lab.evidence import gate_verdict_with_evidence  # noqa: E402
from n2s_lab.ground import ground  # noqa: E402
from n2s_lab.neural import (  # noqa: E402
    analyze_free_text,
    extract_from_analysis,
    extract_from_analysis_repair,
    extract_live,
    extract_live_repair,
    extract_mock,
)
from n2s_lab.ollama_client import DEFAULT_MODEL, ollama_reachable, unload_model  # noqa: E402
from n2s_lab.paths import ARTIFACTS_DIR  # noqa: E402
from n2s_lab.predicate import PREDICATE_ID, PREDICATE_TEXT, evaluate_rule  # noqa: E402
from n2s_lab.roundtrip import dual_path_verdict, verify_roundtrip  # noqa: E402
from n2s_lab.types import Extraction, Verdict  # noqa: E402
from n2s_lab.verify import verify_formalization  # noqa: E402

PORT = 8257
STATIC_DIR = ROOT / "static"
RUNS_DIR = ARTIFACTS_DIR / "n2s-eval-runs"

EXAMPLE_NOTES = [
    {
        "id": "EX_BC_E1",
        "label": "BC_E1-style — response then later failure (not contradiction)",
        "gold": "SATISFIED",
        "evidence": (
            "Metastatic NSCLC on first-line pembrolizumab + carboplatin/pemetrexed. "
            "Initial imaging showed partial response. At the 6-month restaging visit, "
            "scans now show progression of known lesions with new nodal disease. "
            "Plan: discontinue current first-line regimen and discuss second-line options."
        ),
    },
    {
        "id": "EX_CONTRA",
        "label": "True contradiction — failed vs still responding",
        "gold": "CONTRADICTION",
        "evidence": (
            "Note A: Patient failed first-line FOLFOX after four cycles. "
            "Addendum same day: Disease remains responsive to FOLFOX; continue current regimen."
        ),
    },
    {
        "id": "EX_PLANNED",
        "label": "Planned only — not yet given",
        "gold": "NOT_SATISFIED",
        "evidence": (
            "Newly diagnosed metastatic colon cancer. Multidisciplinary plan is to start "
            "first-line FOLFOX next week. No systemic therapy has been administered yet."
        ),
    },
]

_lock = threading.Lock()
_job: dict[str, Any] = {
    "id": None,
    "kind": None,
    "state": "idle",
    "step": None,
    "result": None,
    "error": None,
    "log": [],
}


def _log(msg: str) -> None:
    ts = datetime.now(timezone.utc).strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    _job["log"] = (_job.get("log") or [])[-40:] + [line]
    print(f"[n2s-eval] {msg}", flush=True)


def meta_payload() -> dict[str, Any]:
    return {
        "port": PORT,
        "lab_root": str(LAB_ROOT),
        "predicate_id": PREDICATE_ID,
        "predicate_text": PREDICATE_TEXT,
        "ollama_reachable": ollama_reachable(),
        "default_model": DEFAULT_MODEL,
        "models": [
            {"id": "llama3:8b-instruct-q4_0", "label": "Llama 3 8B (fast)", "backend": "ollama"},
            {"id": "mistral:instruct", "label": "Mistral instruct", "backend": "ollama"},
            {"id": "qwen2.5-coder:32b", "label": "Qwen 2.5 coder 32B (Track A primary)", "backend": "ollama"},
            {"id": "qwen2.5:14b", "label": "Qwen 2.5 14B", "backend": "ollama"},
            {"id": "mock", "label": "Mock (no LLM — heuristic extract)", "backend": "mock"},
        ],
        "examples": EXAMPLE_NOTES,
        "capabilities": {
            "behavioral": True,
            "activations": False,
            "unload_ollama_after_run": True,
            "note": (
                "Fast path = Track A (extract→ground→verify→Dual→AUTO/REVIEW). "
                "After Evaluate/Deep dive, Ollama VRAM is freed (keep_alive=0) by default "
                "and verified via /api/ps before returning. "
                "Deep dive adds HF forensics when GPU free. "
                "Pass unload_after=false to keep the model warm."
            ),
        },
    }


def _verified_pipeline(
    *,
    source_text: str,
    analysis_for_verify: str,
    model: str,
    case_id: str,
    gold: str,
    live: bool,
    from_analysis: bool,
) -> dict[str, Any]:
    if not live:
        extraction = extract_mock(source_text)
        grounding = ground(extraction)
        rule = evaluate_rule(grounding)
        return {
            "extraction": extraction.to_dict(),
            "grounding": grounding.to_dict(),
            "verify": {"ok": True, "mode": "mock_skip"},
            "verify_attempts": 0,
            "rule": rule.to_dict(),
            "verdict": rule.verdict.value,
            "disposition": "AUTO",
            "path": "mock",
        }

    if from_analysis:
        extraction = extract_from_analysis(analysis_for_verify, model=model)
    else:
        extraction = extract_live(source_text, model=model)
    grounding = ground(extraction)
    v1 = verify_formalization(
        analysis=analysis_for_verify,
        extraction=extraction,
        grounding=grounding.to_dict(),
        case_id=case_id,
        gold=gold,
    )
    attempts = 1
    verify_payload: dict[str, Any] = {"first": v1.to_dict()}
    ok = v1.ok
    if not ok:
        if from_analysis:
            extraction = extract_from_analysis_repair(
                analysis_for_verify,
                model=model,
                prior=extraction,
                verify_reasons=v1.reasons,
            )
        else:
            extraction = extract_live_repair(
                source_text,
                model=model,
                prior=extraction,
                verify_reasons=v1.reasons,
            )
        grounding = ground(extraction)
        v2 = verify_formalization(
            analysis=analysis_for_verify,
            extraction=extraction,
            grounding=grounding.to_dict(),
            case_id=case_id,
            gold=gold,
        )
        attempts = 2
        verify_payload["second"] = v2.to_dict()
        ok = v2.ok

    if not ok:
        return {
            "extraction": extraction.to_dict(),
            "grounding": grounding.to_dict(),
            "verify": verify_payload,
            "verify_attempts": attempts,
            "rule": None,
            "verdict": Verdict.REVIEW.value,
            "disposition": "REVIEW",
            "note": "verification failed — REVIEW",
        }

    rule = evaluate_rule(grounding)
    return {
        "extraction": extraction.to_dict(),
        "grounding": grounding.to_dict(),
        "verify": verify_payload,
        "verify_attempts": attempts,
        "rule": rule.to_dict(),
        "verdict": rule.verdict.value,
        "disposition": "AUTO",
    }


def _mismatch(extraction: dict[str, Any], grounding: dict[str, Any]) -> dict[str, Any]:
    """Lightweight N2S mismatch: neural contradiction claim vs grounded X."""
    neural_x = (extraction.get("contradiction") or {}).get("present")
    if neural_x is None:
        neural_x = extraction.get("contradiction_present")
    grounded_x = grounding.get("contradiction")
    agree = neural_x == grounded_x if neural_x is not None and grounded_x is not None else None
    reasons: list[str] = []
    if agree is False:
        if neural_x is True and grounded_x is False:
            reasons.append(
                "Extractor claimed contradiction, but grounding treated note as "
                "sequenced temporal-change (or cleared soft X) — classic N2S fidelity risk."
            )
        elif neural_x is False and grounded_x is True:
            reasons.append(
                "Extractor denied contradiction, but grounding set X=true from hard conflict cues."
            )
        else:
            reasons.append("Neural contradiction flag disagrees with grounded X.")
    return {
        "neural_contradiction": neural_x,
        "grounded_X": grounded_x,
        "agree": agree,
        "uncertain": agree is None,
        "reasons": reasons,
        "kind": "extract_vs_ground_X",
        "grounding_trace": list(grounding.get("trace") or [])[-6:],
    }


def evaluate_fast(body: dict[str, Any]) -> dict[str, Any]:
    note = (body.get("note") or "").strip()
    if len(note) < 20:
        raise ValueError("Paste a clinical note (at least ~20 characters).")
    model = body.get("model") or DEFAULT_MODEL
    gold = (body.get("gold") or "").strip() or "UNKNOWN"
    case_id = (body.get("case_id") or f"LIVE_{uuid.uuid4().hex[:8]}").strip()
    unload_after = body.get("unload_after", True)
    if isinstance(unload_after, str):
        unload_after = unload_after.strip().lower() not in ("0", "false", "no")
    live = model != "mock" and ollama_reachable()
    if model != "mock" and not live:
        raise RuntimeError("Ollama not reachable at 127.0.0.1:11434 — pick Mock or start Ollama.")

    try:
        # Fast path: note → C (direct extract) + short analysis → D → Dual + evidence
        analysis = note if not live else analyze_free_text(note, model=model)
        c_block = _verified_pipeline(
            source_text=note,
            analysis_for_verify=analysis,
            model=model,
            case_id=case_id,
            gold=gold if gold != "UNKNOWN" else "",
            live=live,
            from_analysis=False,
        )
        d_block = _verified_pipeline(
            source_text=note,
            analysis_for_verify=analysis,
            model=model,
            case_id=case_id,
            gold=gold if gold != "UNKNOWN" else "",
            live=live,
            from_analysis=True,
        )

        c_block = {
            **c_block,
            **gate_verdict_with_evidence(
                verdict=c_block["verdict"],
                extraction=Extraction.from_dict(c_block["extraction"]),
                grounding=c_block.get("grounding") or {},
                source_text=note,
                source_id="note",
            ),
        }
        d_block = {
            **d_block,
            **gate_verdict_with_evidence(
                verdict=d_block["verdict"],
                extraction=Extraction.from_dict(d_block["extraction"]),
                grounding=d_block.get("grounding") or {},
                source_text=analysis,
                source_id="analysis",
            ),
        }

        dual = dual_path_verdict(c_block["verdict"], d_block["verdict"])
        mismatch = _mismatch(c_block["extraction"], c_block.get("grounding") or {})

        verify_ok = (c_block.get("disposition") != "REVIEW") or (
            c_block.get("verdict") != Verdict.REVIEW.value
            and bool((c_block.get("verify") or {}).get("first", {}).get("ok", True))
        )
        # Prefer explicit verify on C
        vfirst = (c_block.get("verify") or {}).get("first") or {}
        if "ok" in vfirst:
            verify_ok = bool(vfirst.get("ok"))
        if (c_block.get("verify") or {}).get("second"):
            verify_ok = bool(c_block["verify"]["second"].get("ok"))

        dual_agreed = c_block["verdict"] == d_block["verdict"] and dual["verdict"] != Verdict.REVIEW.value
        if dual["verdict"] == Verdict.REVIEW.value:
            dual_agreed = False

        gates = apply_gates(
            verify_ok=verify_ok if c_block.get("disposition") != "REVIEW" else False,
            dual_agreed=dual_agreed if dual["verdict"] != Verdict.REVIEW.value else False,
            contradiction_supported=None,
            proposed_verdict=dual["verdict"],
        )
        # Final disposition: Dual REVIEW or gate fail or mismatch → REVIEW
        reasons = list(gates.get("reasons") or [])
        disposition = "AUTO"
        verdict = dual["verdict"]
        if dual["verdict"] == Verdict.REVIEW.value or dual.get("disposition") == "REVIEW":
            disposition = "REVIEW"
            reasons.append("dual_or_path_REVIEW")
        if not gates.get("ok", True):
            disposition = "REVIEW"
            verdict = Verdict.REVIEW.value
        if mismatch.get("agree") is False:
            disposition = "REVIEW"
            if verdict != Verdict.REVIEW.value:
                # Keep symbolic dual verdict visible but force REVIEW disposition
                pass
            reasons.append("n2s_mismatch_extract_vs_ground_X")
            verdict = Verdict.REVIEW.value

        contract = classify_case(
            gold=gold if gold != "UNKNOWN" else verdict,
            verdict=verdict,
            disposition=disposition,
        )
        if gold == "UNKNOWN":
            contract["match_gold"] = None
            contract["wrong_AUTO"] = False
            contract["correct_AUTO"] = disposition == "AUTO"
            contract["bucket"] = "REVIEW" if disposition == "REVIEW" else "AUTO_no_gold"

        result = {
            "schema": "n2s_eval_fast_v1",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "case_id": case_id,
            "model": model,
            "live": live,
            "note": note,
            "gold": gold,
            "analysis": analysis if live else "(mock — analysis = note)",
            "paths": {"C_full": c_block, "D_full": d_block, "Dual_full": dual},
            "mismatch": mismatch,
            "gates": {**gates, "reasons": reasons},
            "verdict": verdict,
            "disposition": disposition,
            "contract": contract,
            "deep_dive_available": True,
            "summary": {
                "headline": (
                    f"{disposition}: {verdict}"
                    + (" · N2S mismatch" if mismatch.get("agree") is False else "")
                ),
                "c_verdict": c_block["verdict"],
                "d_verdict": d_block["verdict"],
                "dual_verdict": dual["verdict"],
            },
        }

        RUNS_DIR.mkdir(parents=True, exist_ok=True)
        path = RUNS_DIR / f"fast-{case_id}-{uuid.uuid4().hex[:6]}.json"
        path.write_text(json.dumps(result, indent=2), encoding="utf-8")
        result["artifact"] = str(path)
        return result
    finally:
        if live and unload_after and model != "mock":
            freed = unload_model(model)
            # Attach only if result locals exist — may be mid-exception
            try:
                result["ollama_unload"] = freed  # type: ignore[name-defined]
            except Exception:  # noqa: BLE001
                pass
            print(
                f"[n2s-eval] ollama unload {model}: ok={freed.get('ok')} "
                f"verified={freed.get('verified')} {freed.get('elapsed_s')}s",
                flush=True,
            )

def _deep_dive_work(fast: dict[str, Any]) -> dict[str, Any]:
    """Expand a fast result: round-trip checks + richer narrative (no HF steer)."""
    _job["step"] = "round-trip C"
    note = fast["note"]
    model = fast["model"]
    live = fast.get("live", False)
    case_id = fast["case_id"]
    gold = fast.get("gold") or ""
    c = fast["paths"]["C_full"]
    d = fast["paths"]["D_full"]
    analysis = fast.get("analysis") or note

    rt_c = None
    rt_d = None
    if live and model != "mock":
        from n2s_lab.neural import paraphrase_from_extraction

        try:
            para_c = paraphrase_from_extraction(
                Extraction.from_dict(c["extraction"]), model=model
            )
        except Exception:
            para_c = json.dumps(c.get("grounding") or {}, indent=2)
        rt_c = verify_roundtrip(
            source_text=note,
            paraphrase=para_c if isinstance(para_c, str) else str(para_c),
            extraction=Extraction.from_dict(c["extraction"]),
            case_id=case_id,
            gold=gold if gold != "UNKNOWN" else "",
        ).to_dict()
        _job["step"] = "round-trip D"
        try:
            para_d = paraphrase_from_extraction(
                Extraction.from_dict(d["extraction"]), model=model
            )
        except Exception:
            para_d = json.dumps(d.get("grounding") or {}, indent=2)
        rt_d = verify_roundtrip(
            source_text=analysis,
            paraphrase=para_d if isinstance(para_d, str) else str(para_d),
            extraction=Extraction.from_dict(d["extraction"]),
            case_id=case_id,
            gold=gold if gold != "UNKNOWN" else "",
        ).to_dict()
    else:
        rt_c = {"ok": None, "note": "skipped (mock / offline)"}
        rt_d = {"ok": None, "note": "skipped (mock / offline)"}

    # Free Ollama VRAM before HF forensics (7B needs ~14 GiB free).
    # unload_model now polls /api/ps until the runner is actually gone.
    if live and model != "mock":
        _job["step"] = "unload ollama"
        freed = unload_model(model)
        _log(
            f"pre-forensics ollama unload {model}: ok={freed.get('ok')} "
            f"verified={freed.get('verified')} elapsed={freed.get('elapsed_s')}s "
            f"via={freed.get('method_that_cleared')}"
        )
        if not freed.get("verified"):
            _log(f"pre-forensics unload NOT verified: {freed.get('error') or freed}")
            # One more hard stop via CLI path inside unload_model already tried;
            # brief extra wait then re-check for HF gate.
            import time

            time.sleep(2.0)
            freed2 = unload_model(model, verify_timeout_s=20)
            _log(
                f"pre-forensics unload retry: ok={freed2.get('ok')} "
                f"verified={freed2.get('verified')}"
            )

    # Deep dive: HF forensics (layer scores) when GPU free; else defer + example
    _job["step"] = "n2s forensics"
    forensics: dict[str, Any] | None = None
    forensics_example: dict[str, Any] | None = None
    try:
        from n2s_lab.n2s_forensics import (
            format_forensics_text,
            forensics_from_frozen_expand_case,
            run_forensics_live,
        )

        # Prefer D_full verdict as "commitment" when Dual is REVIEW (paths disagree)
        d_verdict = (d.get("verdict") or "").upper()
        c_verdict = (c.get("verdict") or "").upper()
        dual_v = (fast.get("paths") or {}).get("Dual_full", {}).get("verdict") or ""
        if dual_v == "REVIEW" and d_verdict == "CONTRADICTION":
            commitment = "CONTRADICTION"
        elif dual_v == "REVIEW" and c_verdict and c_verdict != "REVIEW":
            commitment = c_verdict
        else:
            commitment = dual_v if dual_v != "REVIEW" else d_verdict or c_verdict

        forensics = run_forensics_live(
            note,
            case_id=case_id,
            gold=gold if gold != "UNKNOWN" else "",
            final_commitment=commitment or None,
            unload_after=True,
        )
        if forensics.get("status") == "deferred_gpu_busy":
            forensics_example = forensics_from_frozen_expand_case(
                "TX_E01", final_commitment="CONTRADICTION"
            )
            forensics_example["ascii"] = format_forensics_text(forensics_example)
            forensics["ascii"] = None
        elif forensics.get("ok"):
            forensics["ascii"] = format_forensics_text(forensics)
        else:
            forensics_example = forensics_from_frozen_expand_case(
                "TX_E01", final_commitment="CONTRADICTION"
            )
            forensics_example["ascii"] = format_forensics_text(forensics_example)
    except Exception as exc:  # noqa: BLE001
        forensics = {"ok": False, "status": "error", "message": str(exc)}
        try:
            from n2s_lab.n2s_forensics import (
                format_forensics_text,
                forensics_from_frozen_expand_case,
            )

            forensics_example = forensics_from_frozen_expand_case(
                "TX_E01", final_commitment="CONTRADICTION"
            )
            forensics_example["ascii"] = format_forensics_text(forensics_example)
        except Exception:  # noqa: BLE001
            forensics_example = None

    _job["step"] = "assemble report"
    mismatch = fast.get("mismatch") or {}
    narrative = [
        f"Case {case_id} · model {model} · disposition {fast.get('disposition')} · verdict {fast.get('verdict')}.",
        f"C_full={c.get('verdict')} · D_full={d.get('verdict')} · Dual={fast['paths']['Dual_full'].get('verdict')}.",
    ]
    if mismatch.get("agree") is False:
        narrative.append("N2S mismatch: " + "; ".join(mismatch.get("reasons") or []))
        narrative.append(
            "Track A action: keep REVIEW (do not AUTO). Grounding may already have corrected soft X."
        )
    else:
        narrative.append("Extract contradiction flag and grounded X agree (or one side unknown).")
    if forensics and forensics.get("status") == "deferred_gpu_busy":
        narrative.append(
            "HF forensics deferred (GPU busy): " + str(forensics.get("message") or "")
        )
        narrative.append(
            "Showing EXAMPLE panel from frozen expand TX_E01 (not this live note)."
        )
    elif forensics and forensics.get("ok"):
        commit = str(forensics.get("final_commitment") or "").upper()
        mm = forensics.get("representation_commitment_mismatch")
        if commit == "CONTRADICTION":
            cmp = f"mismatch={mm}"
        else:
            cmp = (
                "comparison=NOT_EVALUATED "
                "(commitment outside temporal-change/contradiction probe label space)"
            )
        narrative.append(
            f"HF forensics: L20 readout={forensics.get('upstream_signal')} · "
            f"commitment={forensics.get('final_commitment')} · {cmp}"
        )
    else:
        narrative.append(
            "HF forensics unavailable this run — example panel may be attached."
        )

    report = {
        "schema": "n2s_eval_deep_v1",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "parent_fast": fast.get("artifact"),
        "case_id": case_id,
        "model": model,
        "roundtrip": {"C": rt_c, "D": rt_d},
        "mismatch": mismatch,
        "gates": fast.get("gates"),
        "paths": fast.get("paths"),
        "narrative": narrative,
        "forensics": forensics,
        "forensics_example": forensics_example,
        "mi_status": {
            "ran": bool(forensics and forensics.get("ok")),
            "reason": (
                "Live HF layer probe (temporal-change vs contradiction) when GPU free; "
                "causal patch/steer still on :8256 / Track B."
            ),
            "next": "Free ~15 GiB VRAM (finish/kill Ollama 32B job) for live forensics on this note.",
        },
    }
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    path = RUNS_DIR / f"deep-{case_id}-{uuid.uuid4().hex[:6]}.json"
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    report["artifact"] = str(path)
    if live and model != "mock":
        freed = unload_model(model)
        report["ollama_unload"] = freed
        _log(f"ollama unload {model}: {freed.get('ok')}")
    return report


def job_status() -> dict[str, Any]:
    with _lock:
        return {
            "id": _job["id"],
            "kind": _job["kind"],
            "state": _job["state"],
            "step": _job["step"],
            "result": _job["result"],
            "error": _job["error"],
            "log": list(_job.get("log") or [])[-20:],
            "busy": _job["state"] in ("queued", "running"),
        }


def submit_deep_dive(body: dict[str, Any]) -> dict[str, Any]:
    fast = body.get("fast_result")
    if not isinstance(fast, dict):
        raise ValueError("fast_result required (run Evaluate first)")
    with _lock:
        if _job["state"] in ("queued", "running"):
            return {"ok": False, "error": "deep dive already running", "job": job_status()}
        jid = uuid.uuid4().hex[:8]
        _job.update(
            {
                "id": jid,
                "kind": "deep_dive",
                "state": "queued",
                "step": "queued",
                "result": None,
                "error": None,
                "log": [],
            }
        )

    def worker() -> None:
        with _lock:
            _job["state"] = "running"
            _job["step"] = "start"
        _log(f"deep dive {jid} start")
        try:
            report = _deep_dive_work(fast)
            with _lock:
                _job["result"] = report
                _job["state"] = "done"
                _job["step"] = "done"
            _log(f"deep dive {jid} done → {report.get('artifact')}")
        except Exception as exc:  # noqa: BLE001
            with _lock:
                _job["state"] = "error"
                _job["error"] = str(exc)
                _job["log"] = (_job.get("log") or []) + [traceback.format_exc()[-600:]]
            _log(f"deep dive ERROR: {exc}")

    threading.Thread(target=worker, daemon=True).start()
    return {"ok": True, "job_id": jid, "job": job_status()}

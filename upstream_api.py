"""N2S Upstream workbench API — port 8258 (live score + frozen replay)."""

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

from n2s_lab.n2s_upstream_live import (  # noqa: E402
    DIRECTIONS_PATH,
    score_note_live,
    run_custom_pair_map,
)
from n2s_lab.n2s_upstream_prefill import DEFAULT_NOTES  # noqa: E402
from n2s_lab.paths import ARTIFACTS_DIR  # noqa: E402

PORT = 8258
STATIC_DIR = ROOT / "static"

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
    with _lock:
        _job["log"] = (_job.get("log") or [])[-40:] + [msg]
        _job["step"] = msg[:120]


def _load_json_if(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return None


def meta_payload() -> dict[str, Any]:
    return {
        "ok": True,
        "port": PORT,
        "lab_root": str(LAB_ROOT),
        "artifacts_dir": str(ARTIFACTS_DIR),
        "eval_workbench": "http://127.0.0.1:8257/",
        "directions_cached": DIRECTIONS_PATH.is_file(),
        "examples": [
            {
                "id": n["id"],
                "label": n["id"],
                "gold": n.get("gold"),
                "evidence": n["evidence"],
                "role": "temporal" if n["id"] == "EX_TEMPORAL_FOLFOX" else "contra",
            }
            for n in DEFAULT_NOTES.values()
        ],
        "capabilities": {
            "frozen_replay": True,
            "live_score": True,
            "live_custom_pair": True,
            "note": (
                "Paste a clinical note → Score vs frozen U-A d (formation direction). "
                "Optional: paste temporal + contra → live custom-pair map. "
                "Not ablation / SAE / α·v. Needs ~14 GiB free GPU."
            ),
        },
        "claim_hygiene": {
            "say": "live score vs frozen U-A d; optional custom-pair map",
            "do_not_say": "ablation UI; upstream editor; freeze-v is temporality",
        },
    }


def _load_json_if(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return None


def frozen_replay_payload() -> dict[str, Any]:
    """Read-only U-A/U-B/U-B2 artifact panel (this port only — not on :8257)."""
    ua_map = _load_json_if(ARTIFACTS_DIR / "n2s-upstream-ua-map.json")
    ua_read = _load_json_if(ARTIFACTS_DIR / "n2s-upstream-ua-readout.json")
    ub = _load_json_if(ARTIFACTS_DIR / "n2s-upstream-ub-patch.json")
    ub_read = _load_json_if(ARTIFACTS_DIR / "n2s-upstream-ub-readout.json")
    ub2 = _load_json_if(ARTIFACTS_DIR / "n2s-upstream-ub2-patch.json")
    ub2_read = _load_json_if(ARTIFACTS_DIR / "n2s-upstream-ub2-readout.json")

    layers_slim: list[dict[str, Any]] = []
    if ua_map:
        for key, block in (ua_map.get("by_layer") or {}).items():
            notes_slim: dict[str, Any] = {}
            for nid, nb in (block.get("notes") or {}).items():
                notes_slim[nid] = {
                    "mean_score_d": nb.get("mean_score_d"),
                    "top_sites": (nb.get("top_sites") or [])[:8],
                }
            layers_slim.append(
                {
                    "layer_key": key,
                    "layer": block.get("layer"),
                    "delta_mean_norm": block.get("delta_mean_norm"),
                    "cos_d_vs_freeze_v": block.get("cos_d_vs_freeze_v"),
                    "notes": notes_slim,
                }
            )

    def _patch_slim(panel: dict[str, Any] | None) -> dict[str, Any] | None:
        if not panel:
            return None
        notes: dict[str, Any] = {}
        for nid, nb in (panel.get("notes") or {}).items():
            notes[nid] = {
                "layer": nb.get("layer"),
                "mode": nb.get("mode") or panel.get("mode"),
                "tokens": nb.get("tokens"),
                "positions": nb.get("positions"),
                "arms": [
                    {
                        "component": a.get("component"),
                        "extract_x": a.get("extract_x"),
                        "margin": a.get("margin"),
                        "verdict": a.get("verdict"),
                    }
                    for a in (nb.get("arms") or [])
                ],
                "deltas_vs_baseline": nb.get("deltas_vs_baseline"),
            }
        return {
            "kind": panel.get("kind"),
            "variant": panel.get("variant"),
            "layer": panel.get("layer"),
            "mode": panel.get("mode"),
            "timestamp": panel.get("timestamp"),
            "quest": panel.get("quest"),
            "notes": notes,
        }

    present = {
        "ua_map": ua_map is not None,
        "ua_readout": ua_read is not None,
        "ub_patch": ub is not None,
        "ub_readout": ub_read is not None,
        "ub2_patch": ub2 is not None,
        "ub2_readout": ub2_read is not None,
    }
    frozen_notes = [
        {
            "id": n["id"],
            "label": n["id"],
            "gold": n.get("gold"),
            "evidence": n["evidence"],
            "role": "temporal" if n["id"] == "EX_TEMPORAL_FOLFOX" else "contra",
        }
        for n in DEFAULT_NOTES.values()
    ]
    return {
        "ok": True,
        "kind": "n2s_upstream_viewer_v1",
        "read_only": True,
        "lab_root": str(LAB_ROOT),
        "artifacts_dir": str(ARTIFACTS_DIR),
        "present": present,
        "program_doc": "docs/TRACKB-UPSTREAM-PROGRAM.md",
        "uses_clinical_note_box": False,
        "frozen_notes": frozen_notes,
        "ua": {
            "best_layer": (ua_read or {}).get("best_layer"),
            "findings": (ua_read or {}).get("findings") or [],
            "top_sites_summary": (ua_read or {}).get("top_sites_summary") or [],
            "formation_hypothesis": (ua_read or {}).get("formation_hypothesis"),
            "pair": (ua_map or {}).get("pair"),
            "model_id": (ua_map or {}).get("model_id"),
            "layers": layers_slim,
            "timestamp": (ua_map or {}).get("timestamp") or (ua_read or {}).get("timestamp"),
        },
        "ub": {
            "patch": _patch_slim(ub),
            "findings": (ub_read or {}).get("findings") or [],
            "formation_hypothesis": (ub_read or {}).get("formation_hypothesis"),
            "shortlist": (ub_read or {}).get("shortlist") or [],
        },
        "ub2": {
            "patch": _patch_slim(ub2),
            "findings": (ub2_read or {}).get("findings") or [],
            "formation_hypothesis": (ub2_read or {}).get("formation_hypothesis"),
            "shortlist": (ub2_read or {}).get("shortlist") or [],
        },
        "claim_hygiene": {
            "say": "U-A formation map + U-B/U-B2 ablation nulls (viewer)",
            "do_not_say": "live upstream editor; circuit found; freeze-v is temporality",
        },
    }


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


def submit_score(body: dict[str, Any]) -> dict[str, Any]:
    note = (body.get("note") or body.get("evidence") or "").strip()
    if not note:
        raise ValueError("note required")
    case_id = (body.get("case_id") or f"LIVE_{uuid.uuid4().hex[:8]}").strip()
    gold = (body.get("gold") or "UNKNOWN").strip() or "UNKNOWN"
    rebuild = bool(body.get("rebuild_directions"))

    with _lock:
        if _job["state"] in ("queued", "running"):
            return {"ok": False, "error": "job already running", "job": job_status()}
        jid = uuid.uuid4().hex[:8]
        _job.update(
            {
                "id": jid,
                "kind": "live_score",
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
            _job["step"] = "score"
        _log(f"live score {jid} · {case_id}")
        try:
            report = score_note_live(
                note,
                case_id=case_id,
                gold=gold,
                rebuild_directions=rebuild,
            )
            with _lock:
                _job["result"] = report
                _job["state"] = "done" if report.get("ok") else "error"
                _job["step"] = "done"
                if not report.get("ok"):
                    _job["error"] = report.get("message") or report.get("error") or "score failed"
            _log(f"live score {jid} → {report.get('artifact') or report.get('status')}")
        except Exception as exc:  # noqa: BLE001
            with _lock:
                _job["state"] = "error"
                _job["error"] = str(exc)
                _job["log"] = (_job.get("log") or []) + [traceback.format_exc()[-600:]]
            _log(f"live score ERROR: {exc}")

    threading.Thread(target=worker, daemon=True).start()
    return {"ok": True, "job_id": jid, "job": job_status()}


def submit_pair(body: dict[str, Any]) -> dict[str, Any]:
    temporal = (body.get("temporal") or body.get("note") or "").strip()
    contra = (body.get("contra") or "").strip()
    if not temporal or not contra:
        raise ValueError("temporal and contra notes required")

    with _lock:
        if _job["state"] in ("queued", "running"):
            return {"ok": False, "error": "job already running", "job": job_status()}
        jid = uuid.uuid4().hex[:8]
        _job.update(
            {
                "id": jid,
                "kind": "live_pair",
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
            _job["step"] = "custom pair"
        _log(f"live pair {jid}")
        try:
            report = run_custom_pair_map(temporal, contra)
            with _lock:
                _job["result"] = report
                _job["state"] = "done" if report.get("ok") else "error"
                _job["step"] = "done"
                if not report.get("ok"):
                    _job["error"] = report.get("message") or report.get("error") or "pair failed"
            _log(f"live pair {jid} → {report.get('artifact') or report.get('status')}")
        except Exception as exc:  # noqa: BLE001
            with _lock:
                _job["state"] = "error"
                _job["error"] = str(exc)
                _job["log"] = (_job.get("log") or []) + [traceback.format_exc()[-600:]]
            _log(f"live pair ERROR: {exc}")

    threading.Thread(target=worker, daemon=True).start()
    return {"ok": True, "job_id": jid, "job": job_status()}

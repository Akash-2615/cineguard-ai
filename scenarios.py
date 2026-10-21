"""Headless scenario runner and the printable evaluation report.

Clips are read through ``FileSource`` with pacing disabled, then each frame is
passed through ``CommandCenterRuntime._consume_frame``. That is the live path.
Timestamps stay ``frame_index / fps``. Nothing in the metrics uses the wall clock.

SIMULATED NIR output is not a real-world accuracy measurement. Synthetic glints
are injected only when a manifest entry asks for them.
"""

from __future__ import annotations

import base64
import html
import json
import tempfile
import threading
from pathlib import Path
from typing import Any, Callable

from config import AppConfig
from sources import FileSource
from utils.logger import get_logger

logger = get_logger(__name__)

MAX_SCENARIO_FRAMES = 5000
DISCLAIMER = (
    "SIMULATED NIR. This proof of concept has no near-infrared camera. "
    "Scores, glints, and this report are not real-world accuracy."
)
LIMITATIONS = [
    "There is no NIR camera. The mono image is a red-weighted simulation, and a manifest entry can inject synthetic glints. Those results must not be quoted as measured detection accuracy.",
    "Without a local pose weights file the runner stays degraded and every clip is a no-alert, because no person is detected.",
    "Eating, stretching, texting, and recording labels are geometric rules, not a trained behavior classifier.",
    "False alarms per hour are extrapolated from the durations of the non-alert clips only. Short clips make that rate unstable. It is not an observed hourly rate in a theatre.",
    "Seat labels stay marked uncalibrated until four corners are saved. A wrong quad maps feet to the wrong seat.",
    "Active time uses frame timestamps. A pause does not add time, and a seek is clamped so it cannot add minutes of recording.",
    "Each clip is capped at 5000 frames so a long file cannot run away.",
    "One 2D camera cannot see a phone that faces the screen and is hidden from this viewpoint.",
]
FUTURE_WORK = [
    "An 850 nm or 940 nm illuminator, an IR-pass filter, and a global-shutter camera in place of the simulated channel.",
    "A second camera or a screen-facing view so a device aimed at the screen is visible.",
    "Reviewer feedback that retunes the threshold on confirmed and false-positive incidents.",
    "Authentication, packaging, and a written operator guide.",
]


def load_manifest(path: str | Path) -> list[dict[str, Any]]:
    file = Path(path)
    payload = json.loads(file.read_text(encoding="utf-8"))
    clips = payload.get("clips", payload if isinstance(payload, list) else None)
    if not isinstance(clips, list) or not clips:
        raise ValueError("manifest must contain a non-empty clips list")
    parsed = []
    for item in clips:
        if not isinstance(item, dict):
            raise ValueError("each manifest entry must be an object")
        expected = str(item.get("expected", "")).strip().lower()
        if expected not in {"alert", "no_alert"}:
            raise ValueError(f"clip {item.get('id')} expected must be alert or no_alert")
        if not item.get("id") or not item.get("file"):
            raise ValueError("each clip needs an id and a file")
        parsed.append({
            "id": str(item["id"]),
            "file": str(item["file"]),
            "label": str(item.get("label") or item["id"]),
            "expected": expected,
            "expected_seat": item.get("expected_seat") or None,
            "description": str(item.get("description") or ""),
            "synthetic_glint": bool(item.get("synthetic_glint", False)),
            "notes": str(item.get("notes") or ""),
        })
    return parsed


def resolve_clip(manifest_path: Path, file_name: str) -> Path:
    from app import ROOT
    candidate = Path(file_name)
    if candidate.is_absolute():
        return candidate
    beside = (manifest_path.parent / candidate).resolve()
    if beside.is_file():
        return beside
    return (ROOT / candidate).resolve()


def run_manifest(
    manifest_path: str | Path,
    config: AppConfig | None = None,
    models: Any | None = None,
    on_progress: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Process every clip in isolation and return a deterministic result document."""
    manifest = Path(manifest_path).resolve()
    clips = load_manifest(manifest)
    base = config or AppConfig(_env_file=None)
    rows = []
    for index, clip in enumerate(clips):
        if on_progress:
            on_progress({"completed": index, "total": len(clips), "current": clip["id"]})
        rows.append(_run_clip(manifest, clip, base, models))
    if on_progress:
        on_progress({"completed": len(clips), "total": len(clips), "current": ""})
    recorded = [row for row in rows if not row["synthetic_glint"]]
    synthetic = [row for row in rows if row["synthetic_glint"]]
    return {
        "manifest": str(manifest),
        "disclaimer": DISCLAIMER,
        "clips": rows,
        "metrics": _metrics(rows),
        "metrics_recorded": _metrics(recorded),
        "metrics_synthetic": _metrics(synthetic),
        "config": base.public_dict(),
        "limitations": list(LIMITATIONS),
        "future_work": list(FUTURE_WORK),
    }


def render_report(results: dict[str, Any]) -> str:
    """Self-contained HTML. Dynamic text is escaped. Printable from the browser."""
    clips = results.get("clips") or []
    rows = "\n".join(_clip_row(clip) for clip in clips)
    if not rows:
        rows = "<tr><td colspan='8'>No scenario run yet.</td></tr>"
    limitations = "".join(f"<li>{html.escape(item)}</li>" for item in results.get("limitations") or LIMITATIONS)
    future = "".join(f"<li>{html.escape(item)}</li>" for item in results.get("future_work") or FUTURE_WORK)
    config = html.escape(json.dumps(results.get("config") or {}, indent=2, sort_keys=True))
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>CineGuard evaluation report</title>
<style>
body {{ font-family: Georgia, serif; color: #1c1917; margin: 2rem; }}
h1 {{ font-size: 1.6rem; }}
.banner {{ background: #fef3c7; border: 1px solid #d97706; padding: 0.8rem 1rem; }}
table {{ border-collapse: collapse; width: 100%; margin: 1rem 0; }}
th, td {{ border: 1px solid #d6d3d1; padding: 0.4rem 0.5rem; vertical-align: top; font-size: 0.9rem; }}
th {{ background: #f5f5f4; text-align: left; }}
.pass {{ color: #166534; }} .fail {{ color: #991b1b; }}
img {{ width: 160px; height: auto; }}
pre {{ white-space: pre-wrap; background: #fafaf9; padding: 0.8rem; }}
@media print {{ .noprint {{ display: none; }} body {{ margin: 0.5cm; }} }}
</style>
</head>
<body>
<button class="noprint" onclick="window.print()">Print or save as PDF</button>
<h1>CineGuard evaluation report</h1>
<p class="banner">{html.escape(results.get("disclaimer") or DISCLAIMER)}</p>
<p>Manifest: {html.escape(str(results.get("manifest") or "not run"))}</p>
<h2>Metrics</h2>
{_metric_block("All clips", results.get("metrics") or {})}
{_metric_block("Recorded clips", results.get("metrics_recorded") or {})}
{_metric_block("Synthetic-glint clips", results.get("metrics_synthetic") or {})}
<h2>Clips</h2>
<table>
<thead><tr><th>Clip</th><th>Expected</th><th>Outcome</th><th>Pass</th><th>Time to alert</th><th>Max score</th><th>Seat</th><th>Keyframe</th></tr></thead>
<tbody>
{rows}
</tbody>
</table>
<h2>Config snapshot</h2>
<pre>{config}</pre>
<h2>Limitations</h2>
<ul>{limitations}</ul>
<h2>Future work</h2>
<ul>{future}</ul>
</body>
</html>
"""


class ScenarioService:
    """One background job. The live loop is held so it does not share the pose model."""

    def __init__(self, runtime: Any) -> None:
        self.runtime = runtime
        self._lock = threading.Lock()
        self.status = "idle"
        self.error = ""
        self.progress = {"completed": 0, "total": 0, "current": ""}
        self.results: dict[str, Any] | None = None
        self._thread: threading.Thread | None = None

    def start(self, manifest_path: Path) -> dict[str, Any]:
        with self._lock:
            if self.status == "running":
                raise RuntimeError("a scenario run is already in progress")
            self.status = "running"
            self.error = ""
            self.progress = {"completed": 0, "total": 0, "current": ""}
            self._thread = threading.Thread(target=self._job, args=(manifest_path,), name="cineguard-scenarios", daemon=True)
            self._thread.start()
        return self.snapshot()

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                "status": self.status,
                "error": self.error,
                "progress": dict(self.progress),
                "results": self.results,
            }

    def _job(self, manifest_path: Path) -> None:
        self.runtime.hold_processing(True)
        try:
            def progress(update: dict[str, Any]) -> None:
                with self._lock:
                    self.progress = dict(update)

            payload = run_manifest(manifest_path, self.runtime.config, self.runtime.models, progress)
            with self._lock:
                self.results = payload
                self.status = "done"
        except Exception as exc:
            logger.exception("scenario run failed")
            with self._lock:
                self.status = "error"
                self.error = f"{type(exc).__name__}: {exc}"
        finally:
            self.runtime.hold_processing(False)


def _run_clip(
    manifest: Path,
    clip: dict[str, Any],
    config: AppConfig,
    models: Any,
) -> dict[str, Any]:
    from app import CommandCenterRuntime
    path = resolve_clip(manifest, clip["file"])
    base = {
        "id": clip["id"],
        "file": clip["file"],
        "label": clip["label"],
        "expected": clip["expected"],
        "expected_seat": clip["expected_seat"],
        "description": clip["description"],
        "synthetic_glint": clip["synthetic_glint"],
        "notes": clip["notes"],
        "outcome": "no_alert",
        "time_to_alert": None,
        "max_score": 0.0,
        "incidents": 0,
        "seat": None,
        "synthetic": False,
        "duration_s": 0.0,
        "passed": False,
        "error": None,
        "keyframe": None,
    }
    if not path.is_file():
        base["error"] = f"file not found: {path.name}"
        return base
    bundle = _isolated_models(models)
    with tempfile.TemporaryDirectory(prefix="cineguard-scenario-") as tmp:
        root = Path(tmp)
        runtime = CommandCenterRuntime(
            config.model_copy(update={
                "db_path": str(root / "cineguard.db"),
                "evidence_dir": str(root / "evidence"),
                "loop_video": False,
                "camera_source": str(path),
            }),
            source=FileSource(path, loop=False, pace=False),
            models=bundle,
            autoload_models=False,
        )
        runtime.prepare()
        if clip["synthetic_glint"]:
            runtime.nir.update_params({"synthetic_glint": True})
        if runtime.models and runtime.models.pose:
            runtime.models.pose.clear_stability()
        try:
            base["duration_s"] = _drain(runtime)
        except Exception as exc:
            logger.exception("scenario clip failed: %s", clip["id"])
            base["error"] = f"{type(exc).__name__}: {exc}"
        finally:
            runtime.evidence.finalize_all()
            runtime.stop()
        if base["error"]:
            return base
        listed = runtime.incidents.list_incidents(limit=200)
        items = listed["items"]
        base["incidents"] = int(listed["total"])
        if items:
            base["outcome"] = "alert"
            base["time_to_alert"] = round(min(float(item["started_ts"]) for item in items), 4)
            base["max_score"] = round(max(float(item["max_score"]) for item in items), 4)
            base["seat"] = items[0]["seat"]
            base["synthetic"] = any(bool(item["synthetic"]) for item in items)
            key_path = items[0].get("keyframe_path")
            if key_path and Path(key_path).is_file():
                base["keyframe"] = base64.b64encode(Path(key_path).read_bytes()).decode("ascii")
        else:
            base["max_score"] = 0.0
    seat_ok = clip["expected_seat"] is None or (base["seat"] or "") == clip["expected_seat"]
    base["passed"] = base["outcome"] == clip["expected"] and seat_ok
    base["duration_s"] = round(float(base["duration_s"]), 4)
    return base


def _drain(runtime: Any) -> float:
    source = runtime.source
    source.start()
    last_ts = 0.0
    frames = 0
    while frames < MAX_SCENARIO_FRAMES:
        frame = source.read()
        if frame is None:
            state = source.info().get("state")
            if state in {"ended", "error"}:
                break
            if not source.is_alive() and state != "open":
                break
            if runtime._stop.wait(0.002):
                break
            continue
        frames += 1
        last_ts = float(frame.ts)
        runtime._consume_frame(frame, runtime.config)
    return last_ts


def _isolated_models(models: Any) -> Any:
    from app import ModelBundle
    from engine.tracker import SpatialProximityTracker
    if models is not None and models.pose is not None and models.loaded:
        return ModelBundle(
            pose=models.pose,
            fusion=None,
            tracker=SpatialProximityTracker(max_missed_frames=30),
            loaded=True,
            error=None,
        )
    message = models.error if models is not None and models.error else "pose model not loaded"
    return ModelBundle.unavailable(message)


def _metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(rows)
    if total == 0:
        return {
            "clips": 0, "accuracy": None, "precision": None, "recall": None,
            "mean_time_to_alert": None, "false_alarms_per_hour": None,
            "caveat": LIMITATIONS[3],
        }
    passed = sum(1 for row in rows if row["passed"])
    usable = [row for row in rows if not row["error"]]
    tp = sum(1 for row in usable if row["expected"] == "alert" and row["outcome"] == "alert")
    fp = sum(1 for row in usable if row["expected"] == "no_alert" and row["outcome"] == "alert")
    fn = sum(1 for row in usable if row["expected"] == "alert" and row["outcome"] == "no_alert")
    predicted = tp + fp
    positives = tp + fn
    times = [float(row["time_to_alert"]) for row in usable if row["outcome"] == "alert" and row["time_to_alert"] is not None]
    negative_s = sum(float(row["duration_s"]) for row in usable if row["expected"] == "no_alert")
    rate = None
    if negative_s > 0:
        rate = round(fp / (negative_s / 3600.0), 4)
    return {
        "clips": total,
        "passed": passed,
        "accuracy": round(passed / total, 4),
        "precision": round(tp / predicted, 4) if predicted else None,
        "recall": round(tp / positives, 4) if positives else None,
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "mean_time_to_alert": round(sum(times) / len(times), 4) if times else None,
        "false_alarms_per_hour": rate,
        "negative_duration_s": round(negative_s, 4),
        "caveat": LIMITATIONS[3],
    }


def _metric_block(title: str, metrics: dict[str, Any]) -> str:
    if not metrics:
        return f"<h3>{html.escape(title)}</h3><p>No clips.</p>"
    def cell(key: str) -> str:
        value = metrics.get(key)
        shown = "n/a" if value is None else value
        return f"<li>{html.escape(key)}: {html.escape(str(shown))}</li>"
    keys = ("clips", "accuracy", "precision", "recall", "mean_time_to_alert", "false_alarms_per_hour")
    items = "".join(cell(key) for key in keys)
    caveat = html.escape(str(metrics.get("caveat") or ""))
    return f"<h3>{html.escape(title)}</h3><ul>{items}</ul><p>{caveat}</p>"


def _clip_row(clip: dict[str, Any]) -> str:
    passed = "pass" if clip.get("passed") else "fail"
    image = ""
    if clip.get("keyframe"):
        image = f'<img alt="keyframe" src="data:image/jpeg;base64,{clip["keyframe"]}">'
    time_to = "n/a" if clip.get("time_to_alert") is None else clip["time_to_alert"]
    error = f"<br>{html.escape(clip['error'])}" if clip.get("error") else ""
    synthetic = " synthetic" if clip.get("synthetic") or clip.get("synthetic_glint") else ""
    return (
        "<tr>"
        f"<td>{html.escape(str(clip.get('label')))}{synthetic}{error}</td>"
        f"<td>{html.escape(str(clip.get('expected')))}</td>"
        f"<td>{html.escape(str(clip.get('outcome')))}</td>"
        f"<td class='{passed}'>{passed}</td>"
        f"<td>{html.escape(str(time_to))}</td>"
        f"<td>{html.escape(str(clip.get('max_score')))}</td>"
        f"<td>{html.escape(str(clip.get('seat') or ''))}</td>"
        f"<td>{image}</td>"
        "</tr>"
    )

# Architecture

CineGuard is one process. `uvicorn app:app` loads `app.py`, which owns the camera thread, the pose model, the incident database, and the dashboard files in `static/`.

```text
LiveSource or FileSource
        |  Frame(id, ts, image)     ts is frame time, not the wall clock
        v
PoseVectorEngine  ->  SpatialProximityTracker  ->  per-track stability
        |                                              |
        v                                              v
NIRSimulator (simulated) or passthrough (real / off)
        |
        v
GlintDetector inside a capped hand region
        |
        v
score_track + PiracyStateMachine per track id
        |
        +--> AuditoriumSeatMapper (homography, or the uncalibrated 4x10 grid)
        |
        +--> IncidentStore (SQLite) and EvidenceStore (keyframe + clip)
        |
        +--> one JPEG on the MJPEG bus, one telemetry dict on the WebSocket
                    |
                    v
             browser canvas (the server frame stays clean)
```

## Modules

| Path | Role |
| --- | --- |
| `sources.py` | Webcam/RTSP latest-frame capture, or a file paced by fps. The scenario runner turns pacing off. |
| `engine/pose_detector.py` | YOLOv8-pose and the elevated-hand rule. Weights are local only. |
| `engine/tracker.py` | Short-occlusion IDs. |
| `engine/nir.py` | Simulated mono image and the glint detector. |
| `engine/scoring.py` | Pose, sightline, glint, persistence, stability, and the behavior label. |
| `engine/state_machine.py` | IDLE, SUSPICIOUS, ALERT_PENDING, PIRACY_ALERT. Large timestamp gaps are clamped. |
| `engine/seat_mapper.py` | Four-point homography to a seat. Out-of-grid points stay `Unmapped`. |
| `incidents.py` | SQLite lifecycle, review, and CSV. A database error sets degraded and does not stop the loop. |
| `evidence.py` | Pre-roll ring, keyframe, clip, SHA-256. |
| `scenarios.py` | One isolated pass per manifest clip and the printable report. |
| `security.py` | Bearer token, body cap, rate limit, and response headers. |
| `static/` | Six-tab dashboard. No build step. |

## Time and threads

The processing thread is the only writer of engine state. HTTP handlers read snapshots or call `apply_config`. Scenario runs pause that thread so they do not share the pose model, then build a fresh tracker and a temporary database per clip.

## What the server refuses to invent

A missing model does not trigger a download. A missing calibration file does not pretend the grid was measured. Simulated NIR stays labeled in the API (`nir.mode`) and on the report.

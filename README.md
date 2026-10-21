# CineGuard

CineGuard is a proof of concept for spotting a phone held up in a cinema seat. It watches one camera, estimates pose, looks for a small bright retroreflection in a simulated near-infrared image, and opens an incident when both stay true long enough.

This is not a measured detector. The default NIR path is simulated. The numbers in the evaluation report are the result of the scripted clips you feed it, not a theatre trial.

## Limitations

- There is no NIR camera in this repository. `NIR_MODE=simulated` builds a red-weighted mono image. It must not be quoted as real-world accuracy.
- A manifest entry can inject a synthetic glint. Those incidents are marked `synthetic` in telemetry, the database, and the report.
- Eating, stretching, texting, and recording labels are geometric rules, not a trained classifier.
- Without `models/yolov8n-pose.pt` the app still starts, `/health` stays degraded, and no person is detected. The file is not downloaded for you.
- Seat labels say `(uncalibrated)` until you save four corners.
- False alarms per hour in the scenario report are extrapolated from short non-alert clips.

## Quick start (Windows, conda py310)

From this directory:

```powershell
conda activate py310
Copy-Item .env.example .env
# Place yolov8n-pose.pt in models\ yourself. Nothing is downloaded.
.\run.ps1 install
.\run.ps1 test
.\run.ps1 run
```

Open http://127.0.0.1:8000 . The header shows `NIR: SIMULATED` and, until you set `API_TOKEN`, an AUTH DISABLED warning.

`run.ps1 scenarios` writes `reports/report.html`. Record the four clips first or every row is a missing-file failure. See [demo_clips/README.md](demo_clips/README.md).

## Quick start (Docker)

The image installs the CPU PyTorch wheel. A GPU is optional: the process uses CPU when CUDA is not visible. To use an NVIDIA GPU, install the container toolkit and add `gpus: all` under the service in `docker-compose.yml`, then install a CUDA build of PyTorch in that image. The default compose file does not request a GPU.

```powershell
Copy-Item .env.example .env
New-Item -ItemType Directory -Force models, data, evidence, demo_clips | Out-Null
# Copy yolov8n-pose.pt into models\ before expecting detections.
docker compose up --build
```

Then open http://127.0.0.1:8000 . The image healthcheck calls `/health`. The container runs as user `cineguard` (uid 1000). The `data/`, `evidence/`, `demo_clips/`, and `models/` folders are mounted from the host.

## Architecture

```text
camera or file
    -> Frame (id, timestamp, image)
    -> pose -> tracker -> per-track score
    -> simulated NIR glint inside the hand region
    -> state machine on the frame timestamp
    -> seat homography
    -> incident row, keyframe, clip
    -> clean MJPEG plus a client canvas
```

Detail and the module map are in [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md). The NIR assumptions are in [docs/NIR.md](docs/NIR.md). The click-through list is in [docs/QA.md](docs/QA.md).

## API

| Method | Path | Auth when `API_TOKEN` is set |
| --- | --- | --- |
| GET | `/health` | no |
| GET | `/` | no |
| GET | `/report` | no |
| GET | `/api/v1/stream` | no |
| GET | `/api/v1/config` | no |
| POST | `/api/v1/config` | bearer |
| POST | `/api/v1/source` | bearer |
| POST | `/api/v1/source/control` | bearer |
| GET | `/api/v1/nir/params` | no |
| POST | `/api/v1/nir/params` | bearer |
| GET | `/api/v1/nir/frame.jpg` | no |
| GET | `/api/v1/nir/candidates` | no |
| GET, PUT | `/api/v1/calibration` | PUT only |
| POST | `/api/v1/calibration/probe` | bearer |
| GET | `/api/v1/incidents` | no |
| GET | `/api/v1/incidents/export.csv` | no |
| GET | `/api/v1/incidents/{id}` | no |
| POST | `/api/v1/incidents/{id}/review` | bearer |
| GET | `/api/v1/evidence/{id}/keyframe.jpg` | bearer or `?token=` |
| GET | `/api/v1/evidence/{id}/clip` | bearer or `?token=` |
| POST | `/api/v1/scenarios/run` | bearer |
| GET | `/api/v1/scenarios/results` | no |
| GET | `/api/v1/logs` | no |
| WebSocket | `/ws/live_telemetry` | `?token=` |

A missing or wrong token on a protected route returns 401. More than `RATE_LIMIT_PER_MINUTE` mutating requests from one address in a minute returns 429. A body over `MAX_BODY_BYTES` returns 413. Invalid JSON fields return 422.

Passwords inside RTSP URLs are stored for the capture thread and shown as `***` in `/api/v1/config` and in logs.

## Scenario clips

Record a raised phone held for the full threshold, a one-second glance, a hand at the mouth, and a brief two-arm stretch. The steps are in [demo_clips/README.md](demo_clips/README.md). Then:

```powershell
python -m tools.run_scenarios --manifest demo_clips/manifest.json --out reports
```

Open `reports/report.html` and use the browser print dialog for a PDF. The page says SIMULATED NIR.

## Switching to a real NIR camera later

1. Set `NIR_MODE=real`. The app then treats the camera as an already mono or NIR frame and skips the red-weighted simulation.
2. Use an 850 nm or 940 nm illuminator beside the lens, an IR-pass filter, a global shutter, and a locked exposure. The checklist is in [docs/NIR.md](docs/NIR.md).
3. Retune the NIR Lab sliders on real footage: percentile, area, circularity, contrast, and persistence. Leave synthetic glint off.
4. Re-record the four scenario clips under that lighting and run the report again. Those numbers are still a scripted check, not a field trial, until you say how the clips were collected.

## Privacy

Clips and keyframes are stored under `evidence/` and removed after `EVIDENCE_RETENTION_DAYS` (7 by default). The pose model looks for bodies and joints. It does not identify a face and it does not build a gallery. A cinema deployment still needs a sign at the entrance and a lawful basis. India's Digital Personal Data Protection Act, 2023, applies when this processing is done in India: say what is recorded, how long it is kept, and who can open an incident. Do not commit `.env`, evidence, or the database.

## Roadmap

Real NIR hardware, a collected dataset, a trained lens detector, and more than one camera. Authentication in this phase is a shared bearer token, not accounts.

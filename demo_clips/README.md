# Demo clips

`manifest.json` lists the four scripted cases the evaluation report expects. Drop extra clips in this folder and add a row. Paths are relative to this folder.

Record each clip from the same camera position you will calibrate. Keep the frame rate steady. A person should be fully visible from head to feet so the pose model and the seat mapper both have something to use. There is no NIR camera in this project: leave `synthetic_glint` false for real recordings. Set it true only when you want the software to paint a fake lens spot for a demo, and say so in `notes`.

1. **Recording** (`recording.mp4`, expected `alert`). Sit down, raise one hand with a phone above the shoulder, and hold it toward the screen for longer than `PIRACY_THRESHOLD_SEC` (30 seconds unless you change it). Do not bring the hand to your mouth.
2. **Glance** (`glance.mp4`, expected `no_alert`). Lift the phone for about a second, look at it, and put it down. The hold must be much shorter than the threshold.
3. **Eating** (`eating.mp4`, expected `no_alert`). Bring one hand to your mouth several times, as if eating popcorn. Keep the wrist close to the nose and the phone down.
4. **Stretching** (`stretching.mp4`, expected `no_alert`). Raise both arms for a moment, then lower them. Do not hold a phone up.

Then, from the `cineguard` directory:

```
python -m tools.run_scenarios --manifest demo_clips/manifest.json --out reports/
```

Open `reports/report.html` and use the browser print dialog to save a PDF. The banner on that page says SIMULATED NIR. The numbers are not a field measurement.

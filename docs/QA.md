# CineGuard UI checklist

Run the app from the `cineguard` directory with `uvicorn app:app --host 127.0.0.1 --port 8000`. Open `/` in a browser. These checks are manual; the automated test only confirms that the page and its scripts are served.

1. The header shows Live or Reconnecting, a source name, FPS, latency, and an NIR badge. Simulated mode is amber and reads `NIR: SIMULATED`. A missing pose model shows the DEGRADED banner. With no API token, AUTH DISABLED is visible.
2. Keys 1 through 6 switch tabs. `?` opens the shortcut list. `M` changes the mute button. `F` asks the video to go fullscreen. Space pauses or plays a file source.
3. Live: the canvas follows the video after a resize. Unchecking Boxes, Skeleton, Glints, Hand ROI, Seat labels, or Score labels removes that layer. Clicking a person fills the breakdown bars and the behavior reason. Opening a file from the demo list, then pause, step, and seek, changes the source state.
4. NIR Lab: Original, Simulated NIR, mask, and heatmap refresh about five times a second until Pause refresh. Moving a detector slider changes the mask or heatmap on the next refresh. The synthetic-glint control shows the demo-only warning. Compare with NIR off changes the badge to `NIR: OFF`.
5. Seat map: an unsaved calibration says uncalibrated. Freezing a frame, clicking four corners, and saving makes a probe click return a seat label. An alert colours that seat and the accessible name includes the state word, not only a colour.
6. Incidents: filters and pagination request the API. A row opens the keyframe, a video player, a download link, and the SHA-256 values. Acknowledge, False positive, and Confirmed send a review.
7. Scenarios: Run all shows progress, then pass or fail chips and the metric cards. Open printable report loads `/report`, which includes the SIMULATED NIR disclaimer and a Limitations section. The browser print dialog can save a PDF.
8. System: health lists the model and errors, the charts move, Apply config accepts a valid threshold and rejects an invalid one, and the log tail shows recent lines.
9. The console stays quiet during the checks above. A new piracy alert adds a toast. After a click anywhere, an unmuted alert also beeps.
10. With `API_TOKEN` empty, the AUTH DISABLED banner is visible and a config change succeeds. Set a token, restart, and confirm a POST without `Authorization: Bearer ...` returns 401, the same POST with the bearer succeeds, and the WebSocket connects only with `?token=`. Store that token in the System tab so evidence images load. The token must not appear in `/api/v1/config` or the log tail. An RTSP URL with a password must show `***` in config.

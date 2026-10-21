"""CineGuard Phase 1 capture diagnostics and stream health display."""

from __future__ import annotations

import signal
import time
from collections import deque

import cv2

from config import AppConfig
from stream import ThreadedRTSPStream
from utils.logger import get_logger


def main() -> None:
    config = AppConfig()
    logger = get_logger(__name__)
    stream = ThreadedRTSPStream(config)
    stopping = False

    def request_stop(_signum: int, _frame: object) -> None:
        nonlocal stopping
        stopping = True

    signal.signal(signal.SIGINT, request_stop)
    if hasattr(signal, "SIGTERM"):
        signal.signal(signal.SIGTERM, request_stop)

    fps_samples: deque[float] = deque(maxlen=30)
    last_display_at = time.monotonic()
    displayed_frames = 0
    stream.start()
    try:
        while not stopping:
            ok, frame = stream.read_latest()
            now = time.monotonic()
            if ok and frame is not None and frame.size:
                displayed_frames += 1
                elapsed = now - last_display_at
                if elapsed > 0:
                    fps_samples.append(1.0 / elapsed)
                last_display_at = now
                average_fps = sum(fps_samples) / len(fps_samples) if fps_samples else 0.0
                stats = stream.stats()
                healthy = stream.is_alive() and (now - stats["last_frame_at"] <= config.io_timeout)
                status = "HEALTHY" if healthy else "DEGRADED"
                cv2.putText(
                    frame,
                    f"FPS {average_fps:05.1f} | {status}",
                    (20, 40),
                    cv2.FONT_HERSHEY_DUPLEX,
                    0.9,
                    (0, 255, 0) if healthy else (0, 0, 255),
                    2,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    frame,
                    f"Captured {stats['frames_captured']} | Dropped {stats['frames_dropped']} | Reconnects {stats['reconnects']}",
                    (20, 78),
                    cv2.FONT_HERSHEY_DUPLEX,
                    0.65,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )
                cv2.imshow("CineGuard AI - Phase 1", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    stopping = True
            else:
                time.sleep(0.005)
    finally:
        stream.stop()
        cv2.destroyAllWindows()
        logger.info("Stopped cleanly after displaying %d frames", displayed_frames)


if __name__ == "__main__":
    main()

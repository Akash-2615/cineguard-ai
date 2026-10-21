"""Write results.json and a printable report for a clip manifest.

Usage, from the cineguard directory:

    python -m tools.run_scenarios --manifest demo_clips/manifest.json --out reports/
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from app import ModelBundle
from config import AppConfig
from scenarios import render_report, run_manifest
from utils.logger import get_logger

logger = get_logger(__name__)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run CineGuard scenario clips and write a report.")
    parser.add_argument("--manifest", default="demo_clips/manifest.json")
    parser.add_argument("--out", default="reports")
    args = parser.parse_args(argv)
    manifest = Path(args.manifest)
    if not manifest.is_file():
        logger.error("manifest not found: %s", manifest)
        return 2
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    config = AppConfig()
    models = ModelBundle.load(config)
    if not models.loaded:
        logger.error("pose model unavailable (%s); clips will not raise alerts", models.error)
    payload = run_manifest(manifest, config, models)
    (out / "results.json").write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    (out / "report.html").write_text(render_report(payload), encoding="utf-8")
    passed = payload["metrics"].get("passed")
    total = payload["metrics"].get("clips")
    logger.info("wrote %s (%s/%s passed)", out / "report.html", passed, total)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

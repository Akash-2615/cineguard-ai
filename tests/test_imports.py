"""Absolute imports only, so uvicorn app:app works from the repo root."""

from pathlib import Path


def test_no_relative_imports() -> None:
    root = Path(__file__).resolve().parents[1]
    offenders: list[str] = []
    for path in root.rglob("*.py"):
        if any(part in {".git", ".venv", "tests", "__pycache__"} for part in path.parts):
            continue
        text = path.read_text(encoding="utf-8")
        for line in text.splitlines():
            stripped = line.strip()
            if stripped.startswith("from .") or stripped.startswith("from .."):
                offenders.append(f"{path.name}: {stripped}")
    assert offenders == []

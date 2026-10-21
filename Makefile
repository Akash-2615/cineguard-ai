.PHONY: install run test scenarios

install:
	python -m pip install -r requirements.txt -r requirements-dev.txt

run:
	uvicorn app:app --host 127.0.0.1 --port 8000

test:
	python -m ruff check .
	python -m pytest -q

scenarios:
	python -m tools.run_scenarios --manifest demo_clips/manifest.json --out reports

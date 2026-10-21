param(
    [ValidateSet("install", "run", "test", "scenarios")]
    [string]$Task = "run"
)

$Python = Join-Path $env:USERPROFILE "anaconda3\envs\py310\python.exe"
if (-not (Test-Path $Python)) { $Python = "python" }

switch ($Task) {
    "install" {
        & $Python -m pip install -r requirements.txt -r requirements-dev.txt
    }
    "test" {
        & $Python -m ruff check .
        if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
        & $Python -m pytest -q
    }
    "scenarios" {
        & $Python -m tools.run_scenarios --manifest demo_clips/manifest.json --out reports
    }
    default {
        & $Python -m uvicorn app:app --host 127.0.0.1 --port 8000
    }
}
exit $LASTEXITCODE

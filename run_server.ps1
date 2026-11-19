# Start the local OCR inference backend (FastAPI + uvicorn).
# Run from the project root; requires dependencies installed in the current Python env.
$env:PYTHONIOENCODING = "utf-8"
$env:OMP_NUM_THREADS = "8"
$env:MKL_NUM_THREADS = "8"
Set-Location $PSScriptRoot

Write-Host "Starting OCR backend on http://0.0.0.0:8000 ..."
Write-Host "(first request loads the models, please wait ~30s)"
python -m uvicorn deploy.server:app --host 0.0.0.0 --port 8000

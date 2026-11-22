Set-Location $PSScriptRoot
Write-Host "Starting PeakCut Studio..." -ForegroundColor Cyan
if (Test-Path ".venv\Scripts\python.exe") {
    & ".venv\Scripts\python.exe" -m studio.server
} else {
    python -m studio.server
}

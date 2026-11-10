[CmdletBinding()]
param()

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
$VenvPath = Join-Path $RepoRoot ".venv"
$VenvPython = Join-Path $VenvPath "Scripts\python.exe"

if (-not (Test-Path -LiteralPath $VenvPython)) {
    Write-Host "Creating Python environment at $VenvPath"
    python -m venv $VenvPath
}

Write-Host "Installing the project and test dependencies"
& $VenvPython -m pip install --upgrade pip
& $VenvPython -m pip install -e "$RepoRoot[dev]"

Write-Host "Checking the smoke-run configuration"
Push-Location $RepoRoot
try {
    & $VenvPython -m hallucination_detection doctor --config "configs\smoke.toml"
}
finally {
    Pop-Location
}

Write-Host "Setup complete. Run .\scripts\run_tests.ps1 next."


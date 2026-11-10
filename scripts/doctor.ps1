[CmdletBinding()]
param(
    [ValidateSet("smoke", "pilot", "full", "synthetic")]
    [string]$Profile = "smoke"
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
$VenvPython = Join-Path $RepoRoot ".venv\Scripts\python.exe"
$Python = if (Test-Path -LiteralPath $VenvPython) { $VenvPython } else { "python" }
$env:PYTHONPATH = Join-Path $RepoRoot "src"

Push-Location $RepoRoot
try {
    & $Python -m hallucination_detection doctor --config "configs\$Profile.toml"
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}
finally {
    Pop-Location
}


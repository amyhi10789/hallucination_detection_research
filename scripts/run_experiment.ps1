[CmdletBinding()]
param(
    [ValidateSet("smoke", "pilot", "full", "synthetic")]
    [string]$Profile = "smoke",
    [string]$ResumeRun = ""
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
$VenvPython = Join-Path $RepoRoot ".venv\Scripts\python.exe"
$Python = if (Test-Path -LiteralPath $VenvPython) { $VenvPython } else { "python" }
$Config = "configs\$Profile.toml"
$Arguments = @("-m", "hallucination_detection", "run", "--config", $Config)
if ($ResumeRun) {
    $Arguments += @("--resume-run", $ResumeRun)
}

$env:PYTHONUTF8 = "1"
$env:PYTHONPATH = Join-Path $RepoRoot "src"
Push-Location $RepoRoot
try {
    & $Python @Arguments
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}
finally {
    Pop-Location
}


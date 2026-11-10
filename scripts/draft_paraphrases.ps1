[CmdletBinding()]
param(
    [ValidateSet("smoke", "pilot", "full")]
    [string]$Profile = "pilot",
    [int]$Limit = 0
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
$VenvPython = Join-Path $RepoRoot ".venv\Scripts\python.exe"
$Python = if (Test-Path -LiteralPath $VenvPython) { $VenvPython } else { "python" }
$Output = "data\paraphrase_candidates_$Profile.csv"
$Arguments = @(
    "-m", "hallucination_detection", "draft-paraphrases",
    "--config", "configs\$Profile.toml",
    "--output", $Output
)
if ($Limit -gt 0) {
    $Arguments += @("--limit", "$Limit")
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


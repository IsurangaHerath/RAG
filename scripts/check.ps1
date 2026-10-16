<#
.SYNOPSIS
    Runs every quality check for this project in one command (Windows).

.DESCRIPTION
    Stage 07 standard workflow. Executes, in order:

      1. ruff format --check   -> is the code formatted?
      2. ruff check            -> lint errors (unused imports, bugs, style)
      3. mypy app              -> static type checking
      4. pytest                -> unit + integration tests, with coverage

    Stops at the first failure so you fix problems in a sensible order.

.EXAMPLE
    .\scripts\check.ps1              # run everything
    .\scripts\check.ps1 -SkipTypes   # skip mypy (faster inner loop)
    .\scripts\check.ps1 -Fix          # auto-format + auto-fix lint issues first
#>
[CmdletBinding()]
param(
    # Apply ruff formatting and safe autofixes before checking.
    [switch]$Fix,
    # Skip type checking when you only want fast feedback.
    [switch]$SkipTypes,
    # Skip tests when you only want static checks.
    [switch]$SkipTests
)

$ErrorActionPreference = "Stop"
Set-Location -Path (Split-Path -Parent $PSScriptRoot)

# Always run from inside the virtualenv if it exists; otherwise the commands
# will not be found at all, which is a confusing failure for a newcomer.
$venvPython = Join-Path $PSScriptRoot "..\.venv\Scripts\python.exe"
if (Test-Path $venvPython) {
    $Python = $venvPython
} else {
    Write-Warning ".venv not found - falling back to python on PATH."
    Write-Warning "Create it with:  python -m venv .venv"
    $Python = "python"
}

function Invoke-Step {
    param(
        [Parameter(Mandatory)][string]$Name,
        [Parameter(Mandatory)][string[]]$Args
    )
    Write-Host ""
    Write-Host "==> $Name" -ForegroundColor Cyan
    & $Python @Args
    if ($LASTEXITCODE -ne 0) {
        Write-Host "FAILED: $Name" -ForegroundColor Red
        exit $LASTEXITCODE
    }
}

if ($Fix) {
    Invoke-Step "ruff format (apply)" @("-m", "ruff", "format", ".")
    Invoke-Step "ruff check --fix (apply)" @("-m", "ruff", "check", ".", "--fix")
}

Invoke-Step "ruff format --check" @("-m", "ruff", "format", "--check", ".")
Invoke-Step "ruff check (lint)" @("-m", "ruff", "check", ".")

if (-not $SkipTypes) {
    Invoke-Step "mypy (types)" @("-m", "mypy", "app")
}

if (-not $SkipTests) {
    Invoke-Step "pytest (tests + coverage)" @("-m", "pytest", "--cov=app", "--cov-report=term-missing", "-q")
}

Write-Host ""
Write-Host "All checks passed." -ForegroundColor Green

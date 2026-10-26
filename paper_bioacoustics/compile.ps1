# Compile paper\main.tex to paper\main.pdf with the tectonic binary kept inside the project (tools\tectonic), keeping every cache on D:.
# Usage (from anywhere):   powershell -File D:\AI\Claude\research\bat_audit\paper\compile.ps1
# tectonic downloads the TeX files it needs on first use into tools\tectonic_cache (D:).  Its font library (fontconfig) always writes a ~2 MB cache to
# %LOCALAPPDATA%\fontconfig on C: and cannot be redirected; this script removes that folder afterwards, but only if it did not exist before the run.
$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
$fc = Join-Path $env:LOCALAPPDATA 'fontconfig'
$existedBefore = Test-Path -LiteralPath $fc
$env:TECTONIC_CACHE_DIR = Join-Path $root 'tools\tectonic_cache'
Push-Location $PSScriptRoot
try {
    foreach ($doc in 'manuscript') {
        & (Join-Path $root 'tools\tectonic\tectonic.exe') -X compile "$doc.tex" --outdir build
        Copy-Item -LiteralPath (Join-Path $PSScriptRoot "build\$doc.pdf") -Destination (Join-Path $PSScriptRoot "$doc.pdf") -Force
        Write-Output "PDF: $(Join-Path $PSScriptRoot "$doc.pdf")"
    }
}
finally {
    Pop-Location
    if (-not $existedBefore -and (Test-Path -LiteralPath $fc)) {
        Remove-Item -LiteralPath $fc -Recurse -Force
        Write-Output 'removed the fontconfig cache that tectonic created on C:'
    }
}

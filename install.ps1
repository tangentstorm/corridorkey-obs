<#
.SYNOPSIS
    Builds and installs the CorridorKey OBS filter.

.DESCRIPTION
    Copies the plugin DLL, the effect, and any exported ONNX models into OBS's
    third-party plugin directory.

    By default this is %ProgramData%\obs-studio\plugins, which is the path OBS
    for Windows actually scans (frontend/widgets/OBSBasic.cpp calls
    GetProgramDataPath, not GetAppConfigPath — so %APPDATA%\obs-studio\plugins
    is silently ignored, despite what a lot of guides say). Standard users can
    normally write there without elevation.

    -IntoObsDir installs into the OBS program folder instead, which does need an
    elevated shell.

.EXAMPLE
    .\install.ps1
    .\install.ps1 -IntoObsDir
    .\install.ps1 -IntoObsDir -ObsDir "D:\obs-studio"
#>
[CmdletBinding()]
param(
    [string]$ObsDir = "C:\Program Files\obs-studio",
    [switch]$IntoObsDir,
    [switch]$SkipBuild
)

$ErrorActionPreference = "Stop"
$root = $PSScriptRoot

# OBS resolves a module's data directory from the DLL's base name, so these two
# must agree.
$PluginName = "corridorkey-obs"

if ($IntoObsDir) {
    if (-not (Test-Path (Join-Path $ObsDir "bin\64bit\obs.dll"))) {
        throw "No OBS install at '$ObsDir' (expected bin\64bit\obs.dll). Pass -ObsDir to point at yours."
    }
    $binDir = Join-Path $ObsDir "obs-plugins\64bit"
    $dataDir = Join-Path $ObsDir "data\obs-plugins\$PluginName"
} else {
    $base = Join-Path $env:ProgramData "obs-studio\plugins\$PluginName"
    $binDir = Join-Path $base "bin\64bit"
    $dataDir = Join-Path $base "data"
}

if (-not $SkipBuild) {
    Write-Host "Building (release)..." -ForegroundColor Cyan
    Push-Location $root
    try {
        # Windows PowerShell 5.1 turns any native stderr output into a
        # NativeCommandError under $ErrorActionPreference='Stop', and cargo writes
        # build-script warnings there on every successful build. Judge the build
        # by its exit code instead.
        $prev = $ErrorActionPreference
        $ErrorActionPreference = 'Continue'
        try {
            & cargo build --release
        } finally {
            $ErrorActionPreference = $prev
        }
        if ($LASTEXITCODE -ne 0) { throw "cargo build failed (exit $LASTEXITCODE)" }
    } finally {
        Pop-Location
    }
}

$dll = Join-Path $root "target\release\corridorkey_obs.dll"
if (-not (Test-Path $dll)) { throw "Built plugin not found at $dll" }

foreach ($d in @($binDir, (Join-Path $dataDir "effects"), (Join-Path $dataDir "models"))) {
    New-Item -ItemType Directory -Force -Path $d | Out-Null
}

# Verify we can actually write before copying anything, so a non-elevated run
# fails with a useful message instead of halfway through.
try {
    $probe = Join-Path $binDir ".corridorkey-write-test"
    [System.IO.File]::WriteAllText($probe, "")
    Remove-Item $probe -Force
} catch {
    throw "Cannot write to '$binDir'. Re-run this script from an elevated PowerShell."
}

Write-Host "Installing to $binDir" -ForegroundColor Cyan
Copy-Item $dll (Join-Path $binDir "$PluginName.dll") -Force

Copy-Item (Join-Path $root "data\effects\corridorkey.effect") (Join-Path $dataDir "effects") -Force

$models = @(Get-ChildItem (Join-Path $root "models\*.onnx") -ErrorAction SilentlyContinue)
if ($models.Count -eq 0) {
    Write-Warning @"
No ONNX models found in $root\models.
The filter will load but stay bypassed until one exists. Generate it with:
    cd tools\export
    uv run --python .venv\Scripts\python.exe python export_onnx.py --size 512 --color green
"@
} else {
    foreach ($m in $models) {
        Write-Host "  model: $($m.Name) ($([math]::Round($m.Length / 1MB)) MB)"
        Copy-Item $m.FullName (Join-Path $dataDir "models") -Force
    }
}

Write-Host ""
Write-Host "Installed." -ForegroundColor Green
Write-Host "Restart OBS, then add 'CorridorKey (Neural Green Screen)' as a filter on your camera source."

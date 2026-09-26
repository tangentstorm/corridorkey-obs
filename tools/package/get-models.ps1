<#
.SYNOPSIS
    Downloads the CorridorKey checkpoints and converts them to ONNX for this plugin.

.DESCRIPTION
    The models are not shipped with the plugin: they derive from checkpoints under
    a non-commercial licence, so this fetches them from the author's HuggingFace
    directly and converts them locally.

    Needs git and uv (https://docs.astral.sh/uv/). uv handles Python itself, so
    you do not need Python installed. Expect a few minutes and roughly 2GB of
    one-off downloads for the conversion toolchain.

    Converted models land in plugin\data\models\, ready for install.ps1.

.PARAMETER Sizes
    Inference resolutions to build, in pixels. Each is a separate ~136MB model and
    becomes an option in the filter's "Quality / speed" setting. Smaller is
    faster, which reduces how far the matte lags a moving subject.

.PARAMETER Colors
    Screen colours to build. Blue is only worth it if you shoot against blue.

.EXAMPLE
    .\get-models.ps1
    .\get-models.ps1 -Sizes 512 -Colors green
#>
[CmdletBinding()]
param(
    [int[]]$Sizes = @(512, 384, 256),
    [ValidateSet("green", "blue")]
    [string[]]$Colors = @("green"),
    [switch]$KeepBuildFiles
)

$ErrorActionPreference = "Stop"
$root = $PSScriptRoot
$work = Join-Path $root ".model-build"
$dest = Join-Path $root "plugin\data\models"

foreach ($tool in "git", "uv") {
    if (-not (Get-Command $tool -ErrorAction SilentlyContinue)) {
        throw @"
'$tool' was not found on PATH.
  git: https://git-scm.com/download/win
  uv : https://docs.astral.sh/uv/getting-started/installation/
"@
    }
}

if (-not (Test-Path (Join-Path $root "export_onnx.py"))) {
    throw "export_onnx.py is missing next to this script; re-download the release package."
}

New-Item -ItemType Directory -Force -Path $work, $dest | Out-Null

# export_onnx.py imports the model definition from the upstream repository rather
# than restating a 300-line architecture that would then have to be kept in sync.
$upstream = Join-Path $work "ref\CorridorKey"
if (Test-Path (Join-Path $upstream ".git")) {
    Write-Host "upstream CorridorKey already present" -ForegroundColor Cyan
} else {
    Write-Host "Cloning CorridorKey (model definition only, no checkpoints)..." -ForegroundColor Cyan
    New-Item -ItemType Directory -Force -Path (Split-Path $upstream) | Out-Null
    & git clone --depth 1 https://github.com/nikopueringer/CorridorKey.git $upstream
    if ($LASTEXITCODE -ne 0) { throw "git clone failed" }
}

# export_onnx.py locates the checkout at <repo root>/ref/CorridorKey, two levels
# up from itself, so it runs from a matching layout inside the work directory.
$exportDir = Join-Path $work "tools\export"
New-Item -ItemType Directory -Force -Path $exportDir | Out-Null
Copy-Item (Join-Path $root "export_onnx.py") $exportDir -Force

$py = Join-Path $exportDir ".venv\Scripts\python.exe"
if (-not (Test-Path $py)) {
    Write-Host "Setting up the conversion environment (this is the slow part)..." -ForegroundColor Cyan
    Push-Location $exportDir
    try {
        # cargo and uv both write progress to stderr; under $ErrorActionPreference
        # = 'Stop' that would be treated as failure, so judge by exit code.
        $prev = $ErrorActionPreference
        $ErrorActionPreference = 'Continue'
        try {
            & uv venv --python 3.12 .venv
            if ($LASTEXITCODE -ne 0) { throw "uv venv failed" }
            & uv pip install --python .venv\Scripts\python.exe torch --index-url https://download.pytorch.org/whl/cpu
            if ($LASTEXITCODE -ne 0) { throw "installing torch failed" }
            & uv pip install --python .venv\Scripts\python.exe timm safetensors onnx onnxruntime huggingface_hub numpy pillow onnxconverter-common
            if ($LASTEXITCODE -ne 0) { throw "installing dependencies failed" }
        } finally {
            $ErrorActionPreference = $prev
        }
    } finally {
        Pop-Location
    }
}

$built = 0
foreach ($color in $Colors) {
    foreach ($size in $Sizes) {
        $name = "corridorkey_${color}_${size}_fp16.onnx"
        if (Test-Path (Join-Path $dest $name)) {
            Write-Host "already have $name" -ForegroundColor DarkGray
            $built++
            continue
        }
        Write-Host "Building $name ..." -ForegroundColor Cyan
        Push-Location $exportDir
        try {
            $prev = $ErrorActionPreference
            $ErrorActionPreference = 'Continue'
            try {
                # The script checks each export against PyTorch and refuses to
                # write a model that does not match, so a clean exit means good.
                & $py export_onnx.py --size $size --color $color --fp16
                $code = $LASTEXITCODE
            } finally {
                $ErrorActionPreference = $prev
            }
        } finally {
            Pop-Location
        }
        if ($code -ne 0) { throw "export failed for $color at ${size}px" }

        $produced = Join-Path $work "models\$name"
        if (-not (Test-Path $produced)) { throw "expected $produced to exist after export" }
        Move-Item $produced (Join-Path $dest $name) -Force
        $built++
    }
}

if (-not $KeepBuildFiles) {
    Write-Host "Cleaning up build files (pass -KeepBuildFiles to keep them for re-runs)..." -ForegroundColor DarkGray
    Remove-Item $work -Recurse -Force -ErrorAction SilentlyContinue
}

Write-Host ""
Write-Host "$built model(s) in $dest" -ForegroundColor Green
Get-ChildItem (Join-Path $dest "*.onnx") | ForEach-Object {
    Write-Host ("  {0}  ({1} MB)" -f $_.Name, [math]::Round($_.Length / 1MB))
}
Write-Host ""
Write-Host "Now run install.ps1, then restart OBS."

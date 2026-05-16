param(
    [ValidateSet("ai", "hand-calib", "pins-v2", "calib", "old-calib", "full", "batch-calib", "batch-hand", "batch-pins-v2", "batch-full")]
    [string]$Mode = "ai",

    [string]$Video = "..\data\patient_001\patient_001camP_1_20241121_10_22_13.mp4",
    [string]$InputRoot = "..\data",
    [string]$Pattern = "*camP_1*.mp4",
    [string]$OutputDir = ".\outputs",
    [int]$Limit = 0,
    [int]$MaxFrames = 0,
    [int]$TrainFrames = 100,
    [int]$FrameSamples = 120
)

$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$Python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"

function Resolve-ProjectPath {
    param([string]$PathValue)

    if ([System.IO.Path]::IsPathRooted($PathValue)) {
        return (Resolve-Path -LiteralPath $PathValue).Path
    }

    return (Resolve-Path -LiteralPath (Join-Path $ProjectRoot $PathValue)).Path
}

if (-not (Test-Path $Python)) {
    throw "Manjka .venv. Najprej zazeni: python -m venv .venv"
}

if ([System.IO.Path]::IsPathRooted($OutputDir)) {
    $ResolvedOutput = $OutputDir
}
else {
    $ResolvedOutput = Join-Path $ProjectRoot $OutputDir
}
New-Item -ItemType Directory -Force -Path $ResolvedOutput | Out-Null

if ($Mode -eq "ai") {
    $ResolvedVideo = Resolve-ProjectPath $Video
    & $Python (Join-Path $ProjectRoot "hand_ai_tracking.py") `
        --video $ResolvedVideo `
        --output-root $ResolvedOutput `
        --train-frames $TrainFrames `
        --max-frames $MaxFrames
}
elseif ($Mode -eq "hand-calib") {
    $ResolvedVideo = Resolve-ProjectPath $Video
    & $Python (Join-Path $ProjectRoot "hand_ai_tracking_calibrated.py") `
        --video $ResolvedVideo `
        --output-root $ResolvedOutput `
        --frame-samples $FrameSamples `
        --max-frames $MaxFrames
}
elseif ($Mode -eq "pins-v2") {
    $ResolvedVideo = Resolve-ProjectPath $Video
    & $Python (Join-Path $ProjectRoot "track_calibrated_hand_and_pins_v2.py") `
        --video $ResolvedVideo `
        --output-root $ResolvedOutput `
        --frame-samples $FrameSamples `
        --max-frames $MaxFrames
}
elseif ($Mode -eq "calib") {
    $ResolvedVideo = Resolve-ProjectPath $Video
    & $Python (Join-Path $ProjectRoot "track_calibrated_hand_and_pins.py") `
        --video $ResolvedVideo `
        --output-root $ResolvedOutput `
        --frame-samples $FrameSamples `
        --calibration-only
}
elseif ($Mode -eq "old-calib") {
    $ResolvedVideo = Resolve-ProjectPath $Video
    & $Python (Join-Path $ProjectRoot "hole_calibration.py") `
        --video $ResolvedVideo `
        --output-dir $ResolvedOutput `
        --frame-samples $FrameSamples
}
elseif ($Mode -eq "full") {
    $ResolvedVideo = Resolve-ProjectPath $Video
    & $Python (Join-Path $ProjectRoot "track_calibrated_hand_and_pins.py") `
        --video $ResolvedVideo `
        --output-root $ResolvedOutput `
        --frame-samples $FrameSamples `
        --max-frames $MaxFrames
}
elseif ($Mode -eq "batch-calib") {
    $ResolvedInputRoot = Resolve-ProjectPath $InputRoot
    & $Python (Join-Path $ProjectRoot "batch_test_calibrated_hand_and_pins.py") `
        --input-root $ResolvedInputRoot `
        --pattern $Pattern `
        --output-root $ResolvedOutput `
        --mode calibration `
        --frame-samples $FrameSamples `
        --limit $Limit
}
elseif ($Mode -eq "batch-hand") {
    $ResolvedInputRoot = Resolve-ProjectPath $InputRoot
    & $Python (Join-Path $ProjectRoot "batch_test_calibrated_hand_and_pins.py") `
        --input-root $ResolvedInputRoot `
        --pattern $Pattern `
        --output-root $ResolvedOutput `
        --mode hand `
        --frame-samples $FrameSamples `
        --max-frames $MaxFrames `
        --limit $Limit
}
elseif ($Mode -eq "batch-pins-v2") {
    $ResolvedInputRoot = Resolve-ProjectPath $InputRoot
    & $Python (Join-Path $ProjectRoot "batch_test_calibrated_hand_and_pins.py") `
        --input-root $ResolvedInputRoot `
        --pattern $Pattern `
        --output-root $ResolvedOutput `
        --mode pins-v2 `
        --frame-samples $FrameSamples `
        --max-frames $MaxFrames `
        --limit $Limit
}
elseif ($Mode -eq "batch-full") {
    $ResolvedInputRoot = Resolve-ProjectPath $InputRoot
    & $Python (Join-Path $ProjectRoot "batch_test_calibrated_hand_and_pins.py") `
        --input-root $ResolvedInputRoot `
        --pattern $Pattern `
        --output-root $ResolvedOutput `
        --mode full `
        --frame-samples $FrameSamples `
        --max-frames $MaxFrames `
        --limit $Limit
}

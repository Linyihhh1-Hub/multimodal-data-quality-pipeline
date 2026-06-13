param(
    [string]$Manifest = "data/raw/manifest.jsonl",
    [string]$RawDataDir = "data/raw",
    [string]$Version = "v1.0",
    [switch]$UseClip
)

$ErrorActionPreference = "Stop"

$argsList = @(
    "-m", "src.pipeline.run_pipeline",
    "--manifest", $Manifest,
    "--raw-data-dir", $RawDataDir,
    "--processed-dir", "data/processed",
    "--export-dir", "data/exports",
    "--version", $Version
)

if (-not $UseClip) {
    $argsList += "--no-clip"
}

python $argsList

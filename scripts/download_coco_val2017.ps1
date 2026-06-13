param(
    [string]$DownloadDir = "data/raw/coco_downloads",
    [string]$OutputDir = "data/raw/coco"
)

$ErrorActionPreference = "Stop"

New-Item -ItemType Directory -Force -Path $DownloadDir, $OutputDir | Out-Null

$annotationsUrl = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
$valImagesUrl = "http://images.cocodataset.org/zips/val2017.zip"

$annotationsZip = Join-Path $DownloadDir "annotations_trainval2017.zip"
$valImagesZip = Join-Path $DownloadDir "val2017.zip"

if (-not (Test-Path $annotationsZip)) {
    Invoke-WebRequest -Uri $annotationsUrl -OutFile $annotationsZip
}

if (-not (Test-Path $valImagesZip)) {
    Invoke-WebRequest -Uri $valImagesUrl -OutFile $valImagesZip
}

Expand-Archive -LiteralPath $annotationsZip -DestinationPath $OutputDir -Force
Expand-Archive -LiteralPath $valImagesZip -DestinationPath $OutputDir -Force

Write-Host "COCO val2017 data is ready under $OutputDir"

param(
    [int]$Port = 8501,
    [string]$MetadataPath = "data/processed/processed_metadata_v1.0.parquet",
    [int]$PortSearchLimit = 10,
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"

function Test-DashboardHealth {
    param([int]$CandidatePort)

    try {
        $response = Invoke-WebRequest `
            -UseBasicParsing `
            -TimeoutSec 2 `
            -Uri "http://127.0.0.1:$CandidatePort/_stcore/health"
        return [int]$response.StatusCode -eq 200
    } catch {
        return $false
    }
}

function Test-PortAvailable {
    param([int]$CandidatePort)

    try {
        $listeners = @(Get-NetTCPConnection -State Listen -LocalPort $CandidatePort -ErrorAction SilentlyContinue)
        if ($listeners.Count -gt 0) {
            return $false
        }
    } catch {
    }

    $client = $null
    try {
        $client = [System.Net.Sockets.TcpClient]::new()
        $connect = $client.ConnectAsync([System.Net.IPAddress]::Loopback, $CandidatePort)
        if ($connect.Wait(300) -and $client.Connected) {
            return $false
        }
    } catch {
    } finally {
        if ($null -ne $client) {
            $client.Close()
        }
    }

    $listener = $null
    try {
        $listener = [System.Net.Sockets.TcpListener]::new([System.Net.IPAddress]::Any, $CandidatePort)
        $listener.Start()
        return $true
    } catch {
        return $false
    } finally {
        if ($null -ne $listener) {
            $listener.Stop()
        }
    }
}

function Resolve-DashboardPort {
    param(
        [int]$StartPort,
        [int]$Limit
    )

    if ($Limit -lt 1) {
        throw "PortSearchLimit must be at least 1."
    }

    for ($offset = 0; $offset -lt $Limit; $offset++) {
        $candidate = $StartPort + $offset
        if (Test-DashboardHealth -CandidatePort $candidate) {
            return @{
                Port = $candidate
                Reuse = $true
            }
        }
        if (Test-PortAvailable -CandidatePort $candidate) {
            return @{
                Port = $candidate
                Reuse = $false
            }
        }
        Write-Host "Port $candidate is occupied; trying next port."
    }

    throw "No available ports from $StartPort across $Limit candidates."
}

$ProjectRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
$LocalTemp = Join-Path $ProjectRoot ".tmp\streamlit"
New-Item -ItemType Directory -Force -Path $LocalTemp | Out-Null

$env:TEMP = $LocalTemp
$env:TMP = $LocalTemp
$env:TMPDIR = $LocalTemp
$env:STREAMLIT_BROWSER_GATHER_USAGE_STATS = "false"

Set-Location $ProjectRoot

$ProjectPython = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
if (Test-Path $ProjectPython) {
    $Python = $ProjectPython
} else {
    $Python = "python"
}

$resolved = Resolve-DashboardPort -StartPort $Port -Limit $PortSearchLimit
$Port = [int]$resolved.Port
$Url = "http://localhost:$Port"

if ($resolved.Reuse) {
    Write-Host "Reusing healthy dashboard: $Url" -ForegroundColor Green
    exit 0
}

Write-Host "Starting dashboard: $Url" -ForegroundColor Green

if ($DryRun) {
    exit 0
}

if (-not (Test-Path $MetadataPath)) {
    Write-Host "Metadata file not found: $MetadataPath" -ForegroundColor Yellow
    Write-Host "Run the local pipeline first, for example:" -ForegroundColor Yellow
    Write-Host "  python scripts/create_demo_data.py"
    Write-Host "  python -m src.cli run --config configs/pipeline.yaml"
}

& $Python -m streamlit run src/dashboard/app.py `
    --server.port $Port `
    --server.headless true `
    --server.fileWatcherType none `
    --browser.gatherUsageStats false

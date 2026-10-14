# launch-aurora-llamacpp.ps1
# Launch Aurora on llama.cpp llama-server via the OpenAIBrain backend.
#
# - Starts (or connects to) a local llama-server serving the Heretic
#   Qwen3-8B Q4_K_M GGUF. Downloads nothing: if the model or server binary
#   is missing it prints exactly what is needed and exits.
# - Selects PRIMORA_AURORA_BACKEND=openai for this process only; the
#   project default (transformers) is untouched, and no model file is
#   overwritten.
# - Runs the live driver, which wires Prime + Aurora to the canonical
#   production memory database (PRIMORA/data/context/aurora_context.db).
#   Live conversations accumulate in production memory as intended.
# - All output is appended to logs\launch-aurora-llamacpp.log. On any
#   failure the console stays open so the error can be read.
#
# Usage:
#   .\launch-aurora-llamacpp.ps1              # start server if needed + run checks
#   .\launch-aurora-llamacpp.ps1 -Chat        # after checks, open a chat loop
#   .\launch-aurora-llamacpp.ps1 -KeepServer  # leave llama-server running
#   .\launch-aurora-llamacpp.ps1 -Port 8081
#   .\launch-aurora-llamacpp.ps1 -NoPause     # do not hold the window open on failure

param(
    [string]$ModelPath = "C:\Users\M\Downloads\qwen3-8b-heretic-Q4_K_M.gguf",
    [string]$ServerExe = "C:\Users\M\Downloads\Llama\llama-b10615-bin-win-cuda-12.4-x64\llama-server.exe",
    [string]$HostAddr = "127.0.0.1",
    [int]$Port = 8080,
    [int]$CtxSize = 4096,
    [string]$ModelAlias = "qwen3-8b-heretic",
    [switch]$Chat,
    [switch]$KeepServer,
    [switch]$NoPause,
    [string]$LogPath = ""
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$BaseUrl = "http://${HostAddr}:${Port}/v1"

# --- Logging ---------------------------------------------------------------------
if (-not $LogPath) {
    $logDir = Join-Path $ProjectRoot "logs"
    if (-not (Test-Path -LiteralPath $logDir)) {
        New-Item -ItemType Directory -Path $logDir | Out-Null
    }
    $LogPath = Join-Path $logDir "launch-aurora-llamacpp.log"
}

$script:CanPause = $false
try {
    $script:CanPause = (-not $NoPause) -and
        [Environment]::UserInteractive -and
        (-not [Console]::IsInputRedirected)
} catch {
    $script:CanPause = $false
}

function Close-WithError([string]$Message, [int]$Code) {
    Write-Host ""
    Write-Host "ERROR: $Message" -ForegroundColor Red
    Write-Host "Full log: $LogPath" -ForegroundColor Yellow
    try { Stop-Transcript | Out-Null } catch {}
    if ($script:CanPause) {
        try { Read-Host "Press Enter to close" | Out-Null } catch {}
    }
    exit $Code
}

# Capture PowerShell and child-process output so failures survive the window.
try {
    Start-Transcript -Path $LogPath -Append | Out-Null
} catch {
    Write-Host "WARNING: could not start transcript at $LogPath : $_" -ForegroundColor Yellow
}
Write-Host "=== launch-aurora-llamacpp started $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') ==="
Write-Host "Command: $($MyInvocation.Line)"
if ($Chat) {
    Write-Host "NOTE: with -Chat, conversation text is also captured in this log." -ForegroundColor Yellow
}

trap {
    Close-WithError ("Unhandled error: " + (($_ | Out-String).Trim())) 1
}

function Test-File([string]$Path, [string]$What) {
    if (-not (Test-Path -LiteralPath $Path)) {
        Write-Host ""
        Write-Host "MISSING: $What" -ForegroundColor Red
        Write-Host "  Expected at: $Path"
        Write-Host "Nothing was downloaded or installed. Place the file and re-run."
        Close-WithError "Required file not found: $What ($Path)" 2
    }
}

Test-File $ServerExe "llama-server binary (llama.cpp CUDA build)"
Test-File $ModelPath "Heretic Qwen3-8B Q4_K_M GGUF model"

# --- GPU sanity (informational) -------------------------------------------------
try {
    $gpu = nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv,noheader 2>$null
    Write-Host "GPU: $gpu"
} catch {
    Write-Host "WARNING: nvidia-smi unavailable; llama-server may fall back to CPU." -ForegroundColor Yellow
}

# --- Server: connect or start ---------------------------------------------------
function Get-ServerHealth {
    try {
        $r = Invoke-WebRequest -Uri "http://${HostAddr}:${Port}/health" -UseBasicParsing -TimeoutSec 2
        return $r.StatusCode -eq 200
    } catch { return $false }
}

$startedHere = $false
if (Get-ServerHealth) {
    Write-Host "Connecting to existing llama-server at $BaseUrl"
} else {
    Write-Host "Starting llama-server on ${HostAddr}:${Port} (ctx=$CtxSize, all layers on GPU)..."
    $serverArgs = @(
        "-m", $ModelPath,
        "--host", $HostAddr,
        "--port", "$Port",
        "-c", "$CtxSize",
        "-ngl", "99",
        "--alias", $ModelAlias
    )
    $serverProc = Start-Process -FilePath $ServerExe -ArgumentList $serverArgs -PassThru -WindowStyle Minimized
    $startedHere = $true

    $deadline = (Get-Date).AddSeconds(180)
    while ((Get-Date) -lt $deadline) {
        if ($serverProc.HasExited) {
            Close-WithError ("llama-server exited early (code $($serverProc.ExitCode)). Check GPU/CUDA availability and that port $Port is free.") 3
        }
        if (Get-ServerHealth) { break }
        Start-Sleep -Milliseconds 1000
    }
    if (-not (Get-ServerHealth)) {
        if ($startedHere -and -not $serverProc.HasExited) { Stop-Process -Id $serverProc.Id -Force }
        Close-WithError "llama-server did not become healthy within 180s." 3
    }
    Write-Host "llama-server healthy (PID $($serverProc.Id))." -ForegroundColor Green
}

# --- Environment for this process only ------------------------------------------
$env:PRIMORA_AURORA_BACKEND = "openai"
$env:PRIMORA_OPENAI_URL = $BaseUrl
$env:PRIMORA_OPENAI_MODEL = $ModelAlias
$env:PRIMORA_OPENAI_TIMEOUT = "300"
# Prime's own provider speaks to the same server so consult/research tooling
# is self-contained; session-only, default (ollama) untouched.
$env:PRIMORA_PROVIDER = "openai"
# No PRIMORA_SESSION_ID override: live conversations use the application's
# normal session behaviour (default unless the operator sets one).

# --- Run the live-test driver ---------------------------------------------------
$driver = Join-Path $ProjectRoot "launch_aurora_llamacpp.py"
$chatFlag = if ($Chat) { "--chat" } else { "" }
$python = (Get-Command python).Source
Write-Host "Driver: $python $driver $chatFlag"
& $python $driver $chatFlag
$exit = $LASTEXITCODE

# --- Shutdown -------------------------------------------------------------------
if ($startedHere -and -not $KeepServer) {
    Write-Host "Stopping llama-server (PID $($serverProc.Id))..."
    if (-not $serverProc.HasExited) { Stop-Process -Id $serverProc.Id -Force }
} elseif ($KeepServer) {
    Write-Host "Leaving llama-server running (-KeepServer)."
}

try { Stop-Transcript | Out-Null } catch {}
if ($exit -ne 0) {
    Write-Host ""
    Write-Host "Driver exited with code $exit. Full log: $LogPath" -ForegroundColor Red
    if ($script:CanPause) {
        try { Read-Host "Press Enter to close" | Out-Null } catch {}
    }
}
exit $exit

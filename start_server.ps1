# Start server script - Permanent port error fix
Write-Host "Starting SupportSentinelEnv Server..." -ForegroundColor Cyan

# Kill any existing process on port 7860
Write-Host "Cleaning up port 7860..." -ForegroundColor Yellow
$existingProcess = netstat -ano | findstr ":7860" | findstr "LISTENING"
if ($existingProcess) {
    $procId = ($existingProcess -split '\s+')[-1]
    Write-Host "Found process on port 7860: PID $procId"
    taskkill /PID $procId /F | Out-Null
    Write-Host "Process terminated [OK]" -ForegroundColor Green
}

# Wait for port to be released
Start-Sleep -Seconds 2

# Start the server
Write-Host "`nStarting Uvicorn server..." -ForegroundColor Cyan
python -m uvicorn server.app:app --host 0.0.0.0 --port 7860 --reload

if ($LASTEXITCODE -eq 0) {
    Write-Host "`nServer started successfully [OK]" -ForegroundColor Green
} else {
    Write-Host "`nServer failed to start [ERROR]" -ForegroundColor Red
}

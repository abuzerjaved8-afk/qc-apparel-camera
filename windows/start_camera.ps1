$pidFile = "$env:USERPROFILE\qc_camera.pid"
if (Test-Path $pidFile) {
    $oldPid = Get-Content $pidFile
    try { Stop-Process -Id $oldPid -Force -ErrorAction SilentlyContinue } catch {}
    Start-Sleep -Seconds 1
}

# "pythonw" on PATH can resolve to the Windows Store app-execution-alias
# stub, which spawns a SEPARATE real process rather than becoming it —
# tracking the stub's PID would then fail to stop the real one. Resolve
# past the stub to the actual interpreter.
$pythonCmd = Get-Command python.exe -All -ErrorAction SilentlyContinue |
    Where-Object { $_.Source -notmatch 'WindowsApps' } | Select-Object -First 1
if (-not $pythonCmd) {
    Write-Error "No real Python installation found (only the Windows Store alias)."
    exit 1
}
$pythonwPath = Join-Path (Split-Path $pythonCmd.Source) "pythonw.exe"

Set-Location $env:USERPROFILE
$env:PYTHONUNBUFFERED = "1"
$proc = Start-Process -FilePath $pythonwPath `
    -ArgumentList "`"$env:USERPROFILE\ip_camera_stream.py`"" `
    -WindowStyle Hidden -PassThru `
    -RedirectStandardOutput "$env:USERPROFILE\stream.log" `
    -RedirectStandardError "$env:USERPROFILE\stream_err.log"

$proc.Id | Out-File $pidFile

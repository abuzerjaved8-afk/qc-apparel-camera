# Refresh PATH from the registry in case Python was installed after this
# desktop session started (a freshly launched cmd/explorer still has the
# PATH that was current at login/boot time, not the installer's update).
$env:Path = [System.Environment]::GetEnvironmentVariable('Path', 'Machine') + ';' +
            [System.Environment]::GetEnvironmentVariable('Path', 'User')

# Stop-Process silently fails to kill a process that was started in a
# different Windows logon session than the one running this script — even
# for the same user account (e.g. a prior run started over SSH vs. this
# one running in the RDP desktop session). That left old instances alive
# and still holding the camera, fighting the new one for the device.
# Invoke-CimMethod's Terminate works across that session boundary, and we
# scan by command line instead of trusting a single possibly-stale PID
# file, so every leftover instance gets cleared no matter how it started.
$stale = Get-CimInstance Win32_Process -Filter "Name='pythonw.exe'" |
    Where-Object { $_.CommandLine -match [regex]::Escape('ip_camera_stream.py') }
foreach ($p in $stale) {
    Invoke-CimMethod -InputObject $p -MethodName Terminate | Out-Null
}
if ($stale) { Start-Sleep -Seconds 1 }

$pidFile = "$env:USERPROFILE\qc_camera.pid"

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

# Stop-Process silently fails across Windows logon session boundaries (see
# start_camera.ps1 for why) — use CIM Terminate and match by command line
# instead of a single possibly-stale PID, so every instance actually dies.
$stale = Get-CimInstance Win32_Process -Filter "Name='pythonw.exe'" |
    Where-Object { $_.CommandLine -match [regex]::Escape('ip_camera_stream.py') }
foreach ($p in $stale) {
    Invoke-CimMethod -InputObject $p -MethodName Terminate | Out-Null
}

$pidFile = "$env:USERPROFILE\qc_camera.pid"
Remove-Item $pidFile -ErrorAction SilentlyContinue

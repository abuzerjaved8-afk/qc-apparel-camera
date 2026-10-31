$pidFile = "$env:USERPROFILE\qc_camera.pid"
if (Test-Path $pidFile) {
    $oldPid = Get-Content $pidFile
    try { Stop-Process -Id $oldPid -Force -ErrorAction SilentlyContinue } catch {}
    Remove-Item $pidFile -ErrorAction SilentlyContinue
}

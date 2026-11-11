@echo off
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0Start-Training-Dashboard.ps1" -NoBrowser
if errorlevel 1 exit /b 1
start "" "http://127.0.0.1:8765/replay"
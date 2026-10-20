@echo off
setlocal
powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%~dp0tools\manage_installation.ps1" -Action Remove %*
set "FSIS_EXIT=%ERRORLEVEL%"
pause
exit /b %FSIS_EXIT%

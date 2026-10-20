@echo off
setlocal
powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%~dp0tools\manage_installation.ps1" -Action Start %*
set "FSIS_EXIT=%ERRORLEVEL%"
if not "%FSIS_EXIT%"=="0" pause
exit /b %FSIS_EXIT%

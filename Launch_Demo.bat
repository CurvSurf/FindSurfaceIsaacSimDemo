@echo off
setlocal
rem Compatibility alias: use the same dependency preparation as Start_Demo.
call "%~dp0Start_Demo.bat" %*
exit /b %ERRORLEVEL%

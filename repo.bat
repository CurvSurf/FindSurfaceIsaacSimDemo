@echo off
setlocal
if defined FSIS_PYTHON (
    "%FSIS_PYTHON%" "%~dp0tools\repo.py" %*
    exit /b %ERRORLEVEL%
)
where python >nul 2>nul
if not errorlevel 1 (
    python "%~dp0tools\repo.py" %*
    exit /b %ERRORLEVEL%
)
where py >nul 2>nul
if not errorlevel 1 (
    py -3 "%~dp0tools\repo.py" %*
    exit /b %ERRORLEVEL%
)
echo ERROR: Python 3 was not found. Set FSIS_PYTHON to a Python executable. 1>&2
exit /b 2

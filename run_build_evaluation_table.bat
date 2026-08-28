@echo off
setlocal

set "PYTHON_EXE=D:\python\python.exe"
if not exist "%PYTHON_EXE%" (
    echo Python was not found at %PYTHON_EXE%
    exit /b 1
)

"%PYTHON_EXE%" ".\python\summarize_build_stats.py" ^
  --without-prefilter ".\build_stats\without_prefilter.csv" ^
  --inline-prefilter ".\build_stats\inline_prefilter.csv" ^
  --output-dir ".\build_stats"

if errorlevel 1 exit /b %errorlevel%
echo Build evaluation tables are in .\build_stats
pause

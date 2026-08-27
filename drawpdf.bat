@echo off
setlocal

rem 切换到 BAT 文件所在的 HashDAG 目录
pushd "%~dp0"

set "PYTHONPATH=%CD%\.plot_dependencies"
set "MPLCONFIGDIR=%CD%\.matplotlib"

if not exist "%MPLCONFIGDIR%" mkdir "%MPLCONFIGDIR%"

"D:\python\python.exe" ".\python\plot_evaluation_frame_times.py" ^
  --original ".\original.stats.csv" ^
  --box-lod ".\box_lod.stats.csv" ^
  --relief-lod ".\relief_lod.stats.csv" ^
  --complete ".\complete.stats.csv" ^
  --output ".\evaluation_frame_times.pdf"

if errorlevel 1 (
    echo.
    echo Plot generation failed.
) else (
    echo.
    echo Plot generated successfully.
    start "" ".\evaluation_frame_times.pdf"
)

echo.
pause
popd
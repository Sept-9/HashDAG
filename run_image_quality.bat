@echo off
setlocal
pushd "%~dp0"

set "TORCH_HOME=%CD%\.torch_cache"

if not exist ".\.iq_venv\Scripts\python.exe" (
    echo Image-quality Python environment is missing.
    echo Create it and install requirements-image-quality.txt first.
    goto :failed
)

".\.iq_venv\Scripts\python.exe" ".\python\evaluate_image_quality.py" ^
  --ground-truth ".\screenshots\ground_truth.png" ^
  --manifest ".\image_quality_manifest.csv" ^
  --output-dir ".\image_quality_results"

if errorlevel 1 goto :failed

".\.iq_venv\Scripts\python.exe" ".\python\plot_image_quality_results.py" ^
  --metrics ".\image_quality_results\quality_metrics.csv" ^
  --manifest ".\image_quality_manifest.csv" ^
  --ground-truth ".\screenshots\ground_truth.png" ^
  --crop-x 285 ^
  --crop-y 752 ^
  --crop-radius 120 ^
  --output-dir ".\image_quality_results"

if errorlevel 1 goto :failed

echo.
echo Image-quality evaluation completed successfully.
echo CSV:   image_quality_results\quality_metrics.csv
echo LaTeX: image_quality_results\quality_metrics.tex
echo Plot:  image_quality_results\quality_metrics_plot.pdf
echo Matrix:image_quality_results\image_quality_matrix.pdf
goto :done

:failed
echo.
echo Image-quality evaluation failed.

:done
echo.
pause
popd

# Image-quality evaluation

## 1. Fix the camera and common rendering settings

Disable replay and automatic exit, then use the fixed camera macros printed by
Enter as described in `EVALUATION.md`:

```cpp
#define ENABLE_REPLAY 0
#define EXIT_AFTER_REPLAY 0
#define USE_GGX_BRDF 1
#define PER_VOXEL_FACE_SHADING 1
#define LINEAR_SPACE_SHADING 1
```

Also explicitly choose the same values for shadows, fog, resolution, lighting,
and all sampling macros in every build. Do not compare screenshots made with
different window or render resolutions.

## 2. Capture the ground truth

Build with:

```cpp
#define LOD_PIXEL_THRESHOLD 0.f
#define ENABLE_PREFILTERED_SHADING 0
#define ENABLE_COVERAGE_AWARE_LOD 0
#define SCREENSHOT_OUTPUT "screenshots/ground_truth.png"
```

Run and press V. The program saves the renderer's 1280x960 RGBA8 texture
directly, so the PNG never contains the UI even when the UI is visible.

## 3. Capture the nine comparisons

Use each threshold `0.5f`, `1.0f`, and `1.5f` for each method below. Change
`SCREENSHOT_OUTPUT` to match `image_quality_manifest.csv` before each build.

Box normal:

```cpp
#define ENABLE_PREFILTERED_SHADING 0
#define ENABLE_COVERAGE_AWARE_LOD 0
```

Relief:

```cpp
#define ENABLE_PREFILTERED_SHADING 1
#define PREFILTER_ROUGHNESS_FROM_VARIANCE 1
#define ENABLE_COVERAGE_AWARE_LOD 0
```

Coverage-aware relief:

```cpp
#define ENABLE_PREFILTERED_SHADING 1
#define PREFILTER_ROUGHNESS_FROM_VARIANCE 1
#define ENABLE_COVERAGE_AWARE_LOD 1
```

The two relief configurations deliberately use the same roughness setting so
the third group isolates the effect of coverage-aware descent.

## 4. Compute metrics and tables

Install the packages in `requirements-image-quality.txt`, then run from the
HashDAG project directory:

```text
python python/evaluate_image_quality.py --ground-truth screenshots/ground_truth.png --manifest image_quality_manifest.csv --output-dir image_quality_results
```

On this Windows workspace, `run_image_quality.bat` performs the same command
using the already prepared isolated Python environment and project-local LPIPS
model cache. Double-clicking it keeps the terminal open so errors remain visible.

The tool validates that every image has the same dimensions, discards alpha,
and computes RGB sRGB PSNR, SSIM, and LPIPS (AlexNet by default). It produces:

- `image_quality_results/quality_metrics.csv`
- `image_quality_results/quality_metrics.tex`
- `image_quality_results/quality_metrics_plot.pdf` and `.png`
- `image_quality_results/image_quality_matrix.pdf` and `.png`

The image matrix uses a 240x240 crop centred at `(x=285, y=752)`, where the
origin is the top-left corner of the PNG. These values can be overridden with
`--crop-x`, `--crop-y`, and `--crop-radius` when running
`python/plot_image_quality_results.py` directly.

The LaTeX table uses `booktabs`, so the report preamble needs
`\usepackage{booktabs}`.

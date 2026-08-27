# Epic Citadel 128k frame-time evaluation

The replay is loaded automatically from `replays/epiccitadel_move.csv` when
`USE_VIDEO` is off. Each replay frame records `frame_time_ms`, the sum of the
CUDA-event timings for path tracing, color resolution, and shadows.

## Recording a run

Set the shared replay options in `src/script_definitions.h`:

```cpp
#define SCENE "epiccitadel"
#define SCENE_DEPTH 17
#define REPLAY_DEPTH 17
#define REPLAY_NAME "move"
#define USE_VIDEO 0
#define REPLAY_TWICE 1
#define ENABLE_SHADOWS 0
#define EXIT_AFTER_REPLAY 1
```

For every build/run, add one configuration and give it a distinct output file.
The examples below use a threshold of `1.0f`; use the same positive threshold
for all three LOD configurations.

```cpp
// Original HashDAG
#define LOD_PIXEL_THRESHOLD 0.f
#define ENABLE_PREFILTERED_SHADING 0
#define ENABLE_COVERAGE_AWARE_LOD 0
#define PER_VOXEL_FACE_SHADING 0
#define FRAME_TIME_OUTPUT "original.stats.csv"

// Box LOD
#define LOD_PIXEL_THRESHOLD 1.0f
#define ENABLE_PREFILTERED_SHADING 0
#define ENABLE_COVERAGE_AWARE_LOD 0
#define PER_VOXEL_FACE_SHADING 0
#define FRAME_TIME_OUTPUT "box_lod.stats.csv"

// Relief LOD
#define LOD_PIXEL_THRESHOLD 1.0f
#define ENABLE_PREFILTERED_SHADING 1
#define PREFILTER_ROUGHNESS_FROM_VARIANCE 0
#define ENABLE_COVERAGE_AWARE_LOD 0
#define PER_VOXEL_FACE_SHADING 1
#define FRAME_TIME_OUTPUT "relief_lod.stats.csv"

// Complete method
#define LOD_PIXEL_THRESHOLD 1.0f
#define ENABLE_PREFILTERED_SHADING 1
#define PREFILTER_ROUGHNESS_FROM_VARIANCE 1
#define ENABLE_COVERAGE_AWARE_LOD 1
#define PER_VOXEL_FACE_SHADING 1
#define FRAME_TIME_OUTPUT "complete.stats.csv"
```

Only keep one configuration block active at a time. Rebuild after changing the
macros, then run from the project directory so the relative replay and output
paths resolve correctly. `REPLAY_TWICE=1` warms the GPU/data on the first pass;
only the second pass is saved.

## Plotting

After all four runs:

```text
python python/plot_evaluation_frame_times.py --original original.stats.csv --box-lod box_lod.stats.csv --relief-lod relief_lod.stats.csv --complete complete.stats.csv --output evaluation_frame_times.pdf
```

The default is the raw per-frame curve. Add `--smooth-window 10` to show a
10-frame moving average. The command also prints frame count, mean, median, and
95th-percentile GPU time for each configuration.

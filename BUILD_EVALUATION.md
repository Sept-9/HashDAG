# HashDAG construction and storage evaluation

This experiment changes only `ENABLE_PREFILTERED_SHADING`. Use the same scene,
compiler configuration, and all other feature macros for both runs. In
particular, set `ENABLE_COVERAGE_AWARE_LOD` to `0` in both builds so the
comparison isolates the inline prefilter.

## 1. Build without the inline prefilter

Put these definitions in `src/script_definitions.h`, build Release, and run once:

```cpp
#define ENABLE_PREFILTERED_SHADING 0
#define ENABLE_COVERAGE_AWARE_LOD 0
#define BUILD_STATS_OUTPUT "build_stats/without_prefilter.csv"
#define EXIT_AFTER_BUILD_STATS 1
```

## 2. Build with the inline prefilter

Change only the prefilter switch and output name, rebuild Release, and run once:

```cpp
#define ENABLE_PREFILTERED_SHADING 1
#define ENABLE_COVERAGE_AWARE_LOD 0
#define BUILD_STATS_OUTPUT "build_stats/inline_prefilter.csv"
#define EXIT_AFTER_BUILD_STATS 1
```

`EXIT_AFTER_BUILD_STATS` releases the construction data and closes the
application after writing the CSV, so rendering time is not part of this
experiment. Source BasicDAG file loading is also excluded from the reported
construction time.

Each run records:

- HashDAG geometry construction, inline-prefilter construction, and upload time;
- color hierarchy construction and upload time;
- total construction time;
- logical geometry storage;
- geometry pages actually allocated;
- logical color storage and total logical storage.

## 3. Generate the comparison table

Double-click `run_build_evaluation_table.bat`, or run:

```text
D:\python\python.exe python\summarize_build_stats.py --without-prefilter build_stats\without_prefilter.csv --inline-prefilter build_stats\inline_prefilter.csv --output-dir build_stats
```

This creates `build_stats/build_evaluation_table.csv` and
`build_stats/build_evaluation_table.tex`. Both include the two measurements,
their absolute difference, and percentage overhead.

For a paper-quality timing result, repeat each configuration several times and
report the median. The current two-file workflow intentionally represents one
run per configuration, as requested.

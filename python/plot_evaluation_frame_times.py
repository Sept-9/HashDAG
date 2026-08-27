"""Plot GPU frame time over replay frame number for LOD evaluation runs."""

from __future__ import annotations

import argparse
import csv
import statistics
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


CONFIGS = (
    ("original", "Original HashDAG", "#0072B2"),
    ("box_lod", "Box LOD", "#E69F00"),
    ("relief_lod", "Relief LOD", "#009E73"),
    ("complete", "Complete method", "#CC79A7"),
)


def load_frame_times(path: Path) -> Tuple[List[int], List[float]]:
    """Read long-form StatsRecorder CSV and return one GPU time per frame."""
    stats: Dict[int, Dict[str, float]] = {}
    with path.open(newline="", encoding="utf-8-sig") as stream:
        for line_number, row in enumerate(csv.reader(stream), start=1):
            if not row or all(not cell.strip() for cell in row):
                continue
            if len(row) != 3:
                raise ValueError(
                    f"{path}:{line_number}: expected frame,name,value; got {len(row)} columns"
                )
            try:
                frame = int(row[0])
                value = float(row[2])
            except ValueError as error:
                raise ValueError(f"{path}:{line_number}: invalid numeric value") from error
            stats.setdefault(frame, {})[row[1].strip()] = value

    if not stats:
        raise ValueError(f"{path}: no statistics found")

    frames = sorted(stats)
    times: List[float] = []
    for frame in frames:
        values = stats[frame]
        if "frame_time_ms" in values:
            times.append(values["frame_time_ms"])
            continue

        # Backward compatibility with results recorded before frame_time_ms was
        # added. Missing shadows mean that shadows were disabled for the run.
        if "paths" in values and "colors" in values:
            times.append(values["paths"] + values["colors"] + values.get("shadows", 0.0))
            continue
        raise ValueError(
            f"{path}: frame {frame} has neither frame_time_ms nor paths/colors timings"
        )

    return frames, times


def moving_average(values: List[float], window: int) -> List[float]:
    if window <= 1:
        return values
    smoothed: List[float] = []
    running_sum = 0.0
    for index, value in enumerate(values):
        running_sum += value
        if index >= window:
            running_sum -= values[index - window]
        count = min(index + 1, window)
        smoothed.append(running_sum / count)
    return smoothed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot Epic Citadel GPU frame time curves from StatsRecorder CSV files."
    )
    for key, label, _ in CONFIGS:
        parser.add_argument(
            "--" + key.replace("_", "-"),
            type=Path,
            metavar="CSV",
            help=f"{label} .stats.csv file",
        )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("evaluation_frame_times.pdf"),
        help="output figure (.pdf, .png, .svg, ...; default: %(default)s)",
    )
    parser.add_argument(
        "--smooth-window",
        type=int,
        default=1,
        metavar="N",
        help="moving-average window; 1 plots raw measurements (default: %(default)s)",
    )
    parser.add_argument(
        "--title",
        default="Epic Citadel 128k",
        help="figure title (default: %(default)s)",
    )
    args = parser.parse_args()
    if args.smooth_window < 1:
        parser.error("--smooth-window must be at least 1")
    if not any(getattr(args, key) for key, _, _ in CONFIGS):
        parser.error("provide at least one configuration CSV")
    return args


def main() -> None:
    args = parse_args()
    fig, ax = plt.subplots(figsize=(10, 5.6), dpi=160)

    for key, label, color in CONFIGS:
        path = getattr(args, key)
        if path is None:
            continue
        frames, raw_times = load_frame_times(path)
        times = moving_average(raw_times, args.smooth_window)
        ax.plot(frames, times, color=color, linewidth=1.35, label=label)

        sorted_times = sorted(raw_times)
        p95_index = min(len(sorted_times) - 1, int(0.95 * len(sorted_times)))
        print(
            f"{label}: {len(raw_times)} frames, "
            f"mean={statistics.fmean(raw_times):.3f} ms, "
            f"median={statistics.median(raw_times):.3f} ms, "
            f"p95={sorted_times[p95_index]:.3f} ms"
        )

    ax.set_title(args.title)
    ax.set_xlabel("Frame number")
    ax.set_ylabel("GPU frame time (ms)")
    ax.grid(True, color="#D9D9D9", linewidth=0.7, alpha=0.8)
    ax.set_axisbelow(True)
    ax.legend(frameon=False)
    ax.margins(x=0)
    fig.tight_layout()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, bbox_inches="tight")
    print(f"Saved figure: {args.output.resolve()}")


if __name__ == "__main__":
    main()

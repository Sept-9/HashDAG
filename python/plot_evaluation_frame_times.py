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


def summarize_performance(
    loaded: Dict[str, Tuple[List[int], List[float]]]
) -> List[Dict[str, object]]:
    """Compare every configuration with Original on matching replay frames."""
    if "original" not in loaded:
        return []

    frame_maps = {
        key: dict(zip(frames, times)) for key, (frames, times) in loaded.items()
    }
    common_frames = sorted(set.intersection(*(set(values) for values in frame_maps.values())))
    if not common_frames:
        raise ValueError("the input files have no frame numbers in common")

    original = frame_maps["original"]
    original_mean = statistics.fmean(original[frame] for frame in common_frames)
    rows: List[Dict[str, object]] = []

    for key, label, _ in CONFIGS:
        if key not in frame_maps:
            continue
        current = frame_maps[key]
        current_mean = statistics.fmean(current[frame] for frame in common_frames)
        row: Dict[str, object] = {
            "configuration": label,
            "frames": len(common_frames),
            "mean_frame_time_ms": current_mean,
            "mean_speedup_x": original_mean / current_mean,
            "max_speedup_in_particular_frame_x": 1.0,
        }

        if key != "original":
            if any(current[frame] <= 0.0 for frame in common_frames):
                raise ValueError(f"{label}: frame times must be positive for speedup")
            max_frame = max(common_frames, key=lambda frame: original[frame] / current[frame])
            original_time = original[max_frame]
            method_time = current[max_frame]
            row["max_speedup_in_particular_frame_x"] = original_time / method_time
        rows.append(row)
    return rows


def write_performance_summary(rows: List[Dict[str, object]], prefix: Path) -> None:
    """Write machine-readable, LaTeX, and rendered versions of the summary table."""
    prefix.parent.mkdir(parents=True, exist_ok=True)
    columns = (
        "configuration",
        "frames",
        "mean_frame_time_ms",
        "mean_speedup_x",
        "max_speedup_in_particular_frame_x",
    )

    csv_path = prefix.with_suffix(".csv")
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)

    tex_path = prefix.with_suffix(".tex")
    tex_lines = [
        r"\begin{tabular}{lrrr}",
        r"\toprule",
        r"Configuration & Mean (ms) & Mean speedup & Max speedup in particular frame \\",
        r"\midrule",
    ]
    for row in rows:
        tex_lines.append(
            f"{row['configuration']} & {float(row['mean_frame_time_ms']):.3f} & "
            f"${float(row['mean_speedup_x']):.2f}\\times$ & "
            f"${float(row['max_speedup_in_particular_frame_x']):.2f}\\times$ \\\\"
        )
    tex_lines.extend([r"\bottomrule", r"\end{tabular}"])
    tex_path.write_text("\n".join(tex_lines) + "\n", encoding="utf-8")

    display_headers = [
        "Configuration",
        "Mean frame\ntime (ms)",
        "Mean\nspeedup",
        "Max speedup in\nparticular frame",
    ]
    display_rows = []
    for row in rows:
        display_rows.append(
            [
                row["configuration"],
                f"{float(row['mean_frame_time_ms']):.3f}",
                f"{float(row['mean_speedup_x']):.2f}x",
                f"{float(row['max_speedup_in_particular_frame_x']):.2f}x",
            ]
        )

    fig, ax = plt.subplots(figsize=(8.2, 2.9), dpi=180)
    ax.axis("off")
    ax.set_title("Epic Citadel 128k performance summary", fontsize=13, pad=12)
    table = ax.table(
        cellText=display_rows,
        colLabels=display_headers,
        cellLoc="center",
        colLoc="center",
        loc="center",
        colWidths=[0.30, 0.22, 0.18, 0.30],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 1.65)
    for (row_index, column_index), cell in table.get_celld().items():
        cell.set_edgecolor("#D0D0D0")
        cell.set_linewidth(0.6)
        if row_index == 0:
            cell.set_facecolor("#303F56")
            cell.set_text_props(color="white", weight="bold")
        elif row_index % 2 == 0:
            cell.set_facecolor("#F2F5F8")
        if column_index == 0 and row_index > 0:
            cell.set_text_props(ha="left")
    fig.tight_layout()
    fig.savefig(prefix.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(prefix.with_suffix(".png"), bbox_inches="tight")
    plt.close(fig)

    print(f"Saved performance summary: {csv_path.resolve()}")
    print(f"Saved LaTeX table: {tex_path.resolve()}")
    print(f"Saved rendered table: {prefix.with_suffix('.pdf').resolve()}")


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
    parser.add_argument(
        "--summary-prefix",
        type=Path,
        default=Path("evaluation_frame_time_summary"),
        help="output prefix for summary .csv/.tex/.pdf/.png files",
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
    loaded: Dict[str, Tuple[List[int], List[float]]] = {}

    for key, label, color in CONFIGS:
        path = getattr(args, key)
        if path is None:
            continue
        frames, raw_times = load_frame_times(path)
        loaded[key] = (frames, raw_times)
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
    plt.close(fig)
    print(f"Saved figure: {args.output.resolve()}")

    summary_rows = summarize_performance(loaded)
    if summary_rows:
        write_performance_summary(summary_rows, args.summary_prefix)
    else:
        print("Skipped performance summary: --original is required as the baseline")


if __name__ == "__main__":
    main()

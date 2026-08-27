"""Create publication-ready metric curves and a local-crop image matrix."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image


METHODS = (
    ("Box normal", "#0072B2", "o"),
    ("Relief", "#E69F00", "s"),
    ("Coverage-aware relief", "#009E73", "^"),
)
METRICS = (
    ("psnr_db", "PSNR (dB) ↑"),
    ("ssim", "SSIM ↑"),
    ("lpips", "LPIPS ↓"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--metrics",
        type=Path,
        default=Path("image_quality_results/quality_metrics.csv"),
    )
    parser.add_argument("--manifest", type=Path, default=Path("image_quality_manifest.csv"))
    parser.add_argument(
        "--ground-truth",
        type=Path,
        default=Path("screenshots/ground_truth.png"),
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("image_quality_results")
    )
    parser.add_argument("--crop-x", type=int, default=285)
    parser.add_argument("--crop-y", type=int, default=752)
    parser.add_argument("--crop-radius", type=int, default=120)
    return parser.parse_args()


def read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as stream:
        return [dict(row) for row in csv.DictReader(stream)]


def save_both(fig, stem: Path) -> None:
    fig.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")


def plot_metric_curves(metric_rows: List[Dict[str, str]], output_dir: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.5), dpi=160)

    for axis, (metric_key, metric_label) in zip(axes, METRICS):
        for method, color, marker in METHODS:
            rows = [row for row in metric_rows if row["method"] == method]
            rows.sort(key=lambda row: float(row["lod"]))
            lod = [float(row["lod"]) for row in rows]
            values = [float(row[metric_key]) for row in rows]
            axis.plot(
                lod,
                values,
                color=color,
                marker=marker,
                markersize=6,
                linewidth=2,
                label=method,
            )
        axis.set_xlabel(r"LOD threshold $\tau$ (pixels)")
        axis.set_ylabel(metric_label)
        axis.set_xticks((0.5, 1.0, 1.5))
        axis.grid(True, color="#D9D9D9", linewidth=0.7, alpha=0.8)
        axis.set_axisbelow(True)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.suptitle("Epic Citadel 128k — image quality", y=0.98)
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.91),
        ncol=3,
        frameon=False,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.82))
    save_both(fig, output_dir / "quality_metrics_plot")
    plt.close(fig)


def resolve_manifest_images(
    manifest_path: Path, rows: List[Dict[str, str]]
) -> Dict[Tuple[str, float], Path]:
    images: Dict[Tuple[str, float], Path] = {}
    for row in rows:
        path = Path(row["image"])
        if not path.is_absolute():
            path = manifest_path.parent / path
        images[(row["method"], float(row["lod"]))] = path
    return images


def validate_crop(image: Image.Image, center_x: int, center_y: int, radius: int) -> None:
    left, top = center_x - radius, center_y - radius
    right, bottom = center_x + radius, center_y + radius
    if left < 0 or top < 0 or right > image.width or bottom > image.height:
        raise ValueError(
            f"crop ({left}, {top})-({right}, {bottom}) exceeds "
            f"image size {image.width}x{image.height}"
        )


def load_crop(path: Path, box: Tuple[int, int, int, int], expected_size) -> Image.Image:
    with Image.open(path) as image:
        rgb = image.convert("RGB")
        if rgb.size != expected_size:
            raise ValueError(f"{path}: expected {expected_size}, got {rgb.size}")
        return rgb.crop(box)


def plot_image_matrix(
    ground_truth_path: Path,
    manifest_path: Path,
    manifest_rows: List[Dict[str, str]],
    output_dir: Path,
    center_x: int,
    center_y: int,
    radius: int,
) -> None:
    with Image.open(ground_truth_path) as image:
        ground_truth = image.convert("RGB")
    validate_crop(ground_truth, center_x, center_y, radius)
    box = (center_x - radius, center_y - radius, center_x + radius, center_y + radius)
    ground_truth_crop = ground_truth.crop(box)
    comparison_images = resolve_manifest_images(manifest_path, manifest_rows)

    fig = plt.figure(figsize=(12.4, 7.6), dpi=160)
    grid = fig.add_gridspec(
        3,
        4,
        width_ratios=(1.45, 1, 1, 1),
        wspace=0.045,
        hspace=0.09,
    )

    ground_truth_grid = grid[:, 0].subgridspec(2, 1, height_ratios=(1.05, 1), hspace=0.18)
    overview_axis = fig.add_subplot(ground_truth_grid[0, 0])
    overview_axis.imshow(ground_truth)
    overview_axis.add_patch(
        Rectangle(
            (center_x - radius, center_y - radius),
            radius * 2,
            radius * 2,
            fill=False,
            edgecolor="#D62728",
            linewidth=2.2,
        )
    )
    overview_axis.plot(center_x, center_y, "+", color="#D62728", markersize=10)
    overview_axis.set_title(r"Ground truth overview ($\tau=0$)")
    overview_axis.set_xlabel(f"Crop centre: ({center_x}, {center_y})")
    overview_axis.set_xticks([])
    overview_axis.set_yticks([])

    ground_truth_axis = fig.add_subplot(ground_truth_grid[1, 0])
    ground_truth_axis.imshow(ground_truth_crop, interpolation="nearest")
    ground_truth_axis.set_title(r"Ground truth local ($\tau=0$)")
    ground_truth_axis.set_xticks([])
    ground_truth_axis.set_yticks([])
    for spine in ground_truth_axis.spines.values():
        spine.set_linewidth(0.8)
        spine.set_color("#777777")

    column_titles = ("Box normal", "Relief", "Coverage-aware")
    lod_values = (0.5, 1.0, 1.5)
    for row_index, lod in enumerate(lod_values):
        crops = []
        for method, _, _ in METHODS:
            path = comparison_images.get((method, lod))
            if path is None:
                raise ValueError(f"manifest is missing {method}, tau={lod}")
            crops.append(load_crop(path, box, ground_truth.size))

        for column_index, (title, crop) in enumerate(zip(column_titles, crops), start=1):
            axis = fig.add_subplot(grid[row_index, column_index])
            axis.imshow(crop, interpolation="nearest")
            if row_index == 0:
                axis.set_title(title)
            if column_index == 1:
                axis.set_ylabel(rf"$\tau={lod:.1f}$", rotation=90, labelpad=8)
            axis.set_xticks([])
            axis.set_yticks([])
            for spine in axis.spines.values():
                spine.set_linewidth(0.6)
                spine.set_color("#777777")

    fig.suptitle("Epic Citadel 128k — 240×240 local comparison", y=0.98)
    save_both(fig, output_dir / "image_quality_matrix")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    metric_rows = read_csv(args.metrics)
    manifest_rows = read_csv(args.manifest)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    plot_metric_curves(metric_rows, args.output_dir)
    plot_image_matrix(
        args.ground_truth,
        args.manifest,
        manifest_rows,
        args.output_dir,
        args.crop_x,
        args.crop_y,
        args.crop_radius,
    )
    print(f"Saved metric plot: {(args.output_dir / 'quality_metrics_plot.png').resolve()}")
    print(f"Saved image matrix: {(args.output_dir / 'image_quality_matrix.png').resolve()}")


if __name__ == "__main__":
    main()

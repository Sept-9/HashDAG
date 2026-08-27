"""Compute PSNR, SSIM, and LPIPS for HashDAG image-quality experiments."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


REQUIRED_MANIFEST_COLUMNS = ("method", "lod", "image")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare rendered images against a ground-truth RGB image."
    )
    parser.add_argument("--ground-truth", required=True, type=Path, metavar="IMAGE")
    parser.add_argument("--manifest", required=True, type=Path, metavar="CSV")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("image_quality_results"),
        help="directory for quality_metrics.csv and quality_metrics.tex",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="LPIPS inference device (default: %(default)s)",
    )
    parser.add_argument(
        "--lpips-net",
        choices=("alex", "vgg", "squeeze"),
        default="alex",
        help="LPIPS backbone (default: %(default)s)",
    )
    return parser.parse_args()


def load_rgb(path: Path) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(f"image not found: {path}")
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0


def load_manifest(path: Path) -> List[Dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"manifest not found: {path}")
    with path.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None:
            raise ValueError(f"{path}: missing header")
        missing = [name for name in REQUIRED_MANIFEST_COLUMNS if name not in reader.fieldnames]
        if missing:
            raise ValueError(f"{path}: missing columns: {', '.join(missing)}")
        rows = []
        for line_number, row in enumerate(reader, start=2):
            cleaned = {key: (value or "").strip() for key, value in row.items()}
            if not any(cleaned.values()):
                continue
            for column in REQUIRED_MANIFEST_COLUMNS:
                if not cleaned[column]:
                    raise ValueError(f"{path}:{line_number}: empty {column}")
            rows.append(cleaned)
    if not rows:
        raise ValueError(f"{path}: no comparison images listed")
    return rows


def resolve_image_path(manifest: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else manifest.parent / path


def make_lpips_model(net: str, device_name: str):
    try:
        import lpips
        import torch
    except ImportError as error:
        raise RuntimeError(
            "LPIPS dependencies are missing. Install requirements-image-quality.txt."
        ) from error

    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested, but CUDA is unavailable in PyTorch")

    torch.manual_seed(0)
    model = lpips.LPIPS(net=net, verbose=False).to(device_name).eval()
    return model, torch, device_name


def lpips_tensor(rgb: np.ndarray, torch, device_name: str):
    # LPIPS expects NCHW RGB values in [-1, 1].
    contiguous = np.ascontiguousarray(rgb.transpose(2, 0, 1))
    return torch.from_numpy(contiguous).unsqueeze(0).mul(2.0).sub(1.0).to(device_name)


def latex_escape(value: str) -> str:
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
    }
    return "".join(replacements.get(character, character) for character in value)


def metric_text(value: float) -> str:
    return "inf" if math.isinf(value) else f"{value:.6f}"


def write_csv(path: Path, rows: Iterable[Dict[str, object]]) -> None:
    fieldnames = ("method", "lod", "psnr_db", "ssim", "lpips", "image")
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    **row,
                    "psnr_db": metric_text(float(row["psnr_db"])),
                    "ssim": metric_text(float(row["ssim"])),
                    "lpips": metric_text(float(row["lpips"])),
                }
            )


def write_latex(path: Path, rows: Iterable[Dict[str, object]]) -> None:
    lines = [
        r"\begin{tabular}{llrrr}",
        r"\toprule",
        r"Method & $\tau$ & PSNR $\uparrow$ & SSIM $\uparrow$ & LPIPS $\downarrow$ \\",
        r"\midrule",
    ]
    for row in rows:
        lines.append(
            f"{latex_escape(str(row['method']))} & {latex_escape(str(row['lod']))} & "
            f"{metric_text(float(row['psnr_db']))} & {metric_text(float(row['ssim']))} & "
            f"{metric_text(float(row['lpips']))} \\\\"
        )
    lines.extend((r"\bottomrule", r"\end{tabular}"))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    ground_truth = load_rgb(args.ground_truth)
    manifest_rows = load_manifest(args.manifest)
    lpips_model, torch, device_name = make_lpips_model(args.lpips_net, args.device)
    ground_truth_tensor = lpips_tensor(ground_truth, torch, device_name)

    print(
        f"Ground truth: {args.ground_truth} "
        f"({ground_truth.shape[1]}x{ground_truth.shape[0]}); LPIPS device={device_name}"
    )

    results: List[Dict[str, object]] = []
    with torch.inference_mode():
        for row in manifest_rows:
            image_path = resolve_image_path(args.manifest, row["image"])
            comparison = load_rgb(image_path)
            if comparison.shape != ground_truth.shape:
                raise ValueError(
                    f"size mismatch: {image_path} is {comparison.shape[1]}x{comparison.shape[0]}, "
                    f"ground truth is {ground_truth.shape[1]}x{ground_truth.shape[0]}"
                )

            psnr = float(peak_signal_noise_ratio(ground_truth, comparison, data_range=1.0))
            ssim = float(
                structural_similarity(ground_truth, comparison, channel_axis=2, data_range=1.0)
            )
            lpips_value = float(
                lpips_model(ground_truth_tensor, lpips_tensor(comparison, torch, device_name))
                .reshape(-1)[0]
                .item()
            )
            result: Dict[str, object] = {
                "method": row["method"],
                "lod": row["lod"],
                "psnr_db": psnr,
                "ssim": ssim,
                "lpips": lpips_value,
                "image": str(image_path),
            }
            results.append(result)
            print(
                f"{row['method']} tau={row['lod']}: "
                f"PSNR={metric_text(psnr)} dB, SSIM={metric_text(ssim)}, "
                f"LPIPS={metric_text(lpips_value)}"
            )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / "quality_metrics.csv"
    latex_path = args.output_dir / "quality_metrics.tex"
    write_csv(csv_path, results)
    write_latex(latex_path, results)
    print(f"Saved: {csv_path.resolve()}")
    print(f"Saved: {latex_path.resolve()}")


if __name__ == "__main__":
    main()

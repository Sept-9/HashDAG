#!/usr/bin/env python3
"""Create paper-ready CSV and LaTeX tables from two HashDAG build runs."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


NUMERIC_FIELDS = (
    "geometry_build_ms",
    "color_build_ms",
    "total_build_ms",
    "geometry_logical_mb",
    "geometry_allocated_mb",
    "color_logical_mb",
    "total_logical_mb",
)


def read_run(path: Path, expected_prefilter: int) -> dict[str, float | str]:
    try:
        with path.open("r", newline="", encoding="utf-8-sig") as handle:
            rows = list(csv.DictReader(handle))
    except OSError as error:
        raise SystemExit(f"Cannot read {path}: {error}") from error

    if len(rows) != 1:
        raise SystemExit(f"Expected exactly one data row in {path}, found {len(rows)}")

    row: dict[str, float | str] = dict(rows[0])
    missing = [field for field in ("prefilter_enabled", *NUMERIC_FIELDS) if field not in row]
    if missing:
        raise SystemExit(f"Missing columns in {path}: {', '.join(missing)}")

    try:
        prefilter_enabled = int(str(row["prefilter_enabled"]))
        for field in NUMERIC_FIELDS:
            row[field] = float(str(row[field]))
    except ValueError as error:
        raise SystemExit(f"Invalid numeric value in {path}: {error}") from error

    if prefilter_enabled != expected_prefilter:
        expected = "enabled" if expected_prefilter else "disabled"
        raise SystemExit(f"{path} was not generated with prefilter {expected}")
    return row


def percentage(delta: float, baseline: float) -> float:
    return 100.0 * delta / baseline if baseline else float("nan")


def derived_rows(without: dict[str, float | str], inline: dict[str, float | str]):
    raw_rows = [
        ("Without prefilter", without),
        ("Inline prefilter", inline),
    ]
    delta = {field: float(inline[field]) - float(without[field]) for field in NUMERIC_FIELDS}
    overhead = {
        field: percentage(delta[field], float(without[field])) for field in NUMERIC_FIELDS
    }
    return raw_rows, delta, overhead


def write_csv(
    path: Path,
    raw_rows: list[tuple[str, dict[str, float | str]]],
    delta: dict[str, float],
    overhead: dict[str, float],
) -> None:
    columns = ("configuration", *NUMERIC_FIELDS)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for name, row in raw_rows:
            writer.writerow({"configuration": name, **{field: row[field] for field in NUMERIC_FIELDS}})
        writer.writerow({"configuration": "Difference", **delta})
        writer.writerow({"configuration": "Overhead (%)", **overhead})


def latex_values(row: dict[str, float | str], percent: bool = False) -> list[str]:
    if percent:
        return [f"{float(row[field]):.2f}\\%" for field in NUMERIC_FIELDS]
    return [
        f"{float(row['geometry_build_ms']) / 1000.0:.3f}",
        f"{float(row['color_build_ms']) / 1000.0:.3f}",
        f"{float(row['total_build_ms']) / 1000.0:.3f}",
        f"{float(row['geometry_logical_mb']):.2f}",
        f"{float(row['geometry_allocated_mb']):.2f}",
        f"{float(row['color_logical_mb']):.2f}",
        f"{float(row['total_logical_mb']):.2f}",
    ]


def write_latex(
    path: Path,
    raw_rows: list[tuple[str, dict[str, float | str]]],
    delta: dict[str, float],
    overhead: dict[str, float],
) -> None:
    lines = [
        r"\begin{tabular}{lrrrrrrr}",
        r"\toprule",
        r"Configuration & HashDAG (s) & Color (s) & Total (s) & Geometry (MB) & Alloc. (MB) & Color (MB) & Total (MB) \\",
        r"\midrule",
    ]
    for name, row in raw_rows:
        lines.append(name + " & " + " & ".join(latex_values(row)) + r" \\")
    lines.append(r"\midrule")
    lines.append(r"$\Delta$ & " + " & ".join(latex_values(delta)) + r" \\")
    lines.append("Overhead (\\%) & " + " & ".join(latex_values(overhead, percent=True)) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--without-prefilter", type=Path, required=True)
    parser.add_argument("--inline-prefilter", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("build_stats"))
    args = parser.parse_args()

    without = read_run(args.without_prefilter, expected_prefilter=0)
    inline = read_run(args.inline_prefilter, expected_prefilter=1)
    raw_rows, delta, overhead = derived_rows(without, inline)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / "build_evaluation_table.csv"
    tex_path = args.output_dir / "build_evaluation_table.tex"
    write_csv(csv_path, raw_rows, delta, overhead)
    write_latex(tex_path, raw_rows, delta, overhead)

    print(f"Wrote {csv_path}")
    print(f"Wrote {tex_path}")
    print(
        "Inline-prefilter overhead: "
        f"build {overhead['total_build_ms']:.2f}%, "
        f"logical storage {overhead['total_logical_mb']:.2f}%"
    )


if __name__ == "__main__":
    main()

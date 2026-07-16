"""Merge panorama depth fusion evaluation summaries into one table."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path


FIELDS = ["dataset", "method", "abs_rel", "rmse", "delta1", "num_images"]


def read_summary(path: Path) -> list[dict[str, str]]:
    if path.is_dir():
        path = path / "summary.csv"
    rows: list[dict[str, str]] = []
    with path.open(newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            rows.append({field: row.get(field, "") for field in FIELDS})
    return rows


def as_float(text: str) -> float:
    try:
        return float(text)
    except Exception:
        return float("nan")


def fmt(text: str) -> str:
    value = as_float(text)
    if math.isnan(value):
        return "NA"
    return f"{value:.4f}"


def method_rank(method: str) -> tuple[int, str]:
    direct_order = {
        "DAP-Direct-Table3": 0,
        "DAC": 1,
        "UniK3D": 2,
    }
    return (direct_order.get(method, 10), method.lower())


def main() -> int:
    parser = argparse.ArgumentParser(description="Merge panorama depth summary CSV files.")
    parser.add_argument("inputs", nargs="+", type=Path, help="summary.csv files or output directories")
    parser.add_argument("--output-dir", type=Path, default=Path("panorama_depth_eval_full"))
    parser.add_argument(
        "--datasets",
        default="",
        help="Optional comma-separated dataset filter, for example Matterport3D,Stanford2D3D.",
    )
    args = parser.parse_args()

    allowed_datasets = {
        dataset.strip()
        for dataset in args.datasets.split(",")
        if dataset.strip()
    }
    merged: dict[tuple[str, str], dict[str, str]] = {}
    for input_path in args.inputs:
        for row in read_summary(input_path):
            if allowed_datasets and row["dataset"] not in allowed_datasets:
                continue
            key = (row["dataset"], row["method"])
            merged[key] = row

    rows = sorted(merged.values(), key=lambda r: (r["dataset"], method_rank(r["method"])))
    args.output_dir.mkdir(parents=True, exist_ok=True)

    with (args.output_dir / "summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    with (args.output_dir / "table3_style.md").open("w", encoding="utf-8") as f:
        f.write("| Dataset | Method | AbsRel↓ | RMSE↓ | δ1↑ | Num Images |\n")
        f.write("|---|---|---:|---:|---:|---:|\n")
        for row in rows:
            f.write(
                f"| {row['dataset']} | {row['method']} | {fmt(row['abs_rel'])} | "
                f"{fmt(row['rmse'])} | {fmt(row['delta1'])} | {row['num_images']} |\n"
            )

    print(f"merged -> {args.output_dir / 'summary.csv'}")
    print(f"merged -> {args.output_dir / 'table3_style.md'}")
    print((args.output_dir / "table3_style.md").read_text(encoding="utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

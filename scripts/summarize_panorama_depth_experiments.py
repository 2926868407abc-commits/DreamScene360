"""Summarize panorama-depth evaluation and comparison outputs.

This is a lightweight reporting helper for the local Table 3 style experiments.
It reads run directories produced by evaluate_panorama_depth_fusion.py and
comparison directories produced by compare_panorama_depth_runs.py, then writes a
single Markdown/CSV summary.
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def to_float(value: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def fmt(value: object) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "NA"
    if not math.isfinite(number):
        return "NA"
    return f"{number:.4f}"


def parse_entry(text: str) -> tuple[str, Path]:
    if "=" not in text:
        raise ValueError(f"Expected LABEL=DIR, got {text}")
    label, path = text.split("=", 1)
    return label.strip(), Path(path.strip())


def write_csv(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def run_rows(entries: list[str]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for entry in entries:
        label, directory = parse_entry(entry)
        summary_path = directory / "summary.csv"
        if not summary_path.exists():
            print(f"[skip] missing run summary: {summary_path}")
            continue
        for row in read_csv(summary_path):
            if row.get("num_images") == "Table3":
                continue
            rows.append(
                {
                    "section": "run",
                    "experiment": label,
                    "dataset": row["dataset"],
                    "method": row["method"],
                    "num_images": row["num_images"],
                    "abs_rel": to_float(row["abs_rel"]),
                    "rmse": to_float(row["rmse"]),
                    "delta1": to_float(row["delta1"]),
                    "baseline": "",
                    "abs_rel_delta": "",
                    "rmse_delta": "",
                    "delta1_delta": "",
                    "mean_metrics_better": "",
                    "all_metrics_win_rate": "",
                }
            )
    return rows


def comparison_rows(entries: list[str]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for entry in entries:
        label, directory = parse_entry(entry)
        summary_path = directory / "comparison_summary.csv"
        if not summary_path.exists():
            print(f"[skip] missing comparison summary: {summary_path}")
            continue

        coverage_path = directory / "coverage.csv"
        coverage = {row["set"]: row["num_rows"] for row in read_csv(coverage_path)} if coverage_path.exists() else {}
        coverage_note = (
            f"common={coverage.get('common_compared', 'NA')}, "
            f"candidate_only={coverage.get('candidate_only', 'NA')}, "
            f"baseline_only={coverage.get('baseline_only', 'NA')}"
        )
        for row in read_csv(summary_path):
            rows.append(
                {
                    "section": "comparison",
                    "experiment": label,
                    "dataset": row["dataset"],
                    "method": "candidate-vs-baseline",
                    "num_images": row["num_images"],
                    "abs_rel": "",
                    "rmse": "",
                    "delta1": "",
                    "baseline": coverage_note,
                    "abs_rel_delta": to_float(row["abs_rel_delta"]),
                    "rmse_delta": to_float(row["rmse_delta"]),
                    "delta1_delta": to_float(row["delta1_delta"]),
                    "mean_metrics_better": row.get("mean_metrics_better", ""),
                    "all_metrics_win_rate": row.get("all_metrics_win_rate", ""),
                }
            )
    return rows


def markdown(rows: list[dict[str, object]]) -> str:
    lines = [
        "| Section | Experiment | Dataset | Method | N | AbsRel | RMSE | delta1 | AbsRel delta | RMSE delta | delta1 delta | Mean better | All-metric win rate | Note |",
        "|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['section']} | {row['experiment']} | {row['dataset']} | {row['method']} | "
            f"{row['num_images']} | {fmt(row['abs_rel'])} | {fmt(row['rmse'])} | {fmt(row['delta1'])} | "
            f"{fmt(row['abs_rel_delta'])} | {fmt(row['rmse_delta'])} | {fmt(row['delta1_delta'])} | "
            f"{row['mean_metrics_better']} | {row['all_metrics_win_rate']} | {row['baseline']} |"
        )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize panorama-depth experiment outputs.")
    parser.add_argument("--run", action="append", default=[], help="Run entry LABEL=DIR.")
    parser.add_argument("--comparison", action="append", default=[], help="Comparison entry LABEL=DIR.")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = run_rows(args.run) + comparison_rows(args.comparison)
    fields = [
        "section",
        "experiment",
        "dataset",
        "method",
        "num_images",
        "abs_rel",
        "rmse",
        "delta1",
        "baseline",
        "abs_rel_delta",
        "rmse_delta",
        "delta1_delta",
        "mean_metrics_better",
        "all_metrics_win_rate",
    ]
    write_csv(args.output_dir / "summary.csv", rows, fields)
    (args.output_dir / "summary.md").write_text(markdown(rows), encoding="utf-8")
    print((args.output_dir / "summary.md").read_text(encoding="utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

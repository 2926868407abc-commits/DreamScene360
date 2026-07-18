"""Compare two panorama-depth evaluation runs on the same images."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def key(row: dict[str, str]) -> tuple[str, str]:
    return row["dataset"], row["scene"]


def to_float(value: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def mean(values: list[float]) -> float:
    values = [v for v in values if math.isfinite(v)]
    if not values:
        return float("nan")
    return sum(values) / len(values)


def fmt(value: float) -> str:
    if not math.isfinite(value):
        return "NA"
    return f"{value:.4f}"


def write_csv(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def markdown(rows: list[dict[str, object]]) -> str:
    lines = [
        "| Dataset | Num Images | AbsRel delta | RMSE delta | delta1 delta | Mean metrics better | All images all metrics better |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['dataset']} | {row['num_images']} | "
            f"{fmt(float(row['abs_rel_delta']))} | {fmt(float(row['rmse_delta']))} | "
            f"{fmt(float(row['delta1_delta']))} | {row['mean_metrics_better']} | "
            f"{row['all_images_all_metrics_better']} |"
        )
    return "\n".join(lines) + "\n"


def coverage_markdown(rows: list[dict[str, object]]) -> str:
    lines = [
        "| Set | Num Rows |",
        "|---|---:|",
    ]
    for row in rows:
        lines.append(f"| {row['set']} | {row['num_rows']} |")
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare two panorama depth evaluation run directories.")
    parser.add_argument("--candidate-dir", type=Path, required=True, help="Run being tested, e.g. fusion.")
    parser.add_argument("--baseline-dir", type=Path, required=True, help="Baseline run, e.g. direct panorama.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--candidate-label", default="candidate")
    parser.add_argument("--baseline-label", default="baseline")
    args = parser.parse_args()

    candidate_rows = {key(row): row for row in read_csv(args.candidate_dir / "metrics_per_image.csv")}
    baseline_rows = {key(row): row for row in read_csv(args.baseline_dir / "metrics_per_image.csv")}
    candidate_keys = set(candidate_rows)
    baseline_keys = set(baseline_rows)
    common_keys = sorted(candidate_keys & baseline_keys)
    if not common_keys:
        raise RuntimeError("No common dataset/scene rows found.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    coverage_rows = [
        {"set": "candidate_total", "num_rows": len(candidate_keys)},
        {"set": "baseline_total", "num_rows": len(baseline_keys)},
        {"set": "common_compared", "num_rows": len(common_keys)},
        {"set": "candidate_only", "num_rows": len(candidate_keys - baseline_keys)},
        {"set": "baseline_only", "num_rows": len(baseline_keys - candidate_keys)},
    ]
    write_csv(args.output_dir / "coverage.csv", coverage_rows, ["set", "num_rows"])
    (args.output_dir / "coverage.md").write_text(coverage_markdown(coverage_rows), encoding="utf-8")

    missing_rows = []
    for dataset, scene in sorted(candidate_keys - baseline_keys):
        missing_rows.append({"side": "candidate_only", "dataset": dataset, "scene": scene})
    for dataset, scene in sorted(baseline_keys - candidate_keys):
        missing_rows.append({"side": "baseline_only", "dataset": dataset, "scene": scene})
    if missing_rows:
        write_csv(args.output_dir / "missing_rows.csv", missing_rows, ["side", "dataset", "scene"])

    per_image: list[dict[str, object]] = []
    for dataset, scene in common_keys:
        cand = candidate_rows[(dataset, scene)]
        base = baseline_rows[(dataset, scene)]
        cand_abs_rel = to_float(cand["abs_rel"])
        base_abs_rel = to_float(base["abs_rel"])
        cand_rmse = to_float(cand["rmse"])
        base_rmse = to_float(base["rmse"])
        cand_delta1 = to_float(cand["delta1"])
        base_delta1 = to_float(base["delta1"])
        abs_rel_better = cand_abs_rel < base_abs_rel
        rmse_better = cand_rmse < base_rmse
        delta1_better = cand_delta1 > base_delta1
        per_image.append(
            {
                "dataset": dataset,
                "scene": scene,
                "candidate": args.candidate_label,
                "baseline": args.baseline_label,
                "candidate_abs_rel": cand_abs_rel,
                "baseline_abs_rel": base_abs_rel,
                "abs_rel_delta": cand_abs_rel - base_abs_rel,
                "abs_rel_better": abs_rel_better,
                "candidate_rmse": cand_rmse,
                "baseline_rmse": base_rmse,
                "rmse_delta": cand_rmse - base_rmse,
                "rmse_better": rmse_better,
                "candidate_delta1": cand_delta1,
                "baseline_delta1": base_delta1,
                "delta1_delta": cand_delta1 - base_delta1,
                "delta1_better": delta1_better,
                "all_metrics_better": abs_rel_better and rmse_better and delta1_better,
            }
        )

    summary: list[dict[str, object]] = []
    for dataset in sorted({str(row["dataset"]) for row in per_image}):
        rows = [row for row in per_image if row["dataset"] == dataset]
        abs_rel_delta = mean([float(row["abs_rel_delta"]) for row in rows])
        rmse_delta = mean([float(row["rmse_delta"]) for row in rows])
        delta1_delta = mean([float(row["delta1_delta"]) for row in rows])
        summary.append(
            {
                "dataset": dataset,
                "num_images": len(rows),
                "abs_rel_delta": abs_rel_delta,
                "rmse_delta": rmse_delta,
                "delta1_delta": delta1_delta,
                "mean_metrics_better": abs_rel_delta < 0 and rmse_delta < 0 and delta1_delta > 0,
                "all_images_all_metrics_better": all(bool(row["all_metrics_better"]) for row in rows),
                "abs_rel_win_rate": mean([float(bool(row["abs_rel_better"])) for row in rows]),
                "rmse_win_rate": mean([float(bool(row["rmse_better"])) for row in rows]),
                "delta1_win_rate": mean([float(bool(row["delta1_better"])) for row in rows]),
                "all_metrics_win_rate": mean([float(bool(row["all_metrics_better"])) for row in rows]),
            }
        )

    per_image_fields = [
        "dataset",
        "scene",
        "candidate",
        "baseline",
        "candidate_abs_rel",
        "baseline_abs_rel",
        "abs_rel_delta",
        "abs_rel_better",
        "candidate_rmse",
        "baseline_rmse",
        "rmse_delta",
        "rmse_better",
        "candidate_delta1",
        "baseline_delta1",
        "delta1_delta",
        "delta1_better",
        "all_metrics_better",
    ]
    summary_fields = [
        "dataset",
        "num_images",
        "abs_rel_delta",
        "rmse_delta",
        "delta1_delta",
        "mean_metrics_better",
        "all_images_all_metrics_better",
        "abs_rel_win_rate",
        "rmse_win_rate",
        "delta1_win_rate",
        "all_metrics_win_rate",
    ]
    write_csv(args.output_dir / "comparison_per_image.csv", per_image, per_image_fields)
    write_csv(args.output_dir / "comparison_summary.csv", summary, summary_fields)
    (args.output_dir / "comparison_summary.md").write_text(markdown(summary), encoding="utf-8")
    print((args.output_dir / "coverage.md").read_text(encoding="utf-8"))
    print((args.output_dir / "comparison_summary.md").read_text(encoding="utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

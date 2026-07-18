"""Inspect Table 3 style panorama-depth evaluation outputs.

This reads the files written by scripts/evaluate_panorama_depth_fusion.py and
prints the parts that are most useful for debugging metric-depth failures:
summary metrics, direct-baseline deltas, and per-image scale diagnostics.
"""

from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


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


def median(values: list[float]) -> float:
    values = sorted(v for v in values if math.isfinite(v))
    if not values:
        return float("nan")
    mid = len(values) // 2
    if len(values) % 2:
        return values[mid]
    return (values[mid - 1] + values[mid]) * 0.5


def fmt(value: float) -> str:
    if not math.isfinite(value):
        return "NA"
    return f"{value:.4f}"


def print_summary(output_dir: Path) -> None:
    summary_path = output_dir / "summary.csv"
    if not summary_path.exists():
        print(f"[warn] missing {summary_path}")
        return

    print("[summary]")
    for row in read_csv(summary_path):
        print(
            f"  {row['dataset']} | {row['method']} | "
            f"AbsRel={row['abs_rel']} RMSE={row['rmse']} "
            f"delta1={row['delta1']} N={row['num_images']}"
        )


def print_direct_comparison(output_dir: Path) -> None:
    comparison_path = output_dir / "table3_direct_comparison.csv"
    if not comparison_path.exists():
        print(f"[warn] missing {comparison_path}")
        return

    print("[vs DAP-Direct-Table3]")
    for row in read_csv(comparison_path):
        print(
            f"  {row['dataset']} | all_better={row['all_metrics_better']} | "
            f"AbsRel_delta={float(row['abs_rel_delta']):+.4f} "
            f"RMSE_delta={float(row['rmse_delta']):+.4f} "
            f"delta1_delta={float(row['delta1_delta']):+.4f}"
        )


def print_scale_diagnostics(output_dir: Path, worst: int) -> None:
    metrics_path = output_dir / "metrics_per_image.csv"
    if not metrics_path.exists():
        print(f"[warn] missing {metrics_path}")
        return

    rows = read_csv(metrics_path)
    by_dataset: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_dataset[row["dataset"]].append(row)

    print("[scale diagnostics]")
    for dataset, dataset_rows in sorted(by_dataset.items()):
        scales = [to_float(row.get("median_scale_to_gt", "")) for row in dataset_rows]
        pred_medians = [to_float(row.get("pred_median", "")) for row in dataset_rows]
        gt_medians = [to_float(row.get("gt_median", "")) for row in dataset_rows]
        abs_rels = [to_float(row.get("abs_rel", "")) for row in dataset_rows]
        delta1s = [to_float(row.get("delta1", "")) for row in dataset_rows]
        print(
            f"  {dataset}: "
            f"mean_scale_to_gt={fmt(mean(scales))} "
            f"median_scale_to_gt={fmt(median(scales))} "
            f"mean_pred_median={fmt(mean(pred_medians))} "
            f"mean_gt_median={fmt(mean(gt_medians))} "
            f"mean_absrel={fmt(mean(abs_rels))} "
            f"mean_delta1={fmt(mean(delta1s))}"
        )
        print(f"    fixed-scale command arg: --scale {dataset}={fmt(median(scales))}")

        ranked = sorted(dataset_rows, key=lambda row: to_float(row.get("abs_rel", "")), reverse=True)
        for row in ranked[:worst]:
            print(
                f"    worst {row['scene']}: "
                f"AbsRel={fmt(to_float(row.get('abs_rel', '')))} "
                f"delta1={fmt(to_float(row.get('delta1', '')))} "
                f"scale_to_gt={fmt(to_float(row.get('median_scale_to_gt', '')))} "
                f"pred_med={fmt(to_float(row.get('pred_median', '')))} "
                f"gt_med={fmt(to_float(row.get('gt_median', '')))}"
            )


def main() -> int:
    parser = argparse.ArgumentParser(description="Inspect panorama-depth Table 3 evaluation outputs.")
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--worst", type=int, default=3)
    args = parser.parse_args()

    if not args.output_dir.exists():
        raise FileNotFoundError(args.output_dir)

    print_summary(args.output_dir)
    print_direct_comparison(args.output_dir)
    print_scale_diagnostics(args.output_dir, args.worst)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

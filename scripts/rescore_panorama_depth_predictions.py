"""Rescore saved panorama-depth predictions with fixed dataset scales.

Use this after evaluate_panorama_depth_fusion.py has been run with
--save-predictions. It does not run any depth model again. It loads the saved
*_pred.npy files, applies a fixed scale per dataset, and recomputes AbsRel,
RMSE, and delta1 against the original manifest ground truth.
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from collections import defaultdict
from statistics import median
from pathlib import Path

import numpy as np
import torch


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


sys.path.insert(0, str(repo_root()))

from scripts.evaluate_panorama_depth_fusion import (  # noqa: E402
    add_table3_baselines,
    build_direct_comparison_rows,
    comparison_markdown,
    compute_metrics,
    depth_scale_stats,
    load_depth,
    load_mask,
    markdown_table,
    resize_like,
    write_csv,
)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def mean_or_nan(values: list[float]) -> float:
    finite = [v for v in values if math.isfinite(v)]
    if not finite:
        return float("nan")
    return float(np.mean(finite))


def parse_scales(values: list[str]) -> dict[str, float]:
    scales: dict[str, float] = {}
    for item in values:
        if "=" not in item:
            raise ValueError(f"Scale must be DATASET=VALUE, got {item}")
        name, value = item.split("=", 1)
        scales[name.strip()] = float(value)
    return scales


def pred_path_for(prediction_dir: Path, dataset: str, scene: str) -> Path:
    stem = f"{dataset}_{scene}".replace("/", "_").replace(" ", "_")
    return prediction_dir / f"{stem}_pred.npy"


def load_prediction_case(row: dict[str, str], prediction_dir: Path) -> dict[str, object] | None:
    dataset = row["dataset"].strip()
    scene = row.get("scene", "").strip() or Path(row["rgb_path"]).stem
    pred_path = pred_path_for(prediction_dir, dataset, scene)
    if not pred_path.exists():
        print(f"[skip] missing prediction: {pred_path}")
        return None

    gt = load_depth(Path(row["depth_path"]).expanduser(), float(row["depth_scale"] or 1.0))
    mask_path = row.get("mask_path", "").strip()
    mask = load_mask(Path(mask_path).expanduser() if mask_path else None, tuple(gt.shape))
    pred = torch.from_numpy(np.load(pred_path).squeeze().astype(np.float32))
    pred = resize_like(pred, tuple(gt.shape))
    return {
        "dataset": dataset,
        "scene": scene,
        "prediction_path": pred_path,
        "pred": pred,
        "gt": gt,
        "mask": mask,
    }


def compute_scale_to_gt(case: dict[str, object]) -> float:
    stats = depth_scale_stats(
        case["pred"],  # type: ignore[arg-type]
        case["gt"],  # type: ignore[arg-type]
        case["mask"],  # type: ignore[arg-type]
    )
    return float(stats["median_scale_to_gt"])


def split_cases(
    cases: list[dict[str, object]],
    calibration_count_per_dataset: int,
    test_count_per_dataset: int,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    if calibration_count_per_dataset <= 0:
        return [], cases

    calibration: list[dict[str, object]] = []
    test: list[dict[str, object]] = []
    by_dataset: dict[str, list[dict[str, object]]] = defaultdict(list)
    for case in cases:
        by_dataset[str(case["dataset"])].append(case)

    for dataset_cases in by_dataset.values():
        calibration.extend(dataset_cases[:calibration_count_per_dataset])
        candidate_test = dataset_cases[calibration_count_per_dataset:]
        if test_count_per_dataset > 0:
            candidate_test = candidate_test[:test_count_per_dataset]
        test.extend(candidate_test)
    return calibration, test


def scales_from_calibration(cases: list[dict[str, object]]) -> dict[str, float]:
    by_dataset: dict[str, list[float]] = defaultdict(list)
    for case in cases:
        scale = compute_scale_to_gt(case)
        if math.isfinite(scale):
            by_dataset[str(case["dataset"])].append(scale)
    return {dataset: float(median(scales)) for dataset, scales in by_dataset.items() if scales}


def leave_one_out_scales(cases: list[dict[str, object]]) -> dict[tuple[str, str], float]:
    by_dataset: dict[str, list[tuple[str, float]]] = defaultdict(list)
    for case in cases:
        scale = compute_scale_to_gt(case)
        if math.isfinite(scale):
            by_dataset[str(case["dataset"])].append((str(case["scene"]), scale))

    scales: dict[tuple[str, str], float] = {}
    for dataset, dataset_scales in by_dataset.items():
        for scene, _ in dataset_scales:
            other_scales = [scale for other_scene, scale in dataset_scales if other_scene != scene]
            if other_scales:
                scales[(dataset, scene)] = float(median(other_scales))
    return scales


def main() -> int:
    parser = argparse.ArgumentParser(description="Rescore saved panorama-depth predictions with fixed scales.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--prediction-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--method-label", default="FixedScale")
    parser.add_argument("--scale", action="append", default=[], help="Dataset scale, for example Matterport3D=2.1")
    parser.add_argument("--default-scale", type=float, default=1.0)
    parser.add_argument(
        "--leave-one-out-scale",
        action="store_true",
        help="For each image, estimate the dataset scale from all other available images in the same dataset.",
    )
    parser.add_argument(
        "--calibration-count-per-dataset",
        type=int,
        default=0,
        help="Use the first N available predictions per dataset to estimate a fixed median scale.",
    )
    parser.add_argument(
        "--test-count-per-dataset",
        type=int,
        default=0,
        help="Evaluate at most N non-calibration predictions per dataset; 0 means all remaining predictions.",
    )
    parser.add_argument("--min-depth", type=float, default=1e-3)
    parser.add_argument("--max-depth", type=float, default=100.0)
    parser.add_argument("--include-table3-direct-baselines", action="store_true")
    args = parser.parse_args()
    if args.leave_one_out_scale and args.calibration_count_per_dataset > 0:
        raise ValueError("--leave-one-out-scale cannot be combined with --calibration-count-per-dataset.")

    scales = parse_scales(args.scale)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    cases = [
        case
        for case in (
            load_prediction_case(row, args.prediction_dir)
            for row in read_csv(args.manifest)
        )
        if case is not None
    ]
    calibration_cases, eval_cases = split_cases(
        cases,
        calibration_count_per_dataset=args.calibration_count_per_dataset,
        test_count_per_dataset=args.test_count_per_dataset,
    )
    if not eval_cases:
        raise RuntimeError(
            "No predictions left for evaluation. Reduce --calibration-count-per-dataset "
            "or check --prediction-dir."
        )
    calibration_scales = scales_from_calibration(calibration_cases)
    scales = {**calibration_scales, **scales}
    if calibration_cases:
        scale_rows = [
            {
                "dataset": dataset,
                "scale": scale,
                "source": f"first_{args.calibration_count_per_dataset}_predictions",
            }
            for dataset, scale in sorted(calibration_scales.items())
        ]
        write_csv(args.output_dir / "calibrated_scales.csv", scale_rows, ["dataset", "scale", "source"])
        print("[calibrated scales]")
        for row in scale_rows:
            print(f"  --scale {row['dataset']}={float(row['scale']):.6f}")

    loo_scales = leave_one_out_scales(eval_cases) if args.leave_one_out_scale else {}
    if args.leave_one_out_scale:
        print("[leave-one-out scales]")
        for dataset, scene in sorted(loo_scales):
            print(f"  {dataset}/{scene}: {loo_scales[(dataset, scene)]:.6f}")

    per_image_rows: list[dict[str, object]] = []
    for case in eval_cases:
        dataset = str(case["dataset"])
        scene = str(case["scene"])
        pred_path = Path(case["prediction_path"])  # type: ignore[arg-type]
        gt = case["gt"]  # type: ignore[assignment]
        mask = case["mask"]  # type: ignore[assignment]
        pred = case["pred"]  # type: ignore[assignment]
        if dataset in scales:
            scale = scales[dataset]
            scale_source = "manual_or_calibration"
        elif (dataset, scene) in loo_scales:
            scale = loo_scales[(dataset, scene)]
            scale_source = "leave_one_out"
        else:
            scale = args.default_scale
            scale_source = "default"
        pred = pred * scale

        scale_stats = depth_scale_stats(pred, gt, mask)
        metrics = compute_metrics(pred, gt, mask, args.min_depth, args.max_depth)
        per_image_rows.append(
            {
                "dataset": dataset,
                "scene": scene,
                "method": args.method_label,
                "prediction_path": str(pred_path),
                "scale": scale,
                "scale_source": scale_source,
                "abs_rel": metrics.abs_rel,
                "rmse": metrics.rmse,
                "delta1": metrics.delta1,
                "valid_pixels": metrics.valid_pixels,
                **scale_stats,
            }
        )

    per_image_fields = [
        "dataset",
        "scene",
        "method",
        "prediction_path",
        "scale",
        "scale_source",
        "abs_rel",
        "rmse",
        "delta1",
        "valid_pixels",
        "pred_median",
        "gt_median",
        "median_scale_to_gt",
        "pred_mean",
        "gt_mean",
    ]
    write_csv(args.output_dir / "metrics_per_image.csv", per_image_rows, per_image_fields)

    summary_rows: list[dict[str, object]] = []
    for dataset in sorted({str(row["dataset"]) for row in per_image_rows}):
        dataset_rows = [row for row in per_image_rows if row["dataset"] == dataset]
        summary_rows.append(
            {
                "dataset": dataset,
                "method": args.method_label,
                "abs_rel": mean_or_nan([float(row["abs_rel"]) for row in dataset_rows]),
                "rmse": mean_or_nan([float(row["rmse"]) for row in dataset_rows]),
                "delta1": mean_or_nan([float(row["delta1"]) for row in dataset_rows]),
                "num_images": len(dataset_rows),
            }
        )

    comparison = build_direct_comparison_rows(list(summary_rows))
    if comparison:
        comparison_fields = [
            "dataset",
            "method",
            "baseline_method",
            "abs_rel",
            "baseline_abs_rel",
            "abs_rel_delta",
            "abs_rel_better",
            "rmse",
            "baseline_rmse",
            "rmse_delta",
            "rmse_better",
            "delta1",
            "baseline_delta1",
            "delta1_delta",
            "delta1_better",
            "all_metrics_better",
        ]
        write_csv(args.output_dir / "table3_direct_comparison.csv", comparison, comparison_fields)
        (args.output_dir / "table3_direct_comparison.md").write_text(
            comparison_markdown(comparison),
            encoding="utf-8",
        )

    if args.include_table3_direct_baselines:
        add_table3_baselines(summary_rows)

    summary_fields = ["dataset", "method", "abs_rel", "rmse", "delta1", "num_images"]
    write_csv(args.output_dir / "summary.csv", summary_rows, summary_fields)
    (args.output_dir / "table3_style.md").write_text(markdown_table(summary_rows), encoding="utf-8")
    print((args.output_dir / "table3_style.md").read_text(encoding="utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Blend direct panorama depth with perspective-fusion panorama depth.

This script is for diagnosing and evaluating a direct-guided fusion variant:

    direct panorama prediction * calibrated_scale
    + perspective-fusion prediction * calibrated_scale

The first N samples per dataset are used to estimate dataset-level scales and,
optionally, a dataset-level blend weight. The remaining samples are evaluated
without per-image GT alignment.
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from collections import defaultdict
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
    load_rgb,
    markdown_table,
    normalize_dataset_layout,
    resize_like,
    write_csv,
)
from scripts.rescore_panorama_depth_predictions import compute_scale_to_gt  # noqa: E402


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def mean_or_nan(values: list[float]) -> float:
    finite = [v for v in values if math.isfinite(v)]
    if not finite:
        return float("nan")
    return float(np.mean(finite))


def pred_path_for(prediction_dir: Path, dataset: str, scene: str) -> Path:
    stem = f"{dataset}_{scene}".replace("/", "_").replace(" ", "_")
    return prediction_dir / f"{stem}_pred.npy"


def load_case(
    row: dict[str, str],
    direct_prediction_dir: Path,
    fusion_prediction_dir: Path,
) -> dict[str, object] | None:
    dataset = row["dataset"].strip()
    scene = row.get("scene", "").strip() or Path(row["rgb_path"]).stem
    direct_path = pred_path_for(direct_prediction_dir, dataset, scene)
    fusion_path = pred_path_for(fusion_prediction_dir, dataset, scene)
    if not direct_path.exists() or not fusion_path.exists():
        print(
            f"[skip] {dataset}/{scene}: "
            f"direct_exists={direct_path.exists()} fusion_exists={fusion_path.exists()}",
            flush=True,
        )
        return None

    gt = load_depth(Path(row["depth_path"]).expanduser(), float(row.get("depth_scale") or 1.0))
    mask_text = row.get("mask_path", "").strip()
    mask = load_mask(Path(mask_text).expanduser() if mask_text else None, tuple(gt.shape))
    if dataset == "Deep360":
        rgb = load_rgb(Path(row["rgb_path"]).expanduser())
        _, gt, mask = normalize_dataset_layout(dataset, rgb, gt, mask)

    direct = torch.from_numpy(np.load(direct_path).squeeze().astype(np.float32))
    fusion = torch.from_numpy(np.load(fusion_path).squeeze().astype(np.float32))
    direct = resize_like(direct, tuple(gt.shape))
    fusion = resize_like(fusion, tuple(gt.shape))
    return {
        "dataset": dataset,
        "scene": scene,
        "direct_path": direct_path,
        "fusion_path": fusion_path,
        "direct": direct,
        "fusion": fusion,
        "gt": gt,
        "mask": mask.bool(),
    }


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


def calibrated_scales(
    cases: list[dict[str, object]],
    key: str,
    min_depth: float,
    max_depth: float,
) -> dict[str, float]:
    by_dataset: dict[str, list[float]] = defaultdict(list)
    for case in cases:
        scale = compute_scale_to_gt(
            {
                "pred": case[key],
                "gt": case["gt"],
                "mask": case["mask"],
            },
            min_depth=min_depth,
            max_depth=max_depth,
        )
        if math.isfinite(scale):
            by_dataset[str(case["dataset"])].append(scale)
    return {dataset: float(np.median(scales)) for dataset, scales in by_dataset.items() if scales}


def blend_prediction(
    case: dict[str, object],
    direct_scale: float,
    fusion_scale: float,
    alpha: float,
) -> torch.Tensor:
    direct = case["direct"]  # type: ignore[assignment]
    fusion = case["fusion"]  # type: ignore[assignment]
    return (1.0 - alpha) * direct.float() * direct_scale + alpha * fusion.float() * fusion_scale


def evaluate_cases(
    cases: list[dict[str, object]],
    direct_scales: dict[str, float],
    fusion_scales: dict[str, float],
    alphas: dict[str, float],
    min_depth: float,
    max_depth: float,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for case in cases:
        dataset = str(case["dataset"])
        scene = str(case["scene"])
        direct_scale = direct_scales.get(dataset, 1.0)
        fusion_scale = fusion_scales.get(dataset, 1.0)
        alpha = alphas.get(dataset, 0.5)
        pred = blend_prediction(case, direct_scale, fusion_scale, alpha)
        gt = case["gt"]  # type: ignore[assignment]
        mask = case["mask"]  # type: ignore[assignment]
        metrics = compute_metrics(pred, gt, mask, min_depth, max_depth)
        scale_stats = depth_scale_stats(pred, gt, mask)
        rows.append(
            {
                "dataset": dataset,
                "scene": scene,
                "method": "",
                "direct_prediction_path": str(case["direct_path"]),
                "fusion_prediction_path": str(case["fusion_path"]),
                "direct_scale": direct_scale,
                "fusion_scale": fusion_scale,
                "alpha": alpha,
                "abs_rel": metrics.abs_rel,
                "rmse": metrics.rmse,
                "delta1": metrics.delta1,
                "valid_pixels": metrics.valid_pixels,
                **scale_stats,
            }
        )
    return rows


def metric_objective(rows: list[dict[str, object]], objective: str) -> float:
    abs_rel = mean_or_nan([float(row["abs_rel"]) for row in rows])
    rmse = mean_or_nan([float(row["rmse"]) for row in rows])
    delta1 = mean_or_nan([float(row["delta1"]) for row in rows])
    if objective == "abs_rel":
        return abs_rel
    if objective == "rmse":
        return rmse
    if objective == "delta1":
        return -delta1
    if objective == "balanced":
        # Roughly balance the three metrics without requiring dataset-specific
        # tuning constants. AbsRel and RMSE are minimized; delta1 is maximized.
        return abs_rel + rmse - delta1
    raise ValueError(f"Unknown blend objective: {objective}")


def choose_alphas(
    calibration_cases: list[dict[str, object]],
    direct_scales: dict[str, float],
    fusion_scales: dict[str, float],
    alpha_grid: list[float],
    objective: str,
    min_depth: float,
    max_depth: float,
) -> dict[str, float]:
    by_dataset: dict[str, list[dict[str, object]]] = defaultdict(list)
    for case in calibration_cases:
        by_dataset[str(case["dataset"])].append(case)

    selected: dict[str, float] = {}
    for dataset, dataset_cases in by_dataset.items():
        best_alpha = alpha_grid[0]
        best_score = float("inf")
        for alpha in alpha_grid:
            rows = evaluate_cases(
                dataset_cases,
                direct_scales,
                fusion_scales,
                {dataset: alpha},
                min_depth=min_depth,
                max_depth=max_depth,
            )
            score = metric_objective(rows, objective)
            if score < best_score:
                best_score = score
                best_alpha = alpha
        selected[dataset] = best_alpha
    return selected


def parse_alpha_values(values: list[str]) -> dict[str, float]:
    alphas: dict[str, float] = {}
    for item in values:
        if "=" not in item:
            raise ValueError(f"Alpha must be DATASET=VALUE, got {item}")
        dataset, value = item.split("=", 1)
        alphas[dataset.strip()] = float(value)
    return alphas


def summary_rows(per_image_rows: list[dict[str, object]], method_label: str) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for dataset in sorted({str(row["dataset"]) for row in per_image_rows}):
        dataset_rows = [row for row in per_image_rows if row["dataset"] == dataset]
        rows.append(
            {
                "dataset": dataset,
                "method": method_label,
                "abs_rel": mean_or_nan([float(row["abs_rel"]) for row in dataset_rows]),
                "rmse": mean_or_nan([float(row["rmse"]) for row in dataset_rows]),
                "delta1": mean_or_nan([float(row["delta1"]) for row in dataset_rows]),
                "num_images": len(dataset_rows),
            }
        )
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description="Blend direct and perspective-fusion panorama depth predictions.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--direct-prediction-dir", type=Path, required=True)
    parser.add_argument("--fusion-prediction-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--method-label", default="DirectGuidedPerspectiveFusion")
    parser.add_argument("--calibration-count-per-dataset", type=int, default=10)
    parser.add_argument("--test-count-per-dataset", type=int, default=0)
    parser.add_argument("--alpha", action="append", default=[], help="Manual dataset alpha, for example Stanford2D3D=0.4")
    parser.add_argument("--auto-alpha", action="store_true", help="Pick alpha from calibration samples.")
    parser.add_argument("--alpha-min", type=float, default=0.0)
    parser.add_argument("--alpha-max", type=float, default=1.0)
    parser.add_argument("--alpha-step", type=float, default=0.05)
    parser.add_argument("--alpha-objective", choices=["balanced", "abs_rel", "rmse", "delta1"], default="balanced")
    parser.add_argument("--min-depth", type=float, default=1e-3)
    parser.add_argument("--max-depth", type=float, default=100.0)
    parser.add_argument("--include-table3-direct-baselines", action="store_true")
    args = parser.parse_args()

    if args.alpha_step <= 0:
        raise ValueError("--alpha-step must be positive.")
    if args.alpha_min > args.alpha_max:
        raise ValueError("--alpha-min cannot be greater than --alpha-max.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    cases = [
        case
        for case in (
            load_case(row, args.direct_prediction_dir, args.fusion_prediction_dir)
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
        raise RuntimeError("No evaluation cases left after calibration split.")

    direct_scales = calibrated_scales(calibration_cases, "direct", args.min_depth, args.max_depth)
    fusion_scales = calibrated_scales(calibration_cases, "fusion", args.min_depth, args.max_depth)
    manual_alphas = parse_alpha_values(args.alpha)

    alpha_grid = [
        round(args.alpha_min + i * args.alpha_step, 10)
        for i in range(int(round((args.alpha_max - args.alpha_min) / args.alpha_step)) + 1)
    ]
    auto_alphas = (
        choose_alphas(
            calibration_cases,
            direct_scales,
            fusion_scales,
            alpha_grid,
            args.alpha_objective,
            args.min_depth,
            args.max_depth,
        )
        if args.auto_alpha
        else {}
    )
    datasets = sorted({str(case["dataset"]) for case in cases})
    alphas = {dataset: 0.5 for dataset in datasets}
    alphas.update(auto_alphas)
    alphas.update(manual_alphas)

    parameter_rows = [
        {
            "dataset": dataset,
            "direct_scale": direct_scales.get(dataset, 1.0),
            "fusion_scale": fusion_scales.get(dataset, 1.0),
            "alpha": alphas.get(dataset, 0.5),
            "alpha_source": "manual" if dataset in manual_alphas else ("auto" if dataset in auto_alphas else "default"),
            "calibration_count": args.calibration_count_per_dataset,
        }
        for dataset in datasets
    ]
    write_csv(
        args.output_dir / "calibrated_parameters.csv",
        parameter_rows,
        ["dataset", "direct_scale", "fusion_scale", "alpha", "alpha_source", "calibration_count"],
    )
    print("[calibrated parameters]")
    for row in parameter_rows:
        print(
            f"  {row['dataset']}: direct_scale={float(row['direct_scale']):.6f} "
            f"fusion_scale={float(row['fusion_scale']):.6f} "
            f"alpha={float(row['alpha']):.4f} ({row['alpha_source']})",
            flush=True,
        )

    per_image_rows = evaluate_cases(
        eval_cases,
        direct_scales,
        fusion_scales,
        alphas,
        min_depth=args.min_depth,
        max_depth=args.max_depth,
    )
    for row in per_image_rows:
        row["method"] = args.method_label

    per_image_fields = [
        "dataset",
        "scene",
        "method",
        "direct_prediction_path",
        "fusion_prediction_path",
        "direct_scale",
        "fusion_scale",
        "alpha",
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

    summaries = summary_rows(per_image_rows, args.method_label)
    comparison = build_direct_comparison_rows(summaries)
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
        add_table3_baselines(summaries)
    write_csv(args.output_dir / "summary.csv", summaries, ["dataset", "method", "abs_rel", "rmse", "delta1", "num_images"])
    (args.output_dir / "table3_style.md").write_text(markdown_table(summaries), encoding="utf-8")
    print((args.output_dir / "table3_style.md").read_text(encoding="utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

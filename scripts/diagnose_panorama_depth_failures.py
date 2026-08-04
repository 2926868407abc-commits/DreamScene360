#!/usr/bin/env python3
"""Diagnose why perspective monocular depth fusion fails on panorama metrics.

This script runs a small, same-sample diagnostic pass and writes:
  - global_metrics.csv: direct panorama and perspective fusion metrics with/without median align.
  - per_view_metrics.csv: per perspective-view scale and center/edge errors.
  - overlap_alignment.csv: progressive overlap-alignment scale statistics.
  - diagnosis.md: compact issue-oriented summary.
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path
from statistics import mean, median, pstdev

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from evaluate_panorama_depth_fusion import (  # noqa: E402
    align_prediction,
    bilinear_splat,
    build_predictor,
    canonical_method,
    compute_metrics,
    depth_scale_stats,
    dirs_to_pano_xy,
    dirs_to_sample_grid,
    filter_manifest_items,
    load_depth,
    load_mask,
    load_rgb,
    normalize_dataset_layout,
    perspective_dirs,
    predict_direct_panorama_depth,
    read_manifest,
    resize_like,
    run_predictor_batch,
    sample_pano_map,
    view_angles,
)


def finite_values(values: list[float]) -> list[float]:
    return [float(v) for v in values if math.isfinite(float(v))]


def mean_or_nan(values: list[float]) -> float:
    vals = finite_values(values)
    return float(mean(vals)) if vals else float("nan")


def median_or_nan(values: list[float]) -> float:
    vals = finite_values(values)
    return float(median(vals)) if vals else float("nan")


def percentile_or_nan(values: list[float], q: float) -> float:
    vals = finite_values(values)
    return float(np.percentile(vals, q)) if vals else float("nan")


def std_or_nan(values: list[float]) -> float:
    vals = finite_values(values)
    return float(pstdev(vals)) if len(vals) > 1 else float("nan")


def format_num(value: object) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "NA"
    if not math.isfinite(number):
        return "NA"
    return f"{number:.4f}"


def write_csv(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def valid_depth_mask(
    pred: torch.Tensor,
    gt: torch.Tensor,
    mask: torch.Tensor,
    min_depth: float,
    max_depth: float,
) -> torch.Tensor:
    return (
        mask.bool()
        & torch.isfinite(pred)
        & torch.isfinite(gt)
        & (pred > min_depth)
        & (gt > min_depth)
        & (gt < max_depth)
    )


def scale_stats_for_mask(pred: torch.Tensor, gt: torch.Tensor, mask: torch.Tensor) -> dict[str, float]:
    valid = mask.bool() & torch.isfinite(pred) & torch.isfinite(gt) & (pred > 1e-6) & (gt > 1e-6)
    if not bool(valid.any()):
        return {"pred_median": float("nan"), "gt_median": float("nan"), "median_scale_to_gt": float("nan")}
    p = pred[valid]
    g = gt[valid]
    pred_median = p.median().clamp_min(1e-6)
    gt_median = g.median()
    return {
        "pred_median": float(pred_median.item()),
        "gt_median": float(gt_median.item()),
        "median_scale_to_gt": float((gt_median / pred_median).item()),
    }


def metric_row(
    dataset: str,
    scene: str,
    mode: str,
    pred: torch.Tensor,
    gt: torch.Tensor,
    mask: torch.Tensor,
    align_mode: str,
    min_depth: float,
    max_depth: float,
) -> dict[str, object]:
    stats = depth_scale_stats(pred, gt, mask)
    aligned = align_prediction(pred, gt, mask, align_mode)
    metrics = compute_metrics(aligned, gt, mask, min_depth, max_depth)
    return {
        "dataset": dataset,
        "scene": scene,
        "mode": mode,
        "align": align_mode,
        "abs_rel": metrics.abs_rel,
        "rmse": metrics.rmse,
        "delta1": metrics.delta1,
        "valid_pixels": metrics.valid_pixels,
        "pred_median": stats["pred_median"],
        "gt_median": stats["gt_median"],
        "median_scale_to_gt": stats["median_scale_to_gt"],
    }


def summarize_global(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    out = []
    keys = sorted({(str(r["dataset"]), str(r["mode"]), str(r["align"])) for r in rows})
    for dataset, mode, align in keys:
        selected = [r for r in rows if r["dataset"] == dataset and r["mode"] == mode and r["align"] == align]
        out.append(
            {
                "dataset": dataset,
                "mode": mode,
                "align": align,
                "num_images": len(selected),
                "abs_rel": mean_or_nan([float(r["abs_rel"]) for r in selected]),
                "rmse": mean_or_nan([float(r["rmse"]) for r in selected]),
                "delta1": mean_or_nan([float(r["delta1"]) for r in selected]),
                "median_scale_to_gt": median_or_nan([float(r["median_scale_to_gt"]) for r in selected]),
                "scale_p10": percentile_or_nan([float(r["median_scale_to_gt"]) for r in selected], 10),
                "scale_p90": percentile_or_nan([float(r["median_scale_to_gt"]) for r in selected], 90),
            }
        )
    return out


def summarize_per_view(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    out = []
    for dataset in sorted({str(r["dataset"]) for r in rows}):
        selected = [r for r in rows if r["dataset"] == dataset]
        scales = [float(r["median_scale_to_gt"]) for r in selected]
        center = mean_or_nan([float(r["center_abs_rel"]) for r in selected])
        edge = mean_or_nan([float(r["edge_abs_rel"]) for r in selected])
        scale_med = median_or_nan(scales)
        scale_std = std_or_nan(scales)
        out.append(
            {
                "dataset": dataset,
                "num_views": len(selected),
                "scale_median": scale_med,
                "scale_mean": mean_or_nan(scales),
                "scale_std": scale_std,
                "scale_cv": scale_std / scale_med if math.isfinite(scale_std) and scale_med > 1e-6 else float("nan"),
                "scale_p10": percentile_or_nan(scales, 10),
                "scale_p90": percentile_or_nan(scales, 90),
                "abs_rel": mean_or_nan([float(r["abs_rel"]) for r in selected]),
                "center_abs_rel": center,
                "edge_abs_rel": edge,
                "edge_center_ratio": edge / center if math.isfinite(center) and center > 1e-6 else float("nan"),
            }
        )
    return out


def summarize_overlap(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    out = []
    for dataset in sorted({str(r["dataset"]) for r in rows}):
        selected = [r for r in rows if r["dataset"] == dataset]
        aligned = [r for r in selected if int(r["overlap_pixels"]) > 0 and math.isfinite(float(r["raw_scale"]))]
        scales = [float(r["raw_scale"]) for r in aligned]
        clamped = [r for r in aligned if str(r["was_clamped"]) == "True"]
        out.append(
            {
                "dataset": dataset,
                "num_views": len(selected),
                "num_aligned": len(aligned),
                "mean_overlap_pixels": mean_or_nan([float(r["overlap_pixels"]) for r in selected]),
                "scale_median": median_or_nan(scales),
                "scale_p10": percentile_or_nan(scales, 10),
                "scale_p90": percentile_or_nan(scales, 90),
                "scale_cv": (std_or_nan(scales) / median_or_nan(scales))
                if math.isfinite(std_or_nan(scales)) and median_or_nan(scales) > 1e-6
                else float("nan"),
                "clamped_fraction": len(clamped) / len(aligned) if aligned else float("nan"),
            }
        )
    return out


def markdown_summary(
    global_summary: list[dict[str, object]],
    per_view_summary: list[dict[str, object]],
    overlap_summary: list[dict[str, object]],
) -> str:
    lines = ["# Panorama Depth Failure Diagnosis", ""]
    lines += [
        "## 1. Global Metric Scale",
        "",
        "| Dataset | Mode | Align | N | AbsRel | RMSE | d1 | Median Scale-to-GT | P10 | P90 |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in global_summary:
        lines.append(
            f"| {r['dataset']} | {r['mode']} | {r['align']} | {r['num_images']} | "
            f"{format_num(r['abs_rel'])} | {format_num(r['rmse'])} | {format_num(r['delta1'])} | "
            f"{format_num(r['median_scale_to_gt'])} | {format_num(r['scale_p10'])} | {format_num(r['scale_p90'])} |"
        )

    lines += [
        "",
        "## 2. Per-View Scale Drift and 4. Edge Projection Error",
        "",
        "| Dataset | Views | Scale Median | Scale CV | P10 | P90 | AbsRel | Center AbsRel | Edge AbsRel | Edge/Center |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in per_view_summary:
        lines.append(
            f"| {r['dataset']} | {r['num_views']} | {format_num(r['scale_median'])} | "
            f"{format_num(r['scale_cv'])} | {format_num(r['scale_p10'])} | {format_num(r['scale_p90'])} | "
            f"{format_num(r['abs_rel'])} | {format_num(r['center_abs_rel'])} | "
            f"{format_num(r['edge_abs_rel'])} | {format_num(r['edge_center_ratio'])} |"
        )

    lines += [
        "",
        "## 3. Overlap Alignment Reliability",
        "",
        "| Dataset | Views | Aligned | Mean Overlap Pixels | Scale Median | P10 | P90 | Scale CV | Clamped Fraction |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in overlap_summary:
        lines.append(
            f"| {r['dataset']} | {r['num_views']} | {r['num_aligned']} | {format_num(r['mean_overlap_pixels'])} | "
            f"{format_num(r['scale_median'])} | {format_num(r['scale_p10'])} | {format_num(r['scale_p90'])} | "
            f"{format_num(r['scale_cv'])} | {format_num(r['clamped_fraction'])} |"
        )

    lines += ["", "## Reading Guide", ""]
    lines.append("- If median scale-to-GT is far from 1, metric scale is wrong.")
    lines.append("- If per-view scale CV or P90/P10 is large, each perspective view has inconsistent scale.")
    lines.append("- If many overlap scales are clamped or overlap pixels are low, overlap alignment is unreliable.")
    lines.append("- If edge/center is much greater than 1, perspective edge projection is amplifying errors.")
    lines.append("- If median-align rows improve a lot over none-align rows, the metric mainly fails because of scale.")
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description="Diagnose panorama depth scale/fusion failure modes.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--method", default="depth_anything3")
    parser.add_argument("--datasets", default="")
    parser.add_argument("--max-items", type=int, default=0)
    parser.add_argument("--max-per-dataset", type=int, default=2)
    parser.add_argument("--yaw-count", type=int, default=8)
    parser.add_argument("--pitch-degrees", default="-45,0,45")
    parser.add_argument("--max-views", type=int, default=0)
    parser.add_argument("--view-size", type=int, default=384)
    parser.add_argument("--fov", type=float, default=90.0)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--center-weight-power", type=float, default=2.0)
    parser.add_argument("--center-threshold", type=float, default=0.8)
    parser.add_argument("--edge-threshold", type=float, default=0.5)
    parser.add_argument("--overlap-min-pixels", type=int, default=2048)
    parser.add_argument("--overlap-scale-min", type=float, default=0.25)
    parser.add_argument("--overlap-scale-max", type=float, default=4.0)
    parser.add_argument("--min-depth", type=float, default=1e-3)
    parser.add_argument("--max-depth", type=float, default=100.0)
    parser.add_argument("--depth-anything3-model", default="depth-anything/DA3-LARGE-1.1")
    parser.add_argument("--depth-anything3-command", default="")
    parser.add_argument("--dap-root", default="")
    parser.add_argument("--dap-model-path", default="")
    parser.add_argument("--dap-command", default="")
    parser.add_argument("--g2vlm-root", default="")
    parser.add_argument("--g2vlm-model-path", default="")
    parser.add_argument("--vggt-root", default="")
    parser.add_argument("--vggt-model-path", default="facebook/VGGT-1B")
    parser.add_argument("--vggt-chunk-size", type=int, default=8)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the configured depth predictors.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    method = canonical_method(args.method)
    predictor = build_predictor(method, args)
    device = torch.device("cuda")
    pitch_degrees = [float(x) for x in args.pitch_degrees.split(",") if x.strip()]
    angles = view_angles(args.yaw_count, pitch_degrees)
    if args.max_views > 0:
        angles = angles[: args.max_views]

    items = filter_manifest_items(
        read_manifest(args.manifest),
        datasets=args.datasets,
        max_items=args.max_items,
        max_per_dataset=args.max_per_dataset,
    )
    if not items:
        raise RuntimeError("No manifest items selected.")

    global_rows: list[dict[str, object]] = []
    per_view_rows: list[dict[str, object]] = []
    overlap_rows: list[dict[str, object]] = []

    for item_index, item in enumerate(items):
        print(f"[{item_index + 1}/{len(items)}] {item.dataset}/{item.scene}", flush=True)
        rgb = load_rgb(item.rgb_path).to(device)
        gt = load_depth(item.depth_path, item.depth_scale).to(device)
        mask = load_mask(item.mask_path, tuple(gt.shape)).to(device)
        rgb, gt, mask = normalize_dataset_layout(item.dataset, rgb, gt, mask)
        _, pano_h, pano_w = rgb.shape

        direct = resize_like(predict_direct_panorama_depth(rgb, predictor), tuple(gt.shape))
        global_rows.append(
            metric_row(item.dataset, item.scene, "direct_panorama", direct, gt, mask, "none", args.min_depth, args.max_depth)
        )
        global_rows.append(
            metric_row(item.dataset, item.scene, "direct_panorama", direct, gt, mask, "median", args.min_depth, args.max_depth)
        )

        value_sum = torch.zeros((pano_h, pano_w), dtype=torch.float32, device=device)
        weight_sum = torch.zeros((pano_h, pano_w), dtype=torch.float32, device=device)
        pano_batch = rgb[None]

        for batch_start in range(0, len(angles), args.batch_size):
            batch_angles = angles[batch_start : batch_start + args.batch_size]
            dirs_batch = []
            ratios_batch = []
            for yaw, pitch in batch_angles:
                dirs, ray_ratio = perspective_dirs(args.view_size, args.fov, yaw, pitch, device)
                dirs_batch.append(dirs)
                ratios_batch.append(ray_ratio)
            dirs_t = torch.stack(dirs_batch, dim=0)
            ratios_t = torch.stack(ratios_batch, dim=0)
            grid = dirs_to_sample_grid(dirs_t, pano_h, pano_w)
            views = F.grid_sample(
                pano_batch.expand(len(batch_angles), -1, -1, -1),
                grid,
                mode="bilinear",
                padding_mode="border",
                align_corners=True,
            )
            gt_views = F.grid_sample(
                gt[None, None].float().expand(len(batch_angles), -1, -1, -1),
                grid,
                mode="bilinear",
                padding_mode="border",
                align_corners=True,
            )[:, 0]
            mask_views = F.grid_sample(
                mask[None, None].float().expand(len(batch_angles), -1, -1, -1),
                grid,
                mode="nearest",
                padding_mode="border",
                align_corners=True,
            )[:, 0] > 0.5

            pred_depth = run_predictor_batch(predictor, views, batch_size=args.batch_size, fov_degrees=args.fov).to(device)
            radial_depth = pred_depth.clamp_min(1e-6) * ratios_t
            center_weight = (1.0 / ratios_t).clamp(0.0, 1.0).pow(args.center_weight_power)
            x, y = dirs_to_pano_xy(dirs_t, pano_h, pano_w)

            for i, (yaw, pitch) in enumerate(batch_angles):
                view_index = batch_start + i
                valid = valid_depth_mask(radial_depth[i], gt_views[i], mask_views[i], args.min_depth, args.max_depth)
                center_valid = valid & (center_weight[i] >= args.center_threshold)
                edge_valid = valid & (center_weight[i] <= args.edge_threshold)
                view_metrics = compute_metrics(radial_depth[i], gt_views[i], valid, args.min_depth, args.max_depth)
                center_abs_rel = (
                    compute_metrics(radial_depth[i], gt_views[i], center_valid, args.min_depth, args.max_depth).abs_rel
                    if int(center_valid.sum().item()) > 0
                    else float("nan")
                )
                edge_abs_rel = (
                    compute_metrics(radial_depth[i], gt_views[i], edge_valid, args.min_depth, args.max_depth).abs_rel
                    if int(edge_valid.sum().item()) > 0
                    else float("nan")
                )
                view_stats = scale_stats_for_mask(radial_depth[i], gt_views[i], valid)
                per_view_rows.append(
                    {
                        "dataset": item.dataset,
                        "scene": item.scene,
                        "view_index": view_index,
                        "yaw": yaw,
                        "pitch": pitch,
                        "valid_pixels": view_metrics.valid_pixels,
                        "abs_rel": view_metrics.abs_rel,
                        "rmse": view_metrics.rmse,
                        "delta1": view_metrics.delta1,
                        "center_abs_rel": center_abs_rel,
                        "edge_abs_rel": edge_abs_rel,
                        **view_stats,
                    }
                )

                ref_scale = float("nan")
                clamped_scale = float("nan")
                overlap_pixels = 0
                was_clamped = False
                values = radial_depth[i]
                if bool((weight_sum > 1e-8).any()):
                    fused_depth = value_sum / weight_sum.clamp_min(1e-8)
                    ref_depth = sample_pano_map(fused_depth, x[i], y[i])
                    ref_weight = sample_pano_map(weight_sum, x[i], y[i])
                    overlap_valid = (
                        (ref_weight > 1e-6)
                        & torch.isfinite(ref_depth)
                        & torch.isfinite(values)
                        & (ref_depth > 0)
                        & (values > 0)
                        & (center_weight[i] > 0.05)
                    )
                    overlap_pixels = int(overlap_valid.sum().item())
                    if overlap_pixels >= args.overlap_min_pixels:
                        ratios = ref_depth[overlap_valid] / values[overlap_valid].clamp_min(1e-6)
                        ref_scale = float(ratios.median().item())
                        clamped = min(max(ref_scale, args.overlap_scale_min), args.overlap_scale_max)
                        clamped_scale = float(clamped)
                        was_clamped = abs(clamped - ref_scale) > 1e-6
                        values = values * clamped
                overlap_rows.append(
                    {
                        "dataset": item.dataset,
                        "scene": item.scene,
                        "view_index": view_index,
                        "yaw": yaw,
                        "pitch": pitch,
                        "overlap_pixels": overlap_pixels,
                        "raw_scale": ref_scale,
                        "clamped_scale": clamped_scale,
                        "was_clamped": was_clamped,
                    }
                )
                bilinear_splat(value_sum, weight_sum, x[i], y[i], values, center_weight[i])

        fused = value_sum / weight_sum.clamp_min(1e-8)
        covered = weight_sum > 1e-8
        if not bool(covered.all()):
            fallback = fused[covered].median() if bool(covered.any()) else torch.tensor(1.0, device=device)
            fused = torch.where(covered, fused, fallback)
        global_rows.append(
            metric_row(item.dataset, item.scene, "perspective_fusion_progressive", fused, gt, mask, "none", args.min_depth, args.max_depth)
        )
        global_rows.append(
            metric_row(item.dataset, item.scene, "perspective_fusion_progressive", fused, gt, mask, "median", args.min_depth, args.max_depth)
        )

    global_fields = [
        "dataset",
        "scene",
        "mode",
        "align",
        "abs_rel",
        "rmse",
        "delta1",
        "valid_pixels",
        "pred_median",
        "gt_median",
        "median_scale_to_gt",
    ]
    per_view_fields = [
        "dataset",
        "scene",
        "view_index",
        "yaw",
        "pitch",
        "valid_pixels",
        "abs_rel",
        "rmse",
        "delta1",
        "center_abs_rel",
        "edge_abs_rel",
        "pred_median",
        "gt_median",
        "median_scale_to_gt",
    ]
    overlap_fields = [
        "dataset",
        "scene",
        "view_index",
        "yaw",
        "pitch",
        "overlap_pixels",
        "raw_scale",
        "clamped_scale",
        "was_clamped",
    ]
    write_csv(args.output_dir / "global_metrics.csv", global_rows, global_fields)
    write_csv(args.output_dir / "per_view_metrics.csv", per_view_rows, per_view_fields)
    write_csv(args.output_dir / "overlap_alignment.csv", overlap_rows, overlap_fields)

    global_summary = summarize_global(global_rows)
    per_view_summary = summarize_per_view(per_view_rows)
    overlap_summary = summarize_overlap(overlap_rows)
    write_csv(
        args.output_dir / "global_summary.csv",
        global_summary,
        ["dataset", "mode", "align", "num_images", "abs_rel", "rmse", "delta1", "median_scale_to_gt", "scale_p10", "scale_p90"],
    )
    write_csv(
        args.output_dir / "per_view_summary.csv",
        per_view_summary,
        [
            "dataset",
            "num_views",
            "scale_median",
            "scale_mean",
            "scale_std",
            "scale_cv",
            "scale_p10",
            "scale_p90",
            "abs_rel",
            "center_abs_rel",
            "edge_abs_rel",
            "edge_center_ratio",
        ],
    )
    write_csv(
        args.output_dir / "overlap_summary.csv",
        overlap_summary,
        [
            "dataset",
            "num_views",
            "num_aligned",
            "mean_overlap_pixels",
            "scale_median",
            "scale_p10",
            "scale_p90",
            "scale_cv",
            "clamped_fraction",
        ],
    )
    (args.output_dir / "diagnosis.md").write_text(
        markdown_summary(global_summary, per_view_summary, overlap_summary),
        encoding="utf-8",
    )
    print(f"[done] {args.output_dir / 'diagnosis.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

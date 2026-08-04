"""Create report-ready figures for panorama-depth experiments.

This script is intentionally lightweight: it only reads saved predictions,
manifests, and metric CSV files. It does not run any depth model.
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.evaluate_panorama_depth_fusion import (  # noqa: E402
    load_depth,
    load_mask,
    load_rgb,
    normalize_dataset_layout,
    resize_like,
)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def find_manifest_row(rows: list[dict[str, str]], dataset: str, scene: str) -> dict[str, str]:
    for row in rows:
        row_dataset = row.get("dataset", "").strip()
        row_scene = row.get("scene", "").strip() or Path(row.get("rgb_path", "")).stem
        if row_dataset == dataset and row_scene == scene:
            return row
    raise KeyError(f"Cannot find manifest row for {dataset}/{scene}")


def common_scene(
    direct_rows: list[dict[str, str]],
    ours_rows: list[dict[str, str]],
    dataset: str,
    requested_scene: str | None,
) -> str:
    direct_scenes = {
        row["scene"]
        for row in direct_rows
        if row.get("dataset", "").strip() == dataset and row.get("scene", "").strip()
    }
    ours_scenes = {
        row["scene"]
        for row in ours_rows
        if row.get("dataset", "").strip() == dataset and row.get("scene", "").strip()
    }
    scenes = sorted(direct_scenes & ours_scenes)
    if requested_scene:
        if requested_scene not in scenes:
            raise KeyError(f"{dataset}/{requested_scene} is not common to direct and ours metrics")
        return requested_scene
    if not scenes:
        raise RuntimeError(f"No common scene found for dataset {dataset}")
    return scenes[0]


def row_for_scene(rows: list[dict[str, str]], dataset: str, scene: str) -> dict[str, str]:
    for row in rows:
        if row.get("dataset", "").strip() == dataset and row.get("scene", "").strip() == scene:
            return row
    raise KeyError(f"Cannot find metrics row for {dataset}/{scene}")


def load_prediction_from_row(row: dict[str, str]) -> np.ndarray:
    if row.get("prediction_path"):
        scale = float(row.get("scale") or 1.0)
        pred = np.load(Path(row["prediction_path"]).expanduser()).squeeze().astype(np.float32)
        return pred * scale

    if row.get("direct_prediction_path") and row.get("fusion_prediction_path"):
        direct = np.load(Path(row["direct_prediction_path"]).expanduser()).squeeze().astype(np.float32)
        fusion = np.load(Path(row["fusion_prediction_path"]).expanduser()).squeeze().astype(np.float32)
        direct_scale = float(row.get("direct_scale") or 1.0)
        fusion_scale = float(row.get("fusion_scale") or 1.0)
        alpha = float(row.get("alpha") or 0.5)
        return (1.0 - alpha) * direct * direct_scale + alpha * fusion * fusion_scale

    raise KeyError(
        "Metrics row must contain either prediction_path or "
        "direct_prediction_path/fusion_prediction_path"
    )


def tensor_to_numpy(tensor) -> np.ndarray:
    return tensor.detach().float().cpu().numpy()


def resize_prediction_like_gt(pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    import torch

    pred_t = torch.from_numpy(pred.squeeze().astype(np.float32))
    gt_shape = tuple(gt.shape)
    pred_t = resize_like(pred_t, gt_shape)
    return tensor_to_numpy(pred_t)


def percentile_range(values: np.ndarray, mask: np.ndarray, lo: float = 2, hi: float = 98) -> tuple[float, float]:
    valid = mask & np.isfinite(values)
    if not np.any(valid):
        return 0.0, 1.0
    a, b = np.percentile(values[valid], [lo, hi])
    if not math.isfinite(a) or not math.isfinite(b) or b <= a:
        b = a + 1.0
    return float(a), float(b)


def colorize(values: np.ndarray, mask: np.ndarray, vmin: float, vmax: float, cmap_name: str) -> np.ndarray:
    import matplotlib.pyplot as plt

    norm = np.clip((values - vmin) / (vmax - vmin + 1e-6), 0.0, 1.0)
    rgb = (plt.get_cmap(cmap_name)(norm)[..., :3] * 255).astype(np.uint8)
    rgb[~mask] = 245
    return rgb


def downsample(image: np.ndarray, max_width: int = 1100) -> np.ndarray:
    h, w = image.shape[:2]
    if w <= max_width:
        return image
    new_w = max_width
    new_h = max(1, int(round(h * new_w / w)))
    pil = Image.fromarray(image)
    return np.asarray(pil.resize((new_w, new_h), Image.BILINEAR))


def make_depth_comparison(args: argparse.Namespace) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    manifest_rows = read_csv(args.manifest)
    direct_rows = read_csv(args.direct_metrics)
    ours_rows = read_csv(args.ours_metrics)
    scene = common_scene(direct_rows, ours_rows, args.dataset, args.scene)

    manifest = find_manifest_row(manifest_rows, args.dataset, scene)
    rgb_t = load_rgb(Path(manifest["rgb_path"]).expanduser())
    gt_t = load_depth(Path(manifest["depth_path"]).expanduser(), float(manifest.get("depth_scale") or 1.0))
    mask_text = manifest.get("mask_path", "").strip()
    mask_t = load_mask(Path(mask_text).expanduser() if mask_text else None, tuple(gt_t.shape))
    rgb_t, gt_t, mask_t = normalize_dataset_layout(args.dataset, rgb_t, gt_t, mask_t)

    rgb = (tensor_to_numpy(rgb_t.permute(1, 2, 0)).clip(0, 1) * 255).astype(np.uint8)
    gt = tensor_to_numpy(gt_t)
    mask = tensor_to_numpy(mask_t.bool()).astype(bool)

    direct = resize_prediction_like_gt(load_prediction_from_row(row_for_scene(direct_rows, args.dataset, scene)), gt)
    ours = resize_prediction_like_gt(load_prediction_from_row(row_for_scene(ours_rows, args.dataset, scene)), gt)

    valid = mask & np.isfinite(gt) & (gt > args.min_depth) & (gt < args.max_depth)
    depth_min, depth_max = percentile_range(gt, valid, 2, 98)

    direct_err = np.abs(direct - gt) / np.maximum(gt, 1e-6)
    ours_err = np.abs(ours - gt) / np.maximum(gt, 1e-6)
    err_all = np.maximum(direct_err, ours_err)
    _, err_max = percentile_range(err_all, valid, 0, 95)
    err_min = 0.0

    panels = [
        ("RGB", downsample(rgb, args.max_panel_width)),
        ("GT Depth", downsample(colorize(gt, valid, depth_min, depth_max, "magma"), args.max_panel_width)),
        ("Direct Depth", downsample(colorize(direct, valid, depth_min, depth_max, "magma"), args.max_panel_width)),
        ("Ours Depth", downsample(colorize(ours, valid, depth_min, depth_max, "magma"), args.max_panel_width)),
        ("Direct Error", downsample(colorize(direct_err, valid, err_min, err_max, "inferno"), args.max_panel_width)),
        ("Ours Error", downsample(colorize(ours_err, valid, err_min, err_max, "inferno"), args.max_panel_width)),
    ]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, len(panels), figsize=(18, 3.7), constrained_layout=True)
    for ax, (title, image) in zip(axes, panels):
        ax.imshow(image)
        ax.set_title(title, fontsize=11)
        ax.axis("off")
    fig.suptitle(f"{args.dataset} / {scene}", fontsize=13)
    fig.savefig(args.output, dpi=180)
    plt.close(fig)
    print(f"[wrote] {args.output}")


def make_metric_bars(args: argparse.Namespace) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = read_csv(args.summary_csv)
    datasets = [row["Dataset"] for row in rows]
    metrics = [
        ("AbsRel", "Baseline AbsRel", "Ours AbsRel", "lower is better"),
        ("RMSE", "Baseline RMSE", "Ours RMSE", "lower is better"),
        ("delta1", "Baseline delta1", "Ours delta1", "higher is better"),
    ]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6), constrained_layout=True)
    x = np.arange(len(datasets))
    width = 0.36
    for ax, (title, base_key, ours_key, subtitle) in zip(axes, metrics):
        baseline = [float(row[base_key]) for row in rows]
        ours = [float(row[ours_key]) for row in rows]
        ax.bar(x - width / 2, baseline, width, label="Direct", color="#9aa6b2")
        ax.bar(x + width / 2, ours, width, label="Ours", color="#2f80ed")
        ax.set_title(f"{title}\n{subtitle}", fontsize=11)
        ax.set_xticks(x)
        ax.set_xticklabels(datasets, rotation=20, ha="right")
        ax.grid(axis="y", alpha=0.25)
        for i, value in enumerate(ours):
            ax.text(i + width / 2, value, f"{value:.3f}", ha="center", va="bottom", fontsize=8)
    axes[0].legend(frameon=False, fontsize=9)
    fig.suptitle("Panorama depth metrics: Direct baseline vs Ours", fontsize=13)
    fig.savefig(args.output, dpi=220)
    plt.close(fig)
    print(f"[wrote] {args.output}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    bars = sub.add_parser("metric-bars", help="Create a three-metric bar chart.")
    bars.add_argument("--summary-csv", type=Path, required=True)
    bars.add_argument("--output", type=Path, required=True)
    bars.set_defaults(func=make_metric_bars)

    comp = sub.add_parser("depth-comparison", help="Create RGB/GT/Direct/Ours/Error comparison.")
    comp.add_argument("--manifest", type=Path, required=True)
    comp.add_argument("--direct-metrics", type=Path, required=True)
    comp.add_argument("--ours-metrics", type=Path, required=True)
    comp.add_argument("--dataset", required=True)
    comp.add_argument("--scene", default="")
    comp.add_argument("--output", type=Path, required=True)
    comp.add_argument("--min-depth", type=float, default=1e-3)
    comp.add_argument("--max-depth", type=float, default=100.0)
    comp.add_argument("--max-panel-width", type=int, default=1100)
    comp.set_defaults(func=make_depth_comparison)

    args = parser.parse_args()
    if hasattr(args, "scene") and not args.scene:
        args.scene = None
    args.func(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

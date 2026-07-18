"""Evaluate perspective-to-panorama depth fusion on panoramic depth datasets.

This script evaluates the experiment idea used for the DAP Table 3 style
benchmarks:

    panorama RGB -> perspective views -> monocular depth -> panorama fusion

The direct panoramic-depth baseline can be added from published Table 3 numbers
with --include-table3-direct-baselines. Ground-truth paths are provided through
a CSV manifest so the same code can be used for Stanford2D3D, Matterport3D, and
Deep360 once the local benchmark files are prepared.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from struct import unpack
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


TABLE3_DIRECT_BASELINES = {
    "Stanford2D3D": {
        "UniK3D": {"abs_rel": 0.1795, "rmse": 0.4850, "delta1": 0.7823},
        "DAC": {"abs_rel": 0.1366, "rmse": 0.4509, "delta1": 0.8393},
        "DAP-Direct-Table3": {"abs_rel": 0.0921, "rmse": 0.3820, "delta1": 0.9135},
    },
    "Matterport3D": {
        "UniK3D": {"abs_rel": 0.2224, "rmse": 0.6680, "delta1": 0.6634},
        "DAC": {"abs_rel": 0.1803, "rmse": 0.9390, "delta1": 0.7203},
        "DAP-Direct-Table3": {"abs_rel": 0.1186, "rmse": 0.7510, "delta1": 0.8518},
    },
    "Deep360": {
        "UniK3D": {"abs_rel": 0.0885, "rmse": 6.1480, "delta1": 0.9293},
        "DAC": {"abs_rel": 0.2611, "rmse": 8.3710, "delta1": 0.6311},
        "DAP-Direct-Table3": {"abs_rel": 0.0659, "rmse": 5.2240, "delta1": 0.9525},
    },
}


@dataclass
class ManifestItem:
    dataset: str
    scene: str
    rgb_path: Path
    depth_path: Path
    mask_path: Path | None
    depth_scale: float


@dataclass
class Metrics:
    abs_rel: float
    rmse: float
    delta1: float
    valid_pixels: int


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def canonical_method(name: str) -> str:
    key = name.lower().replace("-", "_")
    aliases = {
        "da3": "depth_anything3",
        "depthanything3": "depth_anything3",
        "depth_anything_3": "depth_anything3",
        "dreamscene": "dreamscene360",
        "dreamscene_360": "dreamscene360",
        "pano_geo": "dreamscene360",
        "panogeo": "dreamscene360",
        "pano_geo_predictor": "dreamscene360",
        "geometryvlm": "g2vlm",
        "geometry_vlm": "g2vlm",
        "g2_vlm": "g2vlm",
        "vggt": "vggt_omega",
        "vggtomega": "vggt_omega",
    }
    return aliases.get(key, key)


def build_predictor(method: str, args: argparse.Namespace):
    method = canonical_method(method)
    if method == "omnidata":
        from geo_predictors.omnidata_predictor import OmnidataPredictor

        return OmnidataPredictor()
    if method == "depth_anything3":
        from geo_predictors.depth_anything3_predictor import DepthAnything3Predictor

        return DepthAnything3Predictor(
            model_id=args.depth_anything3_model,
            command=args.depth_anything3_command or None,
        )
    if method == "dap":
        from geo_predictors.external_depth_predictor import DAPPredictor

        return DAPPredictor(
            root=args.dap_root or None,
            model_path=args.dap_model_path or None,
            command=args.dap_command or None,
        )
    if method == "g2vlm":
        from geo_predictors.g2vlm_predictor import G2VLMPredictor

        return G2VLMPredictor(
            g2vlm_root=args.g2vlm_root or None,
            model_path=args.g2vlm_model_path or None,
        )
    if method == "vggt_omega":
        from geo_predictors.vggt_predictor import VGGTPredictor

        return VGGTPredictor(
            vggt_root=args.vggt_root or None,
            model_path=args.vggt_model_path or None,
            chunk_size=args.vggt_chunk_size,
        )
    raise ValueError(f"Unknown monocular depth method: {method}")


def build_dreamscene360_predictor(args: argparse.Namespace):
    from geo_predictors.pano_geo_predictor import PanoGeoPredictor

    return PanoGeoPredictor(
        depth_predictor_name=canonical_method(args.dreamscene360_depth_predictor),
        g2vlm_root=args.g2vlm_root or None,
        g2vlm_model_path=args.g2vlm_model_path or None,
        depth_anything3_model=args.depth_anything3_model,
        depth_anything3_command=args.depth_anything3_command or None,
        dap_root=args.dap_root or None,
        dap_model_path=args.dap_model_path or None,
        dap_command=args.dap_command or None,
        vggt_root=args.vggt_root or None,
        vggt_model_path=args.vggt_model_path or None,
        vggt_chunk_size=args.vggt_chunk_size,
    )


def read_manifest(path: Path) -> list[ManifestItem]:
    items: list[ManifestItem] = []
    with path.open(newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            rgb = Path(row["rgb_path"]).expanduser()
            depth = Path(row["depth_path"]).expanduser()
            mask_text = row.get("mask_path", "").strip()
            mask = Path(mask_text).expanduser() if mask_text else None
            scene = row.get("scene", "").strip() or rgb.stem
            scale_text = row.get("depth_scale", "").strip()
            items.append(
                ManifestItem(
                    dataset=row["dataset"].strip(),
                    scene=scene,
                    rgb_path=rgb,
                    depth_path=depth,
                    mask_path=mask,
                    depth_scale=float(scale_text) if scale_text else 1.0,
                )
            )
    return items


def parse_name_set(text: str) -> set[str]:
    return {name.strip() for name in text.split(",") if name.strip()}


def filter_manifest_items(
    items: list[ManifestItem],
    datasets: str,
    max_items: int,
    max_per_dataset: int,
) -> list[ManifestItem]:
    dataset_names = parse_name_set(datasets)
    if dataset_names:
        items = [item for item in items if item.dataset in dataset_names]

    if max_per_dataset > 0:
        counts: dict[str, int] = {}
        selected: list[ManifestItem] = []
        for item in items:
            count = counts.get(item.dataset, 0)
            if count >= max_per_dataset:
                continue
            selected.append(item)
            counts[item.dataset] = count + 1
        items = selected

    if max_items > 0:
        items = items[:max_items]
    return items


def load_rgb(path: Path) -> torch.Tensor:
    image = Image.open(path).convert("RGB")
    array = np.asarray(image).astype(np.float32) / 255.0
    return torch.from_numpy(array).permute(2, 0, 1)


def read_dpt(path: Path) -> np.ndarray:
    """Read Middlebury-style .dpt depth files used by the Bath Matterport3D release."""
    tag_float = 202021.25
    with path.open("rb") as f:
        tag = unpack("f", f.read(4))[0]
        width = unpack("i", f.read(4))[0]
        height = unpack("i", f.read(4))[0]
        if tag != tag_float:
            raise ValueError(f"{path} has an invalid .dpt tag: {tag}")
        if width <= 0 or width >= 100000 or height <= 0 or height >= 100000:
            raise ValueError(f"{path} has an invalid .dpt shape: {width}x{height}")
        depth = np.fromfile(f, np.float32)
    if depth.size != width * height:
        raise ValueError(f"{path} contains {depth.size} values, expected {width * height}")
    return depth.reshape(height, width)


def load_depth(path: Path, depth_scale: float = 1.0) -> torch.Tensor:
    suffix = path.suffix.lower()
    if suffix == ".npy":
        depth = np.load(path)
    elif suffix == ".npz":
        data = np.load(path)
        key = "depth" if "depth" in data.files else data.files[0]
        depth = data[key]
    elif suffix == ".dpt":
        depth = read_dpt(path)
    else:
        depth = np.asarray(Image.open(path))
    depth = np.asarray(depth).squeeze().astype(np.float32)
    if depth_scale != 1.0:
        depth = depth / depth_scale
    return torch.from_numpy(depth)


def load_mask(path: Path | None, shape: tuple[int, int]) -> torch.Tensor:
    if path is None:
        return torch.ones(shape, dtype=torch.bool)
    mask = np.asarray(Image.open(path)).squeeze()
    mask_t = torch.from_numpy(mask > 0)[None, None].float()
    if mask_t.shape[-2:] != shape:
        mask_t = F.interpolate(mask_t, size=shape, mode="nearest")
    return mask_t[0, 0] > 0.5


def save_depth_preview(depth: torch.Tensor, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    depth_np = depth.detach().float().cpu().numpy()
    valid = np.isfinite(depth_np) & (depth_np > 0)
    if not valid.any():
        Image.fromarray(np.zeros(depth_np.shape, dtype=np.uint8)).save(path)
        return
    lo, hi = np.percentile(depth_np[valid], [2, 98])
    norm = np.clip((depth_np - lo) / (hi - lo + 1e-6), 0.0, 1.0)
    rgb = (plt.get_cmap("magma")(norm)[..., :3] * 255).astype(np.uint8)
    Image.fromarray(rgb).save(path)


def equirectangular_dirs(height: int, width: int, device: torch.device) -> torch.Tensor:
    y, x = torch.meshgrid(
        torch.arange(height, device=device, dtype=torch.float32) + 0.5,
        torch.arange(width, device=device, dtype=torch.float32) + 0.5,
        indexing="ij",
    )
    alpha = (0.5 - x / width) * (2.0 * math.pi)
    beta = (0.5 - y / height) * math.pi
    dirs = torch.stack(
        [
            torch.cos(beta) * torch.cos(alpha),
            torch.cos(beta) * torch.sin(alpha),
            torch.sin(beta),
        ],
        dim=-1,
    )
    return F.normalize(dirs, dim=-1)


def dirs_to_pano_xy(dirs: torch.Tensor, height: int, width: int) -> tuple[torch.Tensor, torch.Tensor]:
    dirs = F.normalize(dirs, dim=-1)
    beta = torch.asin(torch.clamp(dirs[..., 2], -1.0, 1.0))
    alpha = torch.atan2(dirs[..., 1], dirs[..., 0])
    x = (-(alpha / (2.0 * math.pi)) + 0.5) * width - 0.5
    y = (-beta / math.pi + 0.5) * height - 0.5
    return x.remainder(width), y.clamp(0, height - 1)


def dirs_to_sample_grid(dirs: torch.Tensor, pano_height: int, pano_width: int) -> torch.Tensor:
    x, y = dirs_to_pano_xy(dirs, pano_height, pano_width)
    grid_x = x / max(pano_width - 1, 1) * 2.0 - 1.0
    grid_y = y / max(pano_height - 1, 1) * 2.0 - 1.0
    return torch.stack([grid_x, grid_y], dim=-1)


def rotation_from_yaw_pitch(yaw_degrees: float, pitch_degrees: float, device: torch.device) -> torch.Tensor:
    yaw = math.radians(yaw_degrees)
    pitch = math.radians(pitch_degrees)
    forward = torch.tensor(
        [
            math.cos(pitch) * math.cos(yaw),
            -math.cos(pitch) * math.sin(yaw),
            math.sin(pitch),
        ],
        dtype=torch.float32,
        device=device,
    )
    forward = F.normalize(forward, dim=0)
    up = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32, device=device)
    if abs(float(torch.dot(forward, up))) > 0.98:
        up = torch.tensor([0.0, 1.0, 0.0], dtype=torch.float32, device=device)
    right = F.normalize(torch.linalg.cross(forward, -up), dim=0)
    down = F.normalize(torch.linalg.cross(forward, right), dim=0)
    return torch.stack([right, down, forward], dim=1)


def perspective_dirs(
    view_size: int,
    fov_degrees: float,
    yaw_degrees: float,
    pitch_degrees: float,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    fov = math.radians(fov_degrees)
    half = math.tan(fov * 0.5)
    coords = torch.linspace(
        -1.0 + 1.0 / view_size,
        1.0 - 1.0 / view_size,
        view_size,
        dtype=torch.float32,
        device=device,
    )
    yy, xx = torch.meshgrid(coords, coords, indexing="ij")
    cam = torch.stack([xx * half, yy * half, torch.ones_like(xx)], dim=-1)
    ray_ratio = torch.linalg.norm(cam, dim=-1)
    cam_dirs = F.normalize(cam, dim=-1)
    rot = rotation_from_yaw_pitch(yaw_degrees, pitch_degrees, device)
    world_dirs = cam_dirs @ rot.T
    return F.normalize(world_dirs, dim=-1), ray_ratio


def view_angles(yaw_count: int, pitch_degrees: Iterable[float]) -> list[tuple[float, float]]:
    pitches = sorted([float(p) for p in pitch_degrees], key=lambda p: (abs(p), p))
    return [
        (360.0 * yaw_index / yaw_count, pitch)
        for pitch in pitches
        for yaw_index in range(yaw_count)
    ]


def run_predictor_batch(
    predictor,
    images: torch.Tensor,
    batch_size: int,
    fov_degrees: float,
) -> torch.Tensor:
    depths = []
    _, _, height, width = images.shape
    focal = float(width) * 0.5 / math.tan(math.radians(fov_degrees) * 0.5)
    intri = {
        "fx": focal,
        "fy": focal,
        "cx": float(width) / 2.0,
        "cy": float(height) / 2.0,
    }
    for start in range(0, images.shape[0], batch_size):
        batch = images[start : start + batch_size]
        with torch.no_grad():
            if hasattr(predictor, "predict_depth_batch"):
                pred = predictor.predict_depth_batch(batch, intrinsics=[intri] * batch.shape[0])
            else:
                pred = torch.cat(
                    [predictor.predict_depth(batch[i : i + 1], intri=intri) for i in range(batch.shape[0])],
                    dim=0,
                )
        if pred.dim() == 3:
            pred = pred[:, None]
        if pred.shape[-2:] != batch.shape[-2:]:
            pred = F.interpolate(pred.float(), size=batch.shape[-2:], mode="bilinear", align_corners=False)
        depths.append(pred[:, 0].float().detach())
    return torch.cat(depths, dim=0)


def bilinear_splat(
    value_sum: torch.Tensor,
    weight_sum: torch.Tensor,
    x: torch.Tensor,
    y: torch.Tensor,
    values: torch.Tensor,
    weights: torch.Tensor,
) -> None:
    height, width = value_sum.shape
    x0 = torch.floor(x).long()
    y0 = torch.floor(y).long()
    dx = x - x0.float()
    dy = y - y0.float()
    x1 = (x0 + 1).remainder(width)
    y1 = (y0 + 1).clamp(0, height - 1)
    x0 = x0.remainder(width)
    y0 = y0.clamp(0, height - 1)

    for xx, yy, ww in (
        (x0, y0, (1.0 - dx) * (1.0 - dy)),
        (x1, y0, dx * (1.0 - dy)),
        (x0, y1, (1.0 - dx) * dy),
        (x1, y1, dx * dy),
    ):
        w = weights * ww
        flat = yy.reshape(-1) * width + xx.reshape(-1)
        value_sum.reshape(-1).scatter_add_(0, flat, (values * w).reshape(-1))
        weight_sum.reshape(-1).scatter_add_(0, flat, w.reshape(-1))


def sample_pano_map(pano: torch.Tensor, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    height, width = pano.shape
    grid_x = x / max(width - 1, 1) * 2.0 - 1.0
    grid_y = y / max(height - 1, 1) * 2.0 - 1.0
    grid = torch.stack([grid_x, grid_y], dim=-1)[None]
    return F.grid_sample(
        pano[None, None].float(),
        grid,
        mode="bilinear",
        padding_mode="border",
        align_corners=True,
    )[0, 0]


def align_view_to_fused_depth(
    radial_depth: torch.Tensor,
    center_weight: torch.Tensor,
    x: torch.Tensor,
    y: torch.Tensor,
    value_sum: torch.Tensor,
    weight_sum: torch.Tensor,
    min_pixels: int,
    scale_min: float,
    scale_max: float,
) -> torch.Tensor:
    if not bool((weight_sum > 1e-8).any()):
        return radial_depth

    fused_depth = value_sum / weight_sum.clamp_min(1e-8)
    ref_depth = sample_pano_map(fused_depth, x, y)
    ref_weight = sample_pano_map(weight_sum, x, y)
    valid = (
        (ref_weight > 1e-6)
        & torch.isfinite(ref_depth)
        & torch.isfinite(radial_depth)
        & (ref_depth > 0)
        & (radial_depth > 0)
        & (center_weight > 0.05)
    )
    if int(valid.sum().item()) < min_pixels:
        return radial_depth

    ratios = ref_depth[valid] / radial_depth[valid].clamp_min(1e-6)
    scale = ratios.median().clamp(scale_min, scale_max)
    return radial_depth * scale


def fuse_perspective_depths(
    pano_rgb: torch.Tensor,
    predictor,
    yaw_count: int,
    pitch_degrees: list[float],
    view_size: int,
    fov_degrees: float,
    batch_size: int,
    center_weight_power: float,
    per_view_normalize: str,
    overlap_align: str,
    overlap_min_pixels: int,
    overlap_scale_min: float,
    overlap_scale_max: float,
) -> torch.Tensor:
    device = pano_rgb.device
    _, pano_h, pano_w = pano_rgb.shape
    angles = view_angles(yaw_count, pitch_degrees)

    value_sum = torch.zeros((pano_h, pano_w), dtype=torch.float32, device=device)
    weight_sum = torch.zeros((pano_h, pano_w), dtype=torch.float32, device=device)
    pano_batch = pano_rgb[None]

    for batch_start in range(0, len(angles), batch_size):
        batch_angles = angles[batch_start : batch_start + batch_size]
        dirs_batch = []
        ratios_batch = []
        for yaw, pitch in batch_angles:
            dirs, ray_ratio = perspective_dirs(view_size, fov_degrees, yaw, pitch, device)
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

        pred_depth = run_predictor_batch(
            predictor,
            views,
            batch_size=batch_size,
            fov_degrees=fov_degrees,
        ).to(device)
        pred_depth = pred_depth.clamp_min(1e-6)
        if per_view_normalize == "mean":
            pred_depth = pred_depth / (pred_depth.mean(dim=(1, 2), keepdim=True) + 1e-6)
        elif per_view_normalize == "median":
            pred_depth = pred_depth / (
                pred_depth.flatten(1).median(dim=1).values[:, None, None] + 1e-6
            )
        elif per_view_normalize != "none":
            raise ValueError(f"Unknown per-view normalization: {per_view_normalize}")

        radial_depth = pred_depth * ratios_t
        center_weight = (1.0 / ratios_t).clamp(0.0, 1.0).pow(center_weight_power)
        x, y = dirs_to_pano_xy(dirs_t, pano_h, pano_w)

        for i in range(len(batch_angles)):
            values = radial_depth[i]
            if overlap_align == "progressive":
                values = align_view_to_fused_depth(
                    radial_depth=values,
                    center_weight=center_weight[i],
                    x=x[i],
                    y=y[i],
                    value_sum=value_sum,
                    weight_sum=weight_sum,
                    min_pixels=overlap_min_pixels,
                    scale_min=overlap_scale_min,
                    scale_max=overlap_scale_max,
                )
            elif overlap_align != "none":
                raise ValueError(f"Unknown overlap alignment mode: {overlap_align}")
            bilinear_splat(value_sum, weight_sum, x[i], y[i], values, center_weight[i])

    fused = value_sum / weight_sum.clamp_min(1e-8)
    valid = weight_sum > 1e-8
    if not bool(valid.all()):
        fallback = fused[valid].median() if bool(valid.any()) else torch.tensor(1.0, device=device)
        fused = torch.where(valid, fused, fallback)
    return fused


def predict_dreamscene360_pano_depth(
    pano_rgb: torch.Tensor,
    predictor,
    gen_res: int,
    reg_loss_weight: float,
    depth_normalize: str,
    all_iter_steps: int,
    num_perspectives: int,
) -> torch.Tensor:
    """Run DreamScene360's PanoGeo depth optimization on one panorama."""
    pano_hwc = pano_rgb.permute(1, 2, 0).contiguous()
    pred, *_ = predictor(
        pano_hwc,
        gen_res=gen_res,
        reg_loss_weight=reg_loss_weight,
        depth_normalize=depth_normalize,
        all_iter_steps=all_iter_steps,
        num_perspectives=num_perspectives,
    )
    if pred.dim() == 3:
        pred = pred[..., 0]
    return pred.float()


def predict_direct_panorama_depth(
    pano_rgb: torch.Tensor,
    predictor,
) -> torch.Tensor:
    """Run a depth predictor directly on the full panorama image."""
    image = pano_rgb[None]
    with torch.no_grad():
        if hasattr(predictor, "predict_depth_batch"):
            pred = predictor.predict_depth_batch(image, intrinsics=[{}])
        else:
            pred = predictor.predict_depth(image, intri={})

    if pred.dim() == 4:
        pred = pred[0, 0]
    elif pred.dim() == 3:
        pred = pred[0]
    elif pred.dim() != 2:
        raise ValueError(f"Unexpected direct panorama prediction shape: {tuple(pred.shape)}")
    return pred.float()


def resize_like(pred: torch.Tensor, target_hw: tuple[int, int]) -> torch.Tensor:
    if pred.shape == target_hw:
        return pred
    pred_t = pred[None, None].float()
    return F.interpolate(pred_t, size=target_hw, mode="bilinear", align_corners=False)[0, 0]


def align_prediction(pred: torch.Tensor, gt: torch.Tensor, mask: torch.Tensor, mode: str) -> torch.Tensor:
    if mode == "none":
        return pred
    pred_valid = pred[mask].float()
    gt_valid = gt[mask].float()
    if pred_valid.numel() == 0:
        return pred
    if mode == "median":
        scale = gt_valid.median() / pred_valid.clamp_min(1e-6).median().clamp_min(1e-6)
        return pred * scale
    if mode == "least_squares":
        denom = (pred_valid * pred_valid).mean().clamp_min(1e-8)
        scale = (pred_valid * gt_valid).mean() / denom
        return pred * scale
    raise ValueError(f"Unknown eval alignment mode: {mode}")


def compute_metrics(
    pred: torch.Tensor,
    gt: torch.Tensor,
    mask: torch.Tensor,
    min_depth: float,
    max_depth: float,
) -> Metrics:
    valid = (
        mask
        & torch.isfinite(pred)
        & torch.isfinite(gt)
        & (pred > 0)
        & (gt > min_depth)
        & (gt < max_depth)
    )
    if int(valid.sum()) == 0:
        return Metrics(float("nan"), float("nan"), float("nan"), 0)

    p = pred[valid].float().clamp_min(1e-6)
    g = gt[valid].float().clamp_min(1e-6)
    abs_rel = torch.mean(torch.abs(p - g) / g)
    rmse = torch.sqrt(torch.mean((p - g) ** 2))
    thresh = torch.maximum(p / g, g / p)
    delta1 = torch.mean((thresh < 1.25).float())
    return Metrics(
        abs_rel=float(abs_rel.item()),
        rmse=float(rmse.item()),
        delta1=float(delta1.item()),
        valid_pixels=int(valid.sum().item()),
    )


def depth_scale_stats(pred: torch.Tensor, gt: torch.Tensor, mask: torch.Tensor) -> dict[str, float]:
    valid = mask & torch.isfinite(pred) & torch.isfinite(gt) & (pred > 0) & (gt > 0)
    if int(valid.sum()) == 0:
        nan = float("nan")
        return {
            "pred_median": nan,
            "gt_median": nan,
            "median_scale_to_gt": nan,
            "pred_mean": nan,
            "gt_mean": nan,
        }
    p = pred[valid].float()
    g = gt[valid].float()
    pred_median = p.median().clamp_min(1e-6)
    gt_median = g.median()
    return {
        "pred_median": float(pred_median.item()),
        "gt_median": float(gt_median.item()),
        "median_scale_to_gt": float((gt_median / pred_median).item()),
        "pred_mean": float(p.mean().item()),
        "gt_mean": float(g.mean().item()),
    }


def mean_or_nan(values: list[float]) -> float:
    finite = [v for v in values if math.isfinite(v)]
    if not finite:
        return float("nan")
    return float(np.mean(finite))


def write_csv(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def format_metric(value: object) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "NA"
    if not math.isfinite(number):
        return "NA"
    return f"{number:.4f}"


def markdown_table(rows: list[dict[str, object]]) -> str:
    lines = [
        "| Dataset | Method | AbsRel↓ | RMSE↓ | δ1↑ | Num Images |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {dataset} | {method} | {abs_rel} | {rmse} | {delta1} | {num_images} |".format(
                dataset=row["dataset"],
                method=row["method"],
                abs_rel=format_metric(row.get("abs_rel")),
                rmse=format_metric(row.get("rmse")),
                delta1=format_metric(row.get("delta1")),
                num_images=row.get("num_images", ""),
            )
        )
    return "\n".join(lines) + "\n"


def format_delta(value: object) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "NA"
    if not math.isfinite(number):
        return "NA"
    return f"{number:+.4f}"


def format_bool(value: object) -> str:
    return "YES" if bool(value) else "NO"


def build_direct_comparison_rows(
    rows: list[dict[str, object]],
    baseline_method: str = "DAP-Direct-Table3",
) -> list[dict[str, object]]:
    comparisons: list[dict[str, object]] = []
    for row in rows:
        dataset = str(row["dataset"])
        baseline = TABLE3_DIRECT_BASELINES.get(dataset, {}).get(baseline_method)
        if baseline is None:
            continue

        abs_rel = float(row["abs_rel"])
        rmse = float(row["rmse"])
        delta1 = float(row["delta1"])
        baseline_abs_rel = float(baseline["abs_rel"])
        baseline_rmse = float(baseline["rmse"])
        baseline_delta1 = float(baseline["delta1"])

        abs_rel_better = abs_rel < baseline_abs_rel
        rmse_better = rmse < baseline_rmse
        delta1_better = delta1 > baseline_delta1
        comparisons.append(
            {
                "dataset": dataset,
                "method": row["method"],
                "baseline_method": baseline_method,
                "abs_rel": abs_rel,
                "baseline_abs_rel": baseline_abs_rel,
                "abs_rel_delta": abs_rel - baseline_abs_rel,
                "abs_rel_better": abs_rel_better,
                "rmse": rmse,
                "baseline_rmse": baseline_rmse,
                "rmse_delta": rmse - baseline_rmse,
                "rmse_better": rmse_better,
                "delta1": delta1,
                "baseline_delta1": baseline_delta1,
                "delta1_delta": delta1 - baseline_delta1,
                "delta1_better": delta1_better,
                "all_metrics_better": abs_rel_better and rmse_better and delta1_better,
            }
        )
    return comparisons


def comparison_markdown(rows: list[dict[str, object]]) -> str:
    lines = [
        "| Dataset | Method | Baseline | AbsRel delta | RMSE delta | delta1 delta | All better |",
        "|---|---|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {dataset} | {method} | {baseline_method} | {abs_rel_delta} | {rmse_delta} | {delta1_delta} | {all_better} |".format(
                dataset=row["dataset"],
                method=row["method"],
                baseline_method=row["baseline_method"],
                abs_rel_delta=format_delta(row.get("abs_rel_delta")),
                rmse_delta=format_delta(row.get("rmse_delta")),
                delta1_delta=format_delta(row.get("delta1_delta")),
                all_better=format_bool(row.get("all_metrics_better")),
            )
        )
    return "\n".join(lines) + "\n"


def add_table3_baselines(rows: list[dict[str, object]]) -> None:
    for dataset, methods in TABLE3_DIRECT_BASELINES.items():
        for method, metrics in methods.items():
            rows.append(
                {
                    "dataset": dataset,
                    "method": method,
                    "abs_rel": metrics["abs_rel"],
                    "rmse": metrics["rmse"],
                    "delta1": metrics["delta1"],
                    "num_images": "Table3",
                }
            )


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate perspective-view monocular depth fused into panorama depth.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("panorama_depth_fusion_eval"))
    parser.add_argument(
        "--method",
        default="dap",
        help="Method to evaluate: omnidata, depth_anything3, dap, g2vlm, vggt_omega, or dreamscene360.",
    )
    parser.add_argument(
        "--eval-mode",
        choices=["perspective_fusion", "direct_panorama"],
        default="perspective_fusion",
        help="perspective_fusion cuts perspective views and fuses them; direct_panorama predicts depth on the full panorama.",
    )
    parser.add_argument("--method-label", default="", help="Name used in output tables")
    parser.add_argument("--datasets", default="", help="Comma-separated dataset names to evaluate, for example Matterport3D,Stanford2D3D.")
    parser.add_argument("--max-items", type=int, default=0, help="Evaluate only the first N manifest rows; 0 means all rows.")
    parser.add_argument("--max-per-dataset", type=int, default=0, help="Evaluate at most N rows from each dataset; 0 means no per-dataset cap.")
    parser.add_argument("--seed", type=int, default=-1, help="Set numpy/torch seed when >= 0.")
    parser.add_argument("--yaw-count", type=int, default=12)
    parser.add_argument("--pitch-degrees", default="-45,0,45")
    parser.add_argument("--view-size", type=int, default=384)
    parser.add_argument("--fov", type=float, default=90.0)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--center-weight-power", type=float, default=2.0)
    parser.add_argument("--per-view-normalize", choices=["none", "mean", "median"], default="none")
    parser.add_argument(
        "--overlap-align",
        choices=["none", "progressive"],
        default="none",
        help="Align each perspective depth to previously fused overlapping views without using GT.",
    )
    parser.add_argument("--overlap-min-pixels", type=int, default=2048)
    parser.add_argument("--overlap-scale-min", type=float, default=0.25)
    parser.add_argument("--overlap-scale-max", type=float, default=4.0)
    parser.add_argument("--eval-align", choices=["none", "median", "least_squares"], default="none",
                        help="Optional global scale alignment before metrics. Use none for metric-depth evaluation.")
    parser.add_argument(
        "--prediction-scale",
        type=float,
        default=1.0,
        help="Scale applied to fused predictions before optional eval alignment.",
    )
    parser.add_argument("--min-depth", type=float, default=1e-3)
    parser.add_argument("--max-depth", type=float, default=100.0)
    parser.add_argument("--save-predictions", action="store_true")
    parser.add_argument("--include-table3-direct-baselines", action="store_true")
    parser.add_argument(
        "--dreamscene360-depth-predictor",
        default=os.getenv("DREAMSCENE360_DEPTH_PREDICTOR", "omnidata"),
        help="Inner depth predictor used by --method dreamscene360.",
    )
    parser.add_argument("--pano-geo-gen-res", type=int, default=512)
    parser.add_argument("--pano-geo-reg-loss-weight", type=float, default=1e-1)
    parser.add_argument("--pano-geo-depth-normalize", choices=["none", "mean", "median"], default="mean")
    parser.add_argument("--pano-geo-iters", type=int, default=1500)
    parser.add_argument(
        "--pano-geo-num-perspectives",
        type=int,
        default=20,
        help="Number of DreamScene360 perspective depth views used by PanoGeo optimization.",
    )
    parser.add_argument("--depth-anything3-model", default="depth-anything/DA3-LARGE-1.1")
    parser.add_argument("--depth-anything3-command", default=os.getenv("DEPTH_ANYTHING3_COMMAND", ""))
    parser.add_argument("--dap-root", default=os.getenv("DAP_ROOT", ""))
    parser.add_argument("--dap-model-path", default=os.getenv("DAP_MODEL_PATH", ""))
    parser.add_argument("--dap-command", default=os.getenv("DAP_DEPTH_COMMAND", ""))
    parser.add_argument("--g2vlm-root", default=os.getenv("G2VLM_ROOT", ""))
    parser.add_argument("--g2vlm-model-path", default=os.getenv("G2VLM_MODEL_PATH", ""))
    parser.add_argument("--vggt-root", default=os.getenv("VGGT_ROOT", ""))
    parser.add_argument("--vggt-model-path", default=os.getenv("VGGT_MODEL_PATH", "facebook/VGGT-1B"))
    parser.add_argument("--vggt-chunk-size", type=int, default=8)
    args = parser.parse_args()

    sys.path.insert(0, str(repo_root()))
    if not torch.cuda.is_available():
        raise RuntimeError("This evaluator needs CUDA for the DreamScene360 depth predictors.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    pred_dir = args.output_dir / "predictions"
    if args.save_predictions:
        pred_dir.mkdir(parents=True, exist_ok=True)

    method = canonical_method(args.method)
    if args.seed >= 0:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
    if method == "dreamscene360":
        if args.eval_mode == "direct_panorama":
            raise ValueError("--method dreamscene360 already runs PanoGeo fusion and cannot use --eval-mode direct_panorama.")
        inner_method = canonical_method(args.dreamscene360_depth_predictor)
        method_label = args.method_label or f"DreamScene360-PanoGeo-{inner_method}"
    elif args.eval_mode == "direct_panorama":
        method_label = args.method_label or f"{method}-direct-panorama"
    else:
        method_label = args.method_label or f"{method}-perspective-fusion"
    pitch_degrees = [float(x) for x in args.pitch_degrees.split(",") if x.strip()]
    items = filter_manifest_items(
        read_manifest(args.manifest),
        datasets=args.datasets,
        max_items=args.max_items,
        max_per_dataset=args.max_per_dataset,
    )
    if not items:
        raise RuntimeError("No manifest items selected. Check --manifest, --datasets, and max item filters.")
    if method == "dreamscene360":
        predictor = build_dreamscene360_predictor(args)
    else:
        predictor = build_predictor(method, args)
    device = torch.device("cuda")

    per_image_rows: list[dict[str, object]] = []
    for index, item in enumerate(items):
        print(f"[{index + 1}/{len(items)}] {item.dataset}/{item.scene}", flush=True)
        rgb = load_rgb(item.rgb_path).to(device)
        gt = load_depth(item.depth_path, item.depth_scale).to(device)
        mask = load_mask(item.mask_path, tuple(gt.shape)).to(device)
        if method == "dreamscene360":
            pred = predict_dreamscene360_pano_depth(
                pano_rgb=rgb,
                predictor=predictor,
                gen_res=args.pano_geo_gen_res,
                reg_loss_weight=args.pano_geo_reg_loss_weight,
                depth_normalize=args.pano_geo_depth_normalize,
                all_iter_steps=args.pano_geo_iters,
                num_perspectives=args.pano_geo_num_perspectives,
            )
        elif args.eval_mode == "direct_panorama":
            pred = predict_direct_panorama_depth(
                pano_rgb=rgb,
                predictor=predictor,
            )
        else:
            pred = fuse_perspective_depths(
                pano_rgb=rgb,
                predictor=predictor,
                yaw_count=args.yaw_count,
                pitch_degrees=pitch_degrees,
                view_size=args.view_size,
                fov_degrees=args.fov,
                batch_size=args.batch_size,
                center_weight_power=args.center_weight_power,
                per_view_normalize=args.per_view_normalize,
                overlap_align=args.overlap_align,
                overlap_min_pixels=args.overlap_min_pixels,
                overlap_scale_min=args.overlap_scale_min,
                overlap_scale_max=args.overlap_scale_max,
            )
        pred = pred * args.prediction_scale
        pred = resize_like(pred, tuple(gt.shape))
        scale_stats = depth_scale_stats(pred, gt, mask)
        pred = align_prediction(pred, gt, mask, args.eval_align)
        metrics = compute_metrics(pred, gt, mask, args.min_depth, args.max_depth)

        row = {
            "dataset": item.dataset,
            "scene": item.scene,
            "method": method_label,
            "rgb_path": str(item.rgb_path),
            "depth_path": str(item.depth_path),
            "abs_rel": metrics.abs_rel,
            "rmse": metrics.rmse,
            "delta1": metrics.delta1,
            "valid_pixels": metrics.valid_pixels,
            **scale_stats,
        }
        per_image_rows.append(row)
        print(
            f"  AbsRel={metrics.abs_rel:.4f} RMSE={metrics.rmse:.4f} "
            f"delta1={metrics.delta1:.4f} valid={metrics.valid_pixels}",
            flush=True,
        )

        if args.save_predictions:
            stem = f"{item.dataset}_{item.scene}".replace("/", "_").replace(" ", "_")
            np.save(pred_dir / f"{stem}_pred.npy", pred.detach().float().cpu().numpy())
            save_depth_preview(pred, pred_dir / f"{stem}_pred.png")

        torch.cuda.empty_cache()

    per_image_fields = [
        "dataset",
        "scene",
        "method",
        "rgb_path",
        "depth_path",
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
        rows = [row for row in per_image_rows if row["dataset"] == dataset]
        summary_rows.append(
            {
                "dataset": dataset,
                "method": method_label,
                "abs_rel": mean_or_nan([float(row["abs_rel"]) for row in rows]),
                "rmse": mean_or_nan([float(row["rmse"]) for row in rows]),
                "delta1": mean_or_nan([float(row["delta1"]) for row in rows]),
                "num_images": len(rows),
            }
        )

    evaluated_summary_rows = list(summary_rows)
    comparison = build_direct_comparison_rows(evaluated_summary_rows)
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
    (args.output_dir / "config.json").write_text(
        json.dumps(vars(args), indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    print((args.output_dir / "table3_style.md").read_text(encoding="utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

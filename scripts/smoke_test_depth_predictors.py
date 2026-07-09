"""
Smoke-test DreamScene360 depth predictors on one small image.

This checks that each predictor can be constructed, run, and return a finite
non-constant depth map before launching long ablation jobs.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def find_image(path: Path) -> Path | None:
    if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
        return path
    if path.is_dir():
        for candidate in sorted(path.rglob("*")):
            if candidate.is_file() and candidate.suffix.lower() in IMAGE_SUFFIXES:
                return candidate
    return None


def synthetic_image(size: int) -> Image.Image:
    x = np.linspace(0.0, 1.0, size, dtype=np.float32)
    y = np.linspace(0.0, 1.0, size, dtype=np.float32)
    xx, yy = np.meshgrid(x, y)
    image = np.stack(
        [
            xx,
            yy,
            0.5 + 0.25 * np.sin(xx * math.pi * 6.0) * np.cos(yy * math.pi * 4.0),
        ],
        axis=-1,
    )
    return Image.fromarray(np.uint8(np.clip(image, 0.0, 1.0) * 255.0))


def load_test_image(path: Path | None, size: int) -> tuple[Image.Image, str]:
    image_path = find_image(path) if path is not None else None
    if image_path is None:
        return synthetic_image(size), "synthetic"

    image = Image.open(image_path).convert("RGB")
    width, height = image.size
    crop = min(width, height)
    left = (width - crop) // 2
    top = (height - crop) // 2
    image = image.crop((left, top, left + crop, top + crop)).resize((size, size), Image.BICUBIC)
    return image, str(image_path)


def image_to_tensor(image: Image.Image) -> torch.Tensor:
    arr = np.asarray(image).astype(np.float32) / 255.0
    tensor = torch.from_numpy(arr).permute(2, 0, 1)[None]
    return tensor.cuda()


def depth_preview(depth: torch.Tensor, path: Path) -> None:
    depth = depth.detach().float().cpu()
    if depth.ndim == 4:
        depth = depth[0, 0]
    elif depth.ndim == 3:
        depth = depth[0]
    finite = torch.isfinite(depth)
    if not finite.any():
        vis = torch.zeros_like(depth)
    else:
        values = depth[finite]
        lo = torch.quantile(values, 0.01)
        hi = torch.quantile(values, 0.99)
        vis = ((depth - lo) / (hi - lo + 1e-6)).clamp(0.0, 1.0)
    Image.fromarray(np.uint8(vis.numpy() * 255.0)).save(path)


def depth_stats(depth: torch.Tensor) -> dict[str, object]:
    depth = depth.detach().float().cpu()
    finite = torch.isfinite(depth)
    if not finite.any():
        return {
            "shape": list(depth.shape),
            "finite": False,
            "min": None,
            "max": None,
            "mean": None,
            "std": None,
        }

    values = depth[finite]
    return {
        "shape": list(depth.shape),
        "finite": bool(finite.all().item()),
        "min": float(values.min().item()),
        "max": float(values.max().item()),
        "mean": float(values.mean().item()),
        "std": float(values.std(unbiased=False).item()),
    }


def canonical_method(name: str) -> str:
    key = name.lower().replace("-", "_")
    aliases = {
        "da3": "depth_anything3",
        "depthanything3": "depth_anything3",
        "depth_anything_3": "depth_anything3",
        "vggt": "vggt_omega",
        "vggt_omega": "vggt_omega",
        "vggtomega": "vggt_omega",
    }
    return aliases.get(key, key)


def build_predictor(method: str, args: argparse.Namespace):
    if method == "omnidata":
        from geo_predictors.omnidata_predictor import OmnidataPredictor

        return OmnidataPredictor()
    if method == "depth_anything3":
        from geo_predictors.depth_anything3_predictor import DepthAnything3Predictor

        return DepthAnything3Predictor(
            model_id=args.depth_anything3_model,
            command=args.depth_anything3_command,
        )
    if method == "dap":
        from geo_predictors.external_depth_predictor import DAPPredictor

        return DAPPredictor(
            root=args.dap_root or None,
            model_path=args.dap_model_path or None,
            command=args.dap_command or None,
        )
    if method == "vggt_omega":
        from geo_predictors.vggt_predictor import VGGTPredictor

        return VGGTPredictor(
            vggt_root=args.vggt_root or None,
            model_path=args.vggt_model_path or None,
            chunk_size=args.vggt_chunk_size,
        )
    if method == "g2vlm":
        from geo_predictors.g2vlm_predictor import G2VLMPredictor

        return G2VLMPredictor(
            g2vlm_root=args.g2vlm_root or None,
            model_path=args.g2vlm_model_path or None,
        )
    raise ValueError(f"Unknown depth predictor: {method}")


def run_predictor(method: str, predictor, image: torch.Tensor, batch_size: int) -> torch.Tensor:
    _, _, height, width = image.shape
    intri = {
        "fx": float(width),
        "fy": float(height),
        "cx": float(width) / 2.0,
        "cy": float(height) / 2.0,
    }
    if hasattr(predictor, "predict_depth_batch"):
        batch = image.repeat(max(1, batch_size), 1, 1, 1)
        return predictor.predict_depth_batch(batch, intrinsics=[intri] * batch.shape[0])
    return predictor.predict_depth(image, intri=intri)


def main() -> int:
    parser = argparse.ArgumentParser(description="Smoke-test DreamScene360 depth predictors")
    parser.add_argument("--methods", nargs="+", default=["omnidata", "depth_anything3", "dap", "vggt_omega"])
    parser.add_argument("--image", type=Path, default=Path("data/alley_pano"))
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--output-dir", type=Path, default=Path("output/depth_predictor_smoke"))
    parser.add_argument("--allow-failures", action="store_true")
    parser.add_argument("--depth-anything3-model", default="depth-anything/DA3-LARGE-1.1")
    parser.add_argument("--depth-anything3-command", default="")
    parser.add_argument("--dap-root", default="")
    parser.add_argument("--dap-model-path", default="")
    parser.add_argument("--dap-command", default="")
    parser.add_argument("--vggt-root", default="")
    parser.add_argument("--vggt-model-path", default="")
    parser.add_argument("--vggt-chunk-size", type=int, default=8)
    parser.add_argument("--g2vlm-root", default="")
    parser.add_argument("--g2vlm-model-path", default="")
    args = parser.parse_args()

    sys.path.insert(0, str(repo_root()))
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if not torch.cuda.is_available():
        raise RuntimeError("DreamScene360 depth smoke tests require CUDA.")

    image, image_source = load_test_image(args.image, args.image_size)
    image.save(args.output_dir / "input.png")
    tensor = image_to_tensor(image)

    results: dict[str, dict[str, object]] = {}
    failures = 0
    for raw_method in args.methods:
        method = canonical_method(raw_method)
        print(f"[smoke] testing {method} ...", flush=True)
        try:
            predictor = build_predictor(method, args)
            with torch.no_grad():
                depth = run_predictor(method, predictor, tensor, args.batch_size)
            stats = depth_stats(depth)
            if not stats["finite"]:
                raise RuntimeError("Depth contains non-finite values.")
            if float(stats["std"] or 0.0) <= 1e-8:
                raise RuntimeError("Depth is nearly constant.")
            depth_preview(depth, args.output_dir / f"{method}_depth.png")
            results[method] = {
                "status": "ok",
                "image_source": image_source,
                "stats": stats,
            }
            print(f"[smoke] {method}: OK {stats}", flush=True)
        except Exception as exc:  # noqa: BLE001
            failures += 1
            results[method] = {
                "status": "failed",
                "image_source": image_source,
                "error": str(exc),
            }
            print(f"[smoke] {method}: FAILED {exc}", flush=True)
        finally:
            torch.cuda.empty_cache()

    (args.output_dir / "summary.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"[smoke] wrote {args.output_dir / 'summary.json'}")

    if failures and not args.allow_failures:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

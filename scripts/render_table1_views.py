"""
Render the six fixed views used for Table 1 style quantitative evaluation.

The paper evaluates rendered exploration views rather than the panorama itself:
front / left / right / back plus pitch up / pitch down. This script renders
those views from a trained DreamScene360 Gaussian model and saves RGB images
that can be fed to scripts/evaluate_table1.py.
"""

from __future__ import annotations

import math
import os
import sys
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import torch
import torchvision

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from arguments import ModelParams, PipelineParams, get_combined_args


VIEW_DIRECTIONS = {
    "front": np.array([1.0, 0.0, 0.0], dtype=np.float32),
    "left": np.array([0.0, 1.0, 0.0], dtype=np.float32),
    "back": np.array([-1.0, 0.0, 0.0], dtype=np.float32),
    "right": np.array([0.0, -1.0, 0.0], dtype=np.float32),
    "up": np.array([1e-4, 0.0, 1.0], dtype=np.float32),
    "down": np.array([1e-4, 0.0, -1.0], dtype=np.float32),
}


def normalize(vec: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vec)
    if norm < 1e-8:
        raise ValueError("Cannot normalize zero vector")
    return vec / norm


def look_at_rotation(position: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return R, T for DreamScene360's Camera convention.

    R stores camera-to-world axes as columns in the same style used by the
    navigation visualizer. T is the camera position.
    """
    forward = normalize(target - position)
    world_up = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    if abs(float(np.dot(forward, world_up))) > 0.98:
        world_up = np.array([0.0, 1.0, 0.0], dtype=np.float32)

    down_hint = -world_up
    right = normalize(np.cross(forward, down_hint))
    down = normalize(np.cross(forward, right))
    rotation = np.stack([right, down, forward], axis=1).astype(np.float32)
    translation = position.astype(np.float32)
    return rotation, translation


def depth_to_vis(depth: torch.Tensor) -> torch.Tensor:
    depth = depth.detach()
    if depth.ndim == 3:
        depth = depth[0]
    depth_min = torch.quantile(depth.reshape(-1), 0.01)
    depth_max = torch.quantile(depth.reshape(-1), 0.99)
    depth = (depth - depth_min) / (depth_max - depth_min + 1e-6)
    return depth.clamp(0.0, 1.0).unsqueeze(0).repeat(3, 1, 1)


def maybe_use_generated_data(dataset) -> None:
    generated_data = Path(dataset.model_path) / "generated_data"
    if generated_data.joinpath("sparse").exists() and not Path(dataset.source_path).joinpath("sparse").exists():
        dataset.source_path = str(generated_data.resolve())
        dataset.images = "images"


def main() -> int:
    parser = ArgumentParser(description="Render Table 1 fixed evaluation views")
    model_params = ModelParams(parser, sentinel=True)
    pipeline_params = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--output-dir", type=Path, default=Path("table1_views"))
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--fov", type=float, default=90.0, help="Horizontal and vertical FOV in degrees")
    parser.add_argument("--translation-radius", type=float, default=0.0,
                        help="Move each view slightly along its viewing direction")
    parser.add_argument("--save-depth", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = get_combined_args(parser)

    from gaussian_renderer import render
    from scene import GaussianModel, Scene
    from scene.cameras import Camera
    from utils.general_utils import safe_state

    safe_state(args.quiet)
    dataset = model_params.extract(args)
    maybe_use_generated_data(dataset)
    pipeline = pipeline_params.extract(args)

    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(
            dataset,
            gaussians,
            load_iteration=args.iteration,
            shuffle=False,
            api_key=None,
            self_refinement=None,
            num_prompt=None,
            max_rounds=None,
        )

        bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
        output_dir = args.output_dir
        rgb_dir = output_dir / "renders"
        depth_dir = output_dir / "depth"
        rgb_dir.mkdir(parents=True, exist_ok=True)
        if args.save_depth:
            depth_dir.mkdir(parents=True, exist_ok=True)

        fov = math.radians(args.fov)
        for idx, (name, direction) in enumerate(VIEW_DIRECTIONS.items()):
            direction = normalize(direction)
            position = direction * float(args.translation_radius)
            target = position + direction
            rotation, translation = look_at_rotation(position, target)

            cam = Camera(
                colmap_id=idx,
                R=rotation,
                T=translation,
                FoVx=fov,
                FoVy=fov,
                image=torch.zeros(3, args.height, args.width),
                gt_alpha_mask=None,
                image_name=name,
                uid=idx,
                data_device="cuda",
            )

            result = render(cam, gaussians, pipeline, background)
            torchvision.utils.save_image(result["render"], rgb_dir / f"{idx:02d}_{name}.png")
            if args.save_depth and "depth" in result:
                torchvision.utils.save_image(depth_to_vis(result["depth"]), depth_dir / f"{idx:02d}_{name}.png")

    print(f"Saved Table 1 views to {rgb_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

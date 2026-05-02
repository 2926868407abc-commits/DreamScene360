"""
Render candidate navigation viewpoints from a trained DreamScene360 scene.

Samples positions on the ground plane, renders a perspective view from each,
and saves them so you can visually pick start and goal.

Usage:
  # Generate 100 candidate views
  python render_viewpoints.py \\
      -m output/Italy_nav_test -s data/Italy_text \\
      --num_views 100 --output_dir nav_candidates

  # After browsing images, note the viewpoint numbers
  # e.g. vp_005.png and vp_042.png

  # Then run path planning:
  python inference_nav.py \\
      -m output/Italy_nav_test -s data/Italy_text \\
      --start_vp 5 --goal_vp 42 \\
      --vp_dir nav_candidates
"""
import os
import sys
import torch
import numpy as np
import torchvision
from argparse import ArgumentParser
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from arguments import ModelParams, PipelineParams
from scene import Scene, GaussianModel
from navigation import OccupancyField, DiffPlanner, PathVisualizer
from gaussian_renderer import render


def main():
    parser = ArgumentParser(description="Render candidate navigation viewpoints")
    model_params = ModelParams(parser)
    pipeline_params = PipelineParams(parser)

    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--num_views", type=int, default=36,
                        help="Number of candidate views to generate")
    parser.add_argument("--output_dir", type=str, default="nav_candidates")
    parser.add_argument("--fov", type=float, default=60,
                        help="Field of view for rendered images")
    parser.add_argument("--img_size", type=int, default=256,
                        help="Output image resolution")
    parser.add_argument("--grid_density", type=float, default=0.7,
                        help="Fraction of grid points to keep after "
                             "filtering occupied ones (0-1)")
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # 1. Load trained model
    # ------------------------------------------------------------------
    print("=" * 60)
    print("[1] Loading 3DGS model ...")
    dataset = model_params.extract(args)
    pipe = pipeline_params.extract(args)

    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians, load_iteration=args.iteration,
                  shuffle=False, api_key=None, self_refinement=None,
                  num_prompt=None, max_rounds=None)
    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    print(f"  {gaussians.get_xyz.shape[0]} Gaussians, "
          f"{len(scene.getTrainCameras())} cameras")

    # ------------------------------------------------------------------
    # 2. Occupancy field + ground
    # ------------------------------------------------------------------
    print("[2] Building occupancy field ...")
    occ = OccupancyField().cuda()
    occ.fit(gaussians, num_steps=1000)

    planner = DiffPlanner(occ, device="cuda")
    ground_h = planner.estimate_ground(gaussians)

    # ------------------------------------------------------------------
    # 3. Sample positions on ground plane
    # ------------------------------------------------------------------
    print("[3] Sampling candidate positions on ground plane ...")

    # scene bounds
    bb_min = occ._bb_min.cpu()
    bb_max = occ._bb_max.cpu()
    margin = 0.3  # stay away from edges

    x_min = bb_min[0].item() + margin
    x_max = bb_max[0].item() - margin
    z_min = bb_min[2].item() + margin
    z_max = bb_max[2].item() - margin

    # Try a dense grid, then filter
    side = int(np.ceil(np.sqrt(args.num_views / args.grid_density)))
    xs = np.linspace(x_min, x_max, side)
    zs = np.linspace(z_min, z_max, side)

    candidates = []
    for x in xs:
        for z in zs:
            candidates.append([x, ground_h, z])

    if len(candidates) == 0:
        print("  ERROR: no candidates in bounds, check scene scale")
        sys.exit(1)

    # filter: keep only free-space positions
    cand_tensor = torch.tensor(candidates, dtype=torch.float).cuda()
    with torch.no_grad():
        occ_vals = occ.forward(cand_tensor).cpu().numpy().flatten()

    free_mask = occ_vals < 0.3
    free_positions = np.array(candidates)[free_mask]

    if len(free_positions) == 0:
        print("  WARNING: all candidates are occupied! Using closest to free.")
        free_positions = np.array(candidates)[np.argsort(occ_vals)[:args.num_views]]

    # take at most num_views, evenly spaced
    n_avail = len(free_positions)
    step = max(1, n_avail // args.num_views)
    selected = free_positions[::step][:args.num_views]

    print(f"  grid: {side}x{side} = {len(candidates)} positions")
    print(f"  free: {n_avail}, selected: {len(selected)}")

    # ------------------------------------------------------------------
    # 4. Render views
    # ------------------------------------------------------------------
    print("[4] Rendering candidate views ...")
    os.makedirs(args.output_dir, exist_ok=True)

    viz = PathVisualizer(gaussians, pipe, background, scene)
    metadata_lines = ["vp_idx\tx\ty\tz"]

    for i, pos in enumerate(tqdm(selected, desc="Rendering")):
        pos = pos.astype(float)
        # look toward scene center
        look_at = [0.0, ground_h, 0.0]

        img, _ = viz.render_view(pos, look_at, fov=args.fov,
                                  width=args.img_size, height=args.img_size)

        fname = f"vp_{i:04d}.png"
        torchvision.utils.save_image(img, os.path.join(args.output_dir, fname))
        metadata_lines.append(f"{i}\t{pos[0]:.4f}\t{pos[1]:.4f}\t{pos[2]:.4f}")

    with open(os.path.join(args.output_dir, "metadata.txt"), "w") as f:
        f.write("\n".join(metadata_lines))

    # Also render a reference top-down view showing the viewpoint locations
    try:
        top = viz.render_topdown_map(None,
              save_path=os.path.join(args.output_dir, "overview.png"))
    except Exception as e:
        print(f"  overview skipped: {e}")

    print("=" * 60)
    print(f"Done! {len(selected)} viewpoints in '{args.output_dir}/':")
    print(f"  vp_0000.png ~ vp_{len(selected)-1:04d}.png")
    print(f"  metadata.txt    — viewpoint index -> 3D position")
    print(f"  overview.png    — bird's-eye view of all positions")
    print()
    print("Next step — run path planning:")
    print(f"  python inference_nav.py -m {args.model_path} "
          f"-s {args.source_path} \\")
    print(f"      --start_vp <N> --goal_vp <M> --vp_dir {args.output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()

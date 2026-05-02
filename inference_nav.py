"""
Navigation in a trained DreamScene360 3DGS scene.

Three input modes (mix and match start/goal):
  A) --start_vp N --goal_vp M   从 render_viewpoints.py 的候选图里选
  B) --start_img A.jpg          拍照片定位
  C) --start_xyz "x g z"        直接写坐标 (g=地面高度)

Usage:
  # 先渲染候选视角: python render_viewpoints.py ...
  # 然后:
  python inference_nav.py -m output/Italy_output -s data/Italy_text \\
      --start_vp 3 --goal_vp 42 --vp_dir nav_candidates

  # 或者照片:
  python inference_nav.py -m output/Italy_output -s data/Italy_text \\
      --start_img start.jpg --goal_img goal.jpg

  # 或者坐标:
  python inference_nav.py -m output/Italy_output -s data/Italy_text \\
      --start_xyz "1.5 g 0.5" --goal_xyz "-1.0 g -1.5"
"""
import os
import sys
import torch
import numpy as np
import torchvision
from argparse import ArgumentParser

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from arguments import ModelParams, PipelineParams
from scene import Scene, GaussianModel
from gaussian_renderer import render
from navigation import OccupancyField, DiffPlanner, PathVisualizer
from navigation.localizer import VisualLocalizer


def parse_xyz(raw, ground_h):
    """Parse 'x y z', replace 'g'/'gnd' with ground height."""
    parts = raw.strip().split()
    assert len(parts) == 3, f"expected 3 values, got {raw}"
    out = []
    for p in parts:
        if p in ("g", "gnd"):
            out.append(float(ground_h))
        else:
            out.append(float(p))
    return out


def load_vp_position(vp_idx, vp_dir, ground_h):
    """Look up viewpoint position from metadata.txt."""
    meta_path = os.path.join(vp_dir, "metadata.txt")
    assert os.path.exists(meta_path), f"metadata not found: {meta_path}"
    with open(meta_path) as f:
        for line in f:
            parts = line.strip().split()
            if not parts or parts[0] == "vp_idx":
                continue
            idx = int(parts[0])
            if idx == vp_idx:
                pos = [float(parts[1]), float(parts[2]), float(parts[3])]
                pos[1] = ground_h  # override with current ground
                return pos
    raise ValueError(f"Viewpoint {vp_idx} not found in {meta_path}")


def resolve_position(start_src, goal_src, args, ground_h, scene):
    """Unified start/goal resolver supporting all three modes."""
    result = {}

    for label, src in [("start", start_src), ("goal", goal_src)]:
        if src is None:
            continue

        if src["mode"] == "xyz":
            pos = parse_xyz(src["val"], ground_h)
            print(f"  {label} from --{label}_xyz: {pos}")
            result[label] = torch.tensor(pos, dtype=torch.float).cuda()

        elif src["mode"] == "vp":
            pos = load_vp_position(int(src["val"]), args.vp_dir, ground_h)
            print(f"  {label} from --{label}_vp {src['val']}: {pos}")
            result[label] = torch.tensor(pos, dtype=torch.float).cuda()

        elif src["mode"] == "img":
            print(f"  localising --{label}_img: {src['val']} ...")
            # lazy init localizer (shared across both images)
            if "_localizer" not in result:
                loc = VisualLocalizer(device="cuda")
                loc.build_index(scene.getTrainCameras(), verbose=True)
                result["_localizer"] = loc
            loc = result["_localizer"]
            r = loc.localize(src["val"], forward_dist=args.forward_dist)
            pos = r["position"].copy()
            pos[1] = ground_h
            print(f"    matched camera #{r['camera_idx']} "
                  f"(sim={r['similarity']:.3f}) → {pos}")
            result[label] = torch.tensor(pos, dtype=torch.float).cuda()

    # clean up helper key
    result.pop("_localizer", None)
    return result["start"], result["goal"]


def main():
    parser = ArgumentParser(description="Navigation in DreamScene360 scene")
    model_params = ModelParams(parser)
    pipeline_params = PipelineParams(parser)

    parser.add_argument("--iteration", default=-1, type=int)

    # mode A: viewpoint indices
    parser.add_argument("--start_vp", type=int, default=None,
                        help="Start viewpoint index")
    parser.add_argument("--goal_vp", type=int, default=None,
                        help="Goal viewpoint index")
    parser.add_argument("--vp_dir", type=str, default="nav_candidates",
                        help="Directory with vp_*.png + metadata.txt")

    # mode B: coordinates
    parser.add_argument("--start_xyz", type=str, default=None,
                        help="Start 'x y z' (g = ground height)")
    parser.add_argument("--goal_xyz", type=str, default=None,
                        help="Goal 'x y z'")

    # mode C: photos
    parser.add_argument("--start_img", type=str, default=None)
    parser.add_argument("--goal_img", type=str, default=None)
    parser.add_argument("--forward_dist", type=float, default=2.0)

    # shared
    parser.add_argument("--num_waypoints", type=int, default=20)
    parser.add_argument("--opt_steps", type=int, default=800)
    parser.add_argument("--output_dir", type=str, default="nav_output")
    parser.add_argument("--fov", type=float, default=60)
    parser.add_argument("--skip_occ_fit", action="store_true")
    parser.add_argument("--occ_checkpoint", type=str, default=None)
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # 1. Load trained 3DGS model
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
    print(f"  iteration {scene.loaded_iter}, "
          f"{gaussians.get_xyz.shape[0]} Gaussians")

    # ------------------------------------------------------------------
    # 2. Occupancy field
    # ------------------------------------------------------------------
    print("=" * 60)
    print("[2] Building occupancy field ...")
    occ_field = OccupancyField().cuda()

    occ_ckpt = args.occ_checkpoint
    if occ_ckpt and os.path.exists(occ_ckpt):
        occ_field.load_state_dict(torch.load(occ_ckpt))
        print(f"  loaded from {occ_ckpt}")
    elif not args.skip_occ_fit:
        occ_field.fit(gaussians, num_steps=2000)
        if occ_ckpt:
            torch.save(occ_field.state_dict(), occ_ckpt)
    else:
        occ_field.fit(gaussians, num_steps=200)

    # ------------------------------------------------------------------
    # 3. Ground + planner
    # ------------------------------------------------------------------
    print("=" * 60)
    print("[3] Detecting ground plane ...")
    planner = DiffPlanner(occ_field, device="cuda")
    ground_h = planner.estimate_ground(gaussians)

    # ------------------------------------------------------------------
    # 4. Resolve start / goal (unified, all three modes)
    # ------------------------------------------------------------------
    print("=" * 60)
    print("[4] Resolving start / goal ...")

    # Build source descriptors for unified resolver
    def _src(mode, val):
        return {"mode": mode, "val": val}

    start_src = None
    goal_src = None

    for label, vp_arg, xyz_arg, img_arg in [
            ("start", args.start_vp, args.start_xyz, args.start_img),
            ("goal", args.goal_vp, args.goal_xyz, args.goal_img)]:
        n_modes = sum(x is not None for x in [vp_arg, xyz_arg, img_arg])
        assert n_modes <= 1, \
            f"{label}: use only one of --{label}_vp/--{label}_xyz/--{label}_img"
        if vp_arg is not None:
            if label == "start":
                start_src = _src("vp", vp_arg)
            else:
                goal_src = _src("vp", vp_arg)
        elif xyz_arg is not None:
            if label == "start":
                start_src = _src("xyz", xyz_arg)
            else:
                goal_src = _src("xyz", xyz_arg)
        elif img_arg is not None:
            if label == "start":
                start_src = _src("img", img_arg)
            else:
                goal_src = _src("img", img_arg)
        else:
            raise AssertionError(
                f"{label}: provide --{label}_vp, --{label}_xyz, or --{label}_img")

    start_pos, goal_pos = resolve_position(start_src, goal_src,
                                            args, ground_h, scene)

    # validate occupancy
    with torch.no_grad():
        start_occ = occ_field.forward(start_pos[None]).item()
        goal_occ = occ_field.forward(goal_pos[None]).item()
    print(f"  occupancy: start={start_occ:.3f}  goal={goal_occ:.3f}")
    if start_occ > 0.5:
        print("  WARNING: start is inside geometry!")
    if goal_occ > 0.5:
        print("  WARNING: goal is inside geometry!")

    # ------------------------------------------------------------------
    # 5. Path optimisation
    # ------------------------------------------------------------------
    print("=" * 60)
    print("[5] Optimising path (gradient descent, no A*/RRT) ...")
    path, losses = planner.plan(
        start_pos, goal_pos,
        num_waypoints=args.num_waypoints,
        lr=0.05, num_steps=args.opt_steps,
        w_collision=5.0, w_smooth=1.0, w_goal=2.0,
        w_length=0.5, w_floor=2.0, w_boundary=1.0,
    )

    # ------------------------------------------------------------------
    # 6. Save
    # ------------------------------------------------------------------
    os.makedirs(args.output_dir, exist_ok=True)

    path_np = path.cpu().numpy()
    np.savetxt(os.path.join(args.output_dir, "path_waypoints.txt"), path_np,
               header="x y z", fmt="%.6f")

    viz = PathVisualizer(gaussians, pipe, background, scene)
    viz.render_path_views(path, look_at=None, fov=args.fov,
                          save_dir=args.output_dir, prefix="nav")
    try:
        viz.render_topdown_map(path,
                               save_path=os.path.join(args.output_dir,
                                                      "topdown.png"))
    except Exception as e:
        print(f"  top-down skipped: {e}")

    print("=" * 60)
    print(f"Done. Output in '{args.output_dir}/':")
    print(f"  path_waypoints.txt    - 3D waypoints")
    print(f"  nav_0000.png ~ N.png  - path sequence")
    print(f"  topdown.png           - bird's-eye view")
    print("=" * 60)


if __name__ == "__main__":
    main()

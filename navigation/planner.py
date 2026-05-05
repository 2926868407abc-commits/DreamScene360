import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

from .path_init import build_initial_paths


class DiffPlanner:
    """Differentiable trajectory optimizer.

    Represents a path as learnable waypoints and optimises with gradient
    descent through the OccupancyField — no discrete search (A*/RRT).
    """

    def __init__(self, occupancy_field, ground_height=None, device="cuda"):
        self.occ = occupancy_field
        self.device = device
        self.ground_height = ground_height

    def _occupancy(self, xyz):
        if hasattr(self.occ, "occupancy"):
            return self.occ.occupancy(xyz)
        return self.occ.forward(xyz)

    def _bounds(self):
        if hasattr(self.occ, "bounds_min") and hasattr(self.occ, "bounds_max"):
            return self.occ.bounds_min, self.occ.bounds_max
        return self.occ._bb_min, self.occ._bb_max

    @staticmethod
    def _sample_segments(path, samples_per_segment):
        if samples_per_segment <= 0 or path.shape[0] < 2:
            return path

        alphas = torch.linspace(
            0, 1, samples_per_segment + 2,
            device=path.device, dtype=path.dtype
        )[1:-1]
        starts = path[:-1]
        ends = path[1:]
        samples = starts[:, None, :] + alphas[None, :, None] * (ends - starts)[:, None, :]
        return torch.cat([path, samples.reshape(-1, 3)], dim=0)

    # ------------------------------------------------------------------
    # Ground plane detection
    # ------------------------------------------------------------------
    def estimate_ground(self, gaussians):
        """Detect ground height from the 3DGS point cloud (bottom quantile mode)."""
        xyz = gaussians.get_xyz.detach()
        opacity = gaussians.get_opacity.detach().squeeze()

        high_conf = xyz[opacity > 0.1]
        if len(high_conf) < 1000:
            high_conf = xyz

        y_vals = high_conf[:, 1]
        bottom = y_vals[y_vals < y_vals.quantile(0.15)]
        if len(bottom) < 10:
            self.ground_height = y_vals.min().item()
        else:
            hist = torch.histc(bottom, bins=64)
            bin_idx = torch.argmax(hist)
            bins = torch.linspace(bottom.min(), bottom.max(), 65, device=self.device)
            self.ground_height = (bins[bin_idx] + bins[bin_idx + 1]).item() * 0.5

        print(f"[DiffPlanner] estimated ground_height = {self.ground_height:.4f}")
        return self.ground_height

    # ------------------------------------------------------------------
    # Path optimization
    # ------------------------------------------------------------------
    def plan(self, start_pos, goal_pos,
             num_waypoints=16,
             lr=0.05,
             num_steps=800,
             w_collision=5.0,
             w_smooth=1.0,
             w_goal=2.0,
             w_length=0.5,
             w_floor=2.0,
             w_boundary=1.0,
             samples_per_segment=4,
             init_path=None,
             progress_bar=True):
        """Optimise a collision-free path from start to goal.

        The path is initialised as a straight line between start and goal,
        then perturbed by learnable offsets.  All loss components are
        differentiable wrt the waypoint positions.

        Args:
            start_pos: [3] tensor, start position.
            goal_pos:  [3] tensor, goal position.
            num_waypoints: number of path points (including start & goal).
            lr: learning rate.
            num_steps: gradient descent iterations.
            w_*: loss weights.
        Returns:
            path: [num_waypoints, 3] optimised waypoints.
            losses: dict of loss histories.
        """
        start_pos = start_pos.to(self.device).float()
        goal_pos = goal_pos.to(self.device).float()

        # --- path initialisation ---
        if init_path is None:
            alphas = torch.linspace(0, 1, num_waypoints, device=self.device)
            init_path = start_pos[None] + alphas[:, None] * (goal_pos - start_pos)[None]
        else:
            init_path = init_path.to(self.device).float()
            num_waypoints = init_path.shape[0]

        # learnable offsets (start & goal pinned to 0)
        perturb = nn.Parameter(torch.zeros_like(init_path))
        fix_start = [0]
        fix_goal = [num_waypoints - 1]

        opt = torch.optim.Adam([perturb], lr=lr)

        # scene bounds from the occupancy field
        bb_min, bb_max = self._bounds()

        losses = {"total": [], "collision": [], "smooth": [],
                  "goal": [], "length": [], "floor": [], "boundary": []}

        iters = tqdm(range(num_steps), desc="Path optim") if progress_bar else range(num_steps)

        for step in iters:
            # pin start / goal
            with torch.no_grad():
                perturb.data[fix_start] = 0.
                perturb.data[fix_goal] = 0.

            path = init_path + perturb  # [N, 3]

            sampled_path = self._sample_segments(path, samples_per_segment)

            # 1. collision
            occ = self._occupancy(sampled_path)
            loss_collision = F.relu(occ - 0.3).mean()

            # 2. smoothness (second-order diff)
            vel = path[1:] - path[:-1]
            acc = vel[1:] - vel[:-1]
            loss_smooth = acc.square().mean()

            # 3. goal-reaching
            loss_goal = (path[-1] - goal_pos).norm()

            # 4. path length
            loss_length = vel.norm(dim=1).mean()

            # 5. floor alignment
            if self.ground_height is not None:
                loss_floor = (sampled_path[:, 1] - self.ground_height).abs().mean()
            else:
                loss_floor = torch.tensor(0.0, device=self.device)

            # 6. boundary penalty
            loss_boundary = (
                F.relu(bb_min - path).sum() + F.relu(path - bb_max).sum()
            ) / path.numel()

            loss = (w_collision * loss_collision +
                    w_smooth * loss_smooth +
                    w_goal * loss_goal +
                    w_length * loss_length +
                    w_floor * loss_floor +
                    w_boundary * loss_boundary)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            losses["total"].append(loss.item())
            losses["collision"].append(loss_collision.item())
            losses["smooth"].append(loss_smooth.item())
            losses["goal"].append(loss_goal.item())
            losses["length"].append(loss_length.item())
            losses["floor"].append(loss_floor.item())
            losses["boundary"].append(loss_boundary.item())

            if progress_bar:
                iters.set_postfix({"loss": f"{loss.item():.4f}",
                                   "coll": f"{loss_collision.item():.4f}"})

        final_path = (init_path + perturb).detach()
        print(f"[DiffPlanner] path optimised ({final_path.shape[0]} waypoints), "
              f"final loss = {loss.item():.4f}")
        return final_path, losses


class MultiInitDiffPlanner:
    """Run several local trajectory optimisations and select the best path."""

    def __init__(self, collision_field, ground_height=None, device="cuda"):
        self.local_planner = DiffPlanner(
            collision_field, ground_height=ground_height, device=device
        )
        self.device = device

    @property
    def ground_height(self):
        return self.local_planner.ground_height

    @ground_height.setter
    def ground_height(self, value):
        self.local_planner.ground_height = value

    def estimate_ground(self, gaussians):
        return self.local_planner.estimate_ground(gaussians)

    def _score_path(self, path, samples_per_segment):
        with torch.no_grad():
            sampled = self.local_planner._sample_segments(path, samples_per_segment)
            occ = self.local_planner._occupancy(sampled)
            collision = F.relu(occ - 0.3).mean().item()
            max_occ = occ.max().item()
            length = (path[1:] - path[:-1]).norm(dim=1).sum().item()
            if path.shape[0] > 2:
                vel = path[1:] - path[:-1]
                smooth = (vel[1:] - vel[:-1]).square().mean().item()
            else:
                smooth = 0.0
        return {
            "total": collision * 100.0 + max_occ * 10.0 + length + smooth,
            "collision": collision,
            "max_occ": max_occ,
            "length": length,
            "smooth": smooth,
        }

    def plan(self, start_pos, goal_pos,
             num_waypoints=16,
             lr=0.05,
             num_steps=800,
             w_collision=5.0,
             w_smooth=1.0,
             w_goal=2.0,
             w_length=0.5,
             w_floor=2.0,
             w_boundary=1.0,
             samples_per_segment=4,
             random_inits=2,
             progress_bar=True):
        start_pos = start_pos.to(self.device).float()
        goal_pos = goal_pos.to(self.device).float()

        init_paths = build_initial_paths(
            start_pos, goal_pos, num_waypoints,
            ground_height=self.ground_height,
            random_count=random_inits,
        )

        candidates = []
        for idx, init_path in enumerate(init_paths):
            print(f"[MultiInitDiffPlanner] optimising candidate {idx + 1}/{len(init_paths)}")
            path, losses = self.local_planner.plan(
                start_pos, goal_pos,
                num_waypoints=num_waypoints,
                lr=lr,
                num_steps=num_steps,
                w_collision=w_collision,
                w_smooth=w_smooth,
                w_goal=w_goal,
                w_length=w_length,
                w_floor=w_floor,
                w_boundary=w_boundary,
                samples_per_segment=samples_per_segment,
                init_path=init_path,
                progress_bar=progress_bar,
            )
            score = self._score_path(path, samples_per_segment)
            candidates.append((score, path, losses))
            print("  score: "
                  f"total={score['total']:.4f}, coll={score['collision']:.4f}, "
                  f"max_occ={score['max_occ']:.4f}, len={score['length']:.4f}")

        best_score, best_path, best_losses = min(
            candidates, key=lambda item: item[0]["total"]
        )
        best_losses = dict(best_losses)
        best_losses["selection_score"] = best_score
        print("[MultiInitDiffPlanner] selected path: "
              f"coll={best_score['collision']:.4f}, "
              f"max_occ={best_score['max_occ']:.4f}, "
              f"length={best_score['length']:.4f}")
        return best_path, best_losses

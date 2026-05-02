import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm


class DiffPlanner:
    """Differentiable trajectory optimizer.

    Represents a path as learnable waypoints and optimises with gradient
    descent through the OccupancyField — no discrete search (A*/RRT).
    """

    def __init__(self, occupancy_field, ground_height=None, device="cuda"):
        self.occ = occupancy_field
        self.device = device
        self.ground_height = ground_height

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

        # --- straight-line initialisation ---
        alphas = torch.linspace(0, 1, num_waypoints, device=self.device)
        init_path = start_pos[None] + alphas[:, None] * (goal_pos - start_pos)[None]

        # learnable offsets (start & goal pinned to 0)
        perturb = nn.Parameter(torch.zeros_like(init_path))
        fix_start = [0]
        fix_goal = [num_waypoints - 1]

        opt = torch.optim.Adam([perturb], lr=lr)

        # scene bounds from the occupancy field
        bb_min = self.occ._bb_min
        bb_max = self.occ._bb_max

        losses = {"total": [], "collision": [], "smooth": [],
                  "goal": [], "length": [], "floor": [], "boundary": []}

        iters = tqdm(range(num_steps), desc="Path optim") if progress_bar else range(num_steps)

        for step in iters:
            # pin start / goal
            with torch.no_grad():
                perturb.data[fix_start] = 0.
                perturb.data[fix_goal] = 0.

            path = init_path + perturb  # [N, 3]

            # 1. collision
            occ = self.occ.forward(path)
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
                loss_floor = (path[:, 1] - self.ground_height).abs().mean()
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

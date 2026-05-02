import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import tinycudann as tcnn


class OccupancyField(nn.Module):
    """Neural implicit occupancy field built from 3DGS point cloud.

    Architecture (same hash-grid pattern as GeometricField):
      (x,y,z) -> [normalize] -> HashGrid -> MLP -> occupancy probability

    Fully differentiable -> gradient flows from path planning into occupancy.
    """

    def __init__(self,
                 n_levels=16,
                 log2_hashmap_size=19,
                 base_res=16,
                 fine_res=2048):
        super().__init__()
        per_level_scale = np.exp(np.log(fine_res / base_res) / (n_levels - 1))
        self.hash_grid = tcnn.Encoding(
            n_input_dims=3,
            encoding_config={
                "otype": "HashGrid",
                "n_levels": n_levels,
                "n_features_per_level": 2,
                "log2_hashmap_size": log2_hashmap_size,
                "base_resolution": base_res,
                "per_level_scale": per_level_scale,
                "interpolation": "Smoothstep",
            }
        )
        self.mlp = nn.Sequential(
            nn.Linear(n_levels * 2, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

        # scene bounds (set during fit)
        self.register_buffer("_bb_min", torch.zeros(3))
        self.register_buffer("_bb_max", torch.ones(3))

    # ------------------------------------------------------------------
    # Normalization helpers
    # ------------------------------------------------------------------
    def _normalize(self, xyz):
        """Map scene coordinates -> [0, 1] using stored bounds."""
        return (xyz - self._bb_min) / (self._bb_max - self._bb_min + 1e-8)

    def _unnormalize(self, xyz_norm):
        """Map [0, 1] -> scene coordinates."""
        return xyz_norm * (self._bb_max - self._bb_min) + self._bb_min

    def forward(self, xyz):
        """Query occupancy probability.

        Args:
            xyz: [..., 3] 3D positions in **scene** coordinate space.
        Returns:
            prob: [...] occupancy in [0, 1].
        """
        orig_shape = xyz.shape[:-1]
        flat = xyz.reshape(-1, 3)
        xyz_norm = self._normalize(flat)           # [0, 1]
        xyz_hash = xyz_norm * 0.98 + 0.01          # [0.01, 0.99] for hash grid
        h = self.hash_grid(xyz_hash).float()
        logits = self.mlp(h)
        prob = torch.sigmoid(logits).view(*orig_shape)
        return prob

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    @torch.no_grad()
    def fit(self, gaussians, num_steps=2000, batch_size=32768,
            scene_margin=0.1, occ_threshold=0.05):
        """Train occupancy field from the 3DGS Gaussian point cloud.

        Args:
            gaussians: trained GaussianModel instance.
            num_steps: training iterations.
            batch_size: per-step sample count.
            scene_margin: fraction padding added around scene bounds.
            occ_threshold: minimum opacity to count as occupied.
        """
        device = gaussians.get_xyz.device
        xyz = gaussians.get_xyz.detach()
        opacity = gaussians.get_opacity.detach()

        # --- compute / store scene bounds (with margin) ---
        scene_min = xyz.min(dim=0)[0]
        scene_max = xyz.max(dim=0)[0]
        extent = scene_max - scene_min
        self._bb_min = scene_min - extent * scene_margin
        self._bb_max = scene_max + extent * scene_margin
        self._bb_min = self._bb_min.to(device)
        self._bb_max = self._bb_max.to(device)

        # --- occupied points ---
        occ_mask = (opacity > occ_threshold).squeeze()
        occ_pts = xyz[occ_mask]
        if occ_pts.shape[0] == 0:
            print("[OccupancyField] WARNING: no points above threshold, using all.")
            occ_pts = xyz

        n_occ = occ_pts.shape[0]
        print(f"[OccupancyField] occupied: {n_occ} / {xyz.shape[0]} points, "
              f"bounds: [{self._bb_min.cpu().numpy()}] "
              f"-> [{self._bb_max.cpu().numpy()}]")

        opt = torch.optim.Adam(self.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, num_steps)

        half_batch = batch_size // 2
        bs = min(half_batch, n_occ)

        for step in range(num_steps):
            # occupied
            idx = torch.randint(0, n_occ, (bs,), device=device)
            occ_batch = occ_pts[idx]

            # free (uniform in bounding box)
            free_batch = torch.rand(bs, 3, device=device)
            free_batch = free_batch * (self._bb_max - self._bb_min) + self._bb_min

            xyz_batch = torch.cat([occ_batch, free_batch], dim=0)
            target = torch.cat([
                torch.ones(bs, 1, device=device),
                torch.zeros(bs, 1, device=device),
            ], dim=0)

            prob = self.forward(xyz_batch)
            loss = F.binary_cross_entropy(prob, target, reduction='mean')

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            scheduler.step()

            if step % 500 == 0:
                print(f"  [step {step:5d}] occupancy_loss = {loss.item():.6f}")

        print("[OccupancyField] done.")
        return self

    # ------------------------------------------------------------------
    # Differentiable cost for planner
    # ------------------------------------------------------------------
    def collision_cost(self, positions, margin=0.3):
        """Differentiable collision penalty.

        Args:
            positions: [N, 3] query points.
            margin: positions with occupancy > margin get penalised.
        Returns:
            scalar cost.
        """
        prob = self.forward(positions)
        return F.relu(prob - margin).mean()

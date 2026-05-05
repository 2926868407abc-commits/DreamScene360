import torch

from .occupancy_field import OccupancyField


class CollisionField:
    """Common planner-facing collision interface.

    Geometry sources can be Gaussian points, depth back-projections, point
    clouds, or a hybrid of several fields. Planners should depend on this
    interface instead of on a specific geometry source.
    """

    def occupancy(self, xyz):
        raise NotImplementedError

    def distance(self, xyz):
        return None

    @property
    def bounds_min(self):
        raise NotImplementedError

    @property
    def bounds_max(self):
        raise NotImplementedError

    def forward(self, xyz):
        return self.occupancy(xyz)


class GaussianCollisionField(CollisionField):
    """Collision field built from DreamScene360 Gaussian geometry."""

    def __init__(self, occupancy_field):
        self.occ = occupancy_field

    @classmethod
    def from_gaussians(cls, gaussians, num_steps=2000, checkpoint=None,
                       skip_fit=False, device="cuda"):
        occ = OccupancyField().to(device)

        if checkpoint is not None:
            state = torch.load(checkpoint, map_location=device)
            occ.load_state_dict(state)
            print(f"  loaded occupancy field from {checkpoint}")
        elif skip_fit:
            occ.fit(gaussians, num_steps=200)
        else:
            occ.fit(gaussians, num_steps=num_steps)

        return cls(occ)

    def occupancy(self, xyz):
        return self.occ.forward(xyz)

    @property
    def bounds_min(self):
        return self.occ._bb_min

    @property
    def bounds_max(self):
        return self.occ._bb_max

    def state_dict(self):
        return self.occ.state_dict()


class DepthCollisionField(CollisionField):
    """Placeholder for G2VLM monocular depth back-projection.

    This class deliberately exposes the same interface as GaussianCollisionField
    so G2VLM can be removed or swapped without touching the planner. The first
    implementation will fuse back-projected depth points into a voxel/ESDF
    field; until then it should not be selected as the only geometry source.
    """

    def __init__(self, points=None, device="cuda"):
        self.device = device
        self.points = points
        if points is None or len(points) == 0:
            self._bb_min = torch.zeros(3, device=device)
            self._bb_max = torch.ones(3, device=device)
        else:
            points = torch.as_tensor(points, dtype=torch.float32, device=device)
            self.points = points
            self._bb_min = points.min(dim=0)[0]
            self._bb_max = points.max(dim=0)[0]

    def occupancy(self, xyz):
        raise NotImplementedError(
            "DepthCollisionField needs G2VLM depth back-projection/voxel fusion "
            "before it can answer occupancy queries."
        )

    @property
    def bounds_min(self):
        return self._bb_min

    @property
    def bounds_max(self):
        return self._bb_max


class HybridCollisionField(CollisionField):
    """Combine DreamScene360 base geometry with optional depth constraints."""

    def __init__(self, base_field, depth_field=None, mode="max"):
        self.base_field = base_field
        self.depth_field = depth_field
        self.mode = mode

    def occupancy(self, xyz):
        base_occ = self.base_field.occupancy(xyz)
        if self.depth_field is None:
            return base_occ

        depth_occ = self.depth_field.occupancy(xyz)
        if self.mode == "max":
            return torch.maximum(base_occ, depth_occ)
        if self.mode == "mean":
            return 0.5 * (base_occ + depth_occ)
        raise ValueError(f"Unknown hybrid combine mode: {self.mode}")

    @property
    def bounds_min(self):
        if self.depth_field is None:
            return self.base_field.bounds_min
        return torch.minimum(self.base_field.bounds_min, self.depth_field.bounds_min)

    @property
    def bounds_max(self):
        if self.depth_field is None:
            return self.base_field.bounds_max
        return torch.maximum(self.base_field.bounds_max, self.depth_field.bounds_max)

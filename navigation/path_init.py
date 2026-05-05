import math
import torch


def _safe_perp(direction):
    horizontal = direction.clone()
    horizontal[1] = 0.0
    norm = horizontal.norm()
    if norm < 1e-6:
        return torch.tensor([1.0, 0.0, 0.0], device=direction.device)
    horizontal = horizontal / norm
    return torch.tensor([-horizontal[2], 0.0, horizontal[0]], device=direction.device)


def straight_path(start, goal, num_waypoints):
    alphas = torch.linspace(0, 1, num_waypoints, device=start.device)
    return start[None] + alphas[:, None] * (goal - start)[None]


def arc_path(start, goal, num_waypoints, side=1.0, height=0.0, strength=0.35):
    base = straight_path(start, goal, num_waypoints)
    delta = goal - start
    length = delta.norm().clamp_min(1e-6)
    perp = _safe_perp(delta)
    alphas = torch.linspace(0, 1, num_waypoints, device=start.device)
    envelope = torch.sin(alphas * math.pi)
    offset = side * strength * length * envelope[:, None] * perp[None]
    offset[:, 1] += height * envelope
    return base + offset


def midpoint_path(start, goal, num_waypoints, offset_vec):
    alphas = torch.linspace(0, 1, num_waypoints, device=start.device)
    base = straight_path(start, goal, num_waypoints)
    envelope = torch.sin(alphas * math.pi)
    return base + envelope[:, None] * offset_vec[None]


def build_initial_paths(start, goal, num_waypoints, ground_height=None,
                        random_count=2):
    """Create several smooth initial trajectories without global search."""
    start = start.float()
    goal = goal.float()
    delta = goal - start
    length = delta.norm().clamp_min(1e-6)
    perp = _safe_perp(delta)
    paths = [
        straight_path(start, goal, num_waypoints),
        arc_path(start, goal, num_waypoints, side=1.0, strength=0.35),
        arc_path(start, goal, num_waypoints, side=-1.0, strength=0.35),
        arc_path(start, goal, num_waypoints, side=1.0, strength=0.6),
        arc_path(start, goal, num_waypoints, side=-1.0, strength=0.6),
    ]

    if ground_height is not None:
        high = max(0.15, 0.15 * float(length.detach().cpu()))
        paths.append(arc_path(start, goal, num_waypoints, side=0.0, height=high))

    for i in range(random_count):
        sign = 1.0 if i % 2 == 0 else -1.0
        mag = (0.25 + 0.15 * i) * length
        paths.append(midpoint_path(start, goal, num_waypoints, sign * mag * perp))

    return paths

import torch

from utils.graphics_utils import fov2focal


def align_depth_scale(pred_depth, ref_depth, mask=None, eps=1e-6):
    """Scale-align monocular depth to a reference depth map by median ratio."""
    pred_depth = pred_depth.float()
    ref_depth = ref_depth.float()
    valid = torch.isfinite(pred_depth) & torch.isfinite(ref_depth)
    valid = valid & (pred_depth > eps) & (ref_depth > eps)
    if mask is not None:
        valid = valid & mask.bool()

    if valid.sum() < 16:
        return pred_depth, torch.tensor(1.0, device=pred_depth.device)

    scale = torch.median(ref_depth[valid] / pred_depth[valid].clamp_min(eps))
    return pred_depth * scale, scale


def backproject_depth_to_camera(depth, fovx, fovy=None):
    """Back-project a depth map into OpenCV-style camera coordinates."""
    if fovy is None:
        fovy = fovx

    h, w = depth.shape[-2:]
    device = depth.device
    dtype = depth.dtype
    fx = fov2focal(float(fovx), w)
    fy = fov2focal(float(fovy), h)

    ys, xs = torch.meshgrid(
        torch.arange(h, device=device, dtype=dtype),
        torch.arange(w, device=device, dtype=dtype),
        indexing="ij",
    )
    cx = (w - 1) * 0.5
    cy = (h - 1) * 0.5
    z = depth
    x = (xs - cx) / fx * z
    y = (ys - cy) / fy * z
    return torch.stack([x, y, z], dim=-1)


def camera_points_to_world(points_cam, camera):
    """Transform camera-space points to DreamScene360 world coordinates."""
    shape = points_cam.shape
    points = points_cam.reshape(-1, 3)
    ones = torch.ones(points.shape[0], 1, device=points.device, dtype=points.dtype)
    points_h = torch.cat([points, ones], dim=-1)

    w2c = camera.world_view_transform
    if w2c.device != points.device:
        w2c = w2c.to(points.device)
    c2w = torch.inverse(w2c)
    world = points_h @ c2w
    return world[:, :3].reshape(*shape)


def backproject_depth_to_world(depth, camera, mask=None):
    """Back-project a depth map from a DreamScene360 camera into world points."""
    points_cam = backproject_depth_to_camera(depth, camera.FoVx, camera.FoVy)
    points_world = camera_points_to_world(points_cam, camera)
    if mask is not None:
        return points_world[mask.bool()]
    return points_world.reshape(-1, 3)

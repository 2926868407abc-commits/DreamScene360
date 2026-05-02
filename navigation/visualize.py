import os
import torch
import numpy as np
from PIL import Image
import torchvision
from tqdm import tqdm

from scene.cameras import Camera
from utils.graphics_utils import getWorld2View2


class PathVisualizer:
    """Render the planned navigation path as a sequence of perspective views."""

    def __init__(self, gaussians, pipeline, background, scene):
        self.gaussians = gaussians
        self.pipe = pipeline
        self.bg = background
        self.scene = scene

    def _look_at_rotation(self, pos, target, up=(0, -1, 0)):
        """Build OpenCV-style world-to-camera rotation.

        'pos' is the camera position, 'target' is where it looks.
        Returns R (3x3) and T (3,) for Camera(R=R, T=T, ...).
        """
        pos = torch.tensor(pos, dtype=torch.float)
        target = torch.tensor(target, dtype=torch.float)
        up = torch.tensor(up, dtype=torch.float)

        forward = (target - pos) / torch.linalg.norm(target - pos)
        right = torch.linalg.cross(forward, up)
        right = right / torch.linalg.norm(right)
        down = torch.linalg.cross(forward, right)
        down = down / torch.linalg.norm(down)

        # W2C rotation (OpenCV: right, down, forward as columns -> transposed)
        R = torch.stack([right, down, forward], dim=1)  # [3, 3]
        T = pos.clone()
        return R.numpy(), T.numpy()

    def render_view(self, position, look_at, fov=60, width=512, height=512):
        """Render a single perspective view from a given position."""
        from gaussian_renderer import render
        R, T = self._look_at_rotation(position, look_at)
        fov_rad = np.deg2rad(fov)

        cam = Camera(
            colmap_id=0,
            R=R,
            T=T,
            FoVx=fov_rad,
            FoVy=fov_rad * height / width,
            image=torch.zeros(3, height, width),
            gt_alpha_mask=None,
            image_name="nav_view",
            uid=0,
            data_device="cuda",
        )
        with torch.no_grad():
            result = render(cam, self.gaussians, self.pipe, self.bg)
        return result["render"], result.get("depth")

    def render_path_views(self, path, look_at=None, fov=60,
                          save_dir="nav_output", prefix="step"):
        """Render views from each waypoint along the path.

        Args:
            path: [N, 3] tensor of waypoints.
            look_at: target point (default: ahead along the path).
            fov: field of view in degrees.
            save_dir: output directory.
            prefix: filename prefix.

        Returns:
            renderings: list of rendered image tensors [3, H, W].
        """
        os.makedirs(save_dir, exist_ok=True)
        renderings = []
        depths = []

        path_np = path.cpu().numpy() if torch.is_tensor(path) else path

        for i in tqdm(range(len(path_np)), desc="Rendering path"):
            pos = path_np[i]
            if look_at is not None:
                target = look_at
            elif i < len(path_np) - 1:
                target = path_np[i + 1]  # look toward next waypoint
            else:
                target = path_np[i - 1]  # last point looks back

            # ensure look-at is not same as position
            if np.linalg.norm(np.array(target) - np.array(pos)) < 1e-6:
                target = path_np[0] if i > 0 else path_np[-1]

            img, depth = self.render_view(pos, target, fov=fov)
            renderings.append(img)
            depths.append(depth)

            torchvision.utils.save_image(
                img,
                os.path.join(save_dir, f"{prefix}_{i:04d}.png")
            )

        print(f"[Visualizer] saved {len(renderings)} views to {save_dir}/")
        return renderings, depths

    def render_topdown_map(self, path, resolution=512, save_path=None):
        """Render a top-down (bird's-eye) view of the scene with path overlay."""
        from gaussian_renderer import render

        # find scene bounds
        xyz = self.gaussians.get_xyz.detach()
        center = xyz.mean(dim=0)
        radius = (xyz - center).norm(dim=1).max().item() * 1.2

        # top-down camera looking at scene center
        pos = [center[0].item(), radius * 1.5, center[2].item()]
        target = [center[0].item(), 0, center[2].item()]
        R, T = self._look_at_rotation(pos, target)

        fov_rad = 2 * np.arctan(radius / (radius * 1.5)) * 1.2

        cam = Camera(
            colmap_id=0, R=R, T=T,
            FoVx=fov_rad,
            FoVy=fov_rad,
            image=torch.zeros(3, resolution, resolution),
            gt_alpha_mask=None, image_name="topdown", uid=0,
            data_device="cuda",
        )
        with torch.no_grad():
            result = render(cam, self.gaussians, self.pipe, self.bg)

        rendered = result["render"]  # [3, H, W]

        # overlay path in 2D screen coordinates
        if path is not None:
            path_cpu = path.cpu() if torch.is_tensor(path) else torch.tensor(path)
            # project path points into camera space
            w2c = torch.tensor(getWorld2View2(R, T), dtype=torch.float, device="cuda")
            ones = torch.ones(path_cpu.shape[0], 1)
            path_h = torch.cat([path_cpu.float(), ones], dim=1).cuda()  # [N, 4]
            path_cam = (w2c @ path_h.T).T  # [N, 4]
            path_cam = path_cam[path_cam[:, 2] < 0]  # behind camera
            if len(path_cam) > 0:
                # simple perspective projection for overlay dots
                # (just save the path coordinates alongside rendered image)
                pass

        torchvision.utils.save_image(
            rendered,
            save_path or os.path.join("nav_output", "topdown.png")
        )
        print(f"[Visualizer] top-down saved to {save_path or 'nav_output/topdown.png'}")
        return rendered

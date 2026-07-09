import json
import os
import shlex
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.transforms.functional import to_pil_image

from .geo_predictor import GeoPredictor


def load_depth_file(path: Path) -> torch.Tensor:
    if path.suffix == ".npy":
        depth = np.load(path)
    elif path.suffix == ".npz":
        data = np.load(path)
        key = "depth" if "depth" in data else data.files[0]
        depth = data[key]
    else:
        depth = np.asarray(Image.open(path)).astype(np.float32)
        if depth.max() > 1.0:
            depth = depth / 255.0

    depth = np.asarray(depth).squeeze().astype(np.float32)
    return torch.from_numpy(depth)[None, None]


class ExternalDepthCommandPredictor(GeoPredictor):
    def __init__(self, name, command_env, command=None, root=None, model_path=None):
        super().__init__()
        self.name = name
        self.command = command or os.getenv(command_env)
        self.root = Path(root).resolve() if root else None
        self.model_path = Path(model_path).resolve() if model_path else None

        if not self.command:
            raise RuntimeError(
                f"{name} depth predictor needs a command template. "
                f"Set {command_env} or pass the corresponding command argument. "
                "The command may use {input}, {output}, {intrinsics}, {root}, and {model_path}."
            )

    @torch.no_grad()
    def predict_depth(self, img, **kwargs):
        if img.dim() != 4 or img.shape[0] != 1:
            raise ValueError(f"{self.name} expects a [1, 3, H, W] tensor.")

        target_hw = img.shape[-2:]
        with tempfile.TemporaryDirectory(prefix=f"{self.name.lower()}_depth_") as tmp:
            tmp_dir = Path(tmp)
            input_path = tmp_dir / "input.png"
            output_path = tmp_dir / "depth.npy"
            intrinsics_path = tmp_dir / "intrinsics.json"

            to_pil_image(img[0].detach().clamp(0.0, 1.0).cpu()).save(input_path)
            intrinsics_path.write_text(json.dumps(kwargs.get("intri", {})), encoding="utf-8")

            rendered = self.command.format(
                input=str(input_path),
                output=str(output_path),
                intrinsics=str(intrinsics_path),
                root=str(self.root or ""),
                model_path=str(self.model_path or ""),
            )
            subprocess.run(
                shlex.split(rendered),
                cwd=str(self.root) if self.root else None,
                check=True,
            )

            depth = load_depth_file(output_path).float().clip(0.0, None)

        if depth.shape[-2:] != target_hw:
            depth = F.interpolate(depth, target_hw, mode="bilinear", align_corners=False)
        return depth.cuda()


class DAPPredictor(ExternalDepthCommandPredictor):
    def __init__(self, root=None, model_path=None, command=None):
        super().__init__(
            name="DAP",
            command_env="DAP_DEPTH_COMMAND",
            command=command,
            root=root or os.getenv("DAP_ROOT"),
            model_path=model_path or os.getenv("DAP_MODEL_PATH"),
        )

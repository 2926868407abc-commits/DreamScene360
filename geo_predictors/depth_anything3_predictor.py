import os
import shlex
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torchvision.transforms.functional import to_pil_image

from .geo_predictor import GeoPredictor


class DepthAnything3Predictor(GeoPredictor):
    def __init__(self, model_id=None, command=None):
        super().__init__()
        self.model_id = model_id or os.getenv("DEPTH_ANYTHING3_MODEL") or "depth-anything/DA3-LARGE-1.1"
        self.command = command or os.getenv("DEPTH_ANYTHING3_COMMAND")
        if not self.model_id:
            raise RuntimeError(
                "Depth Anything 3 model id is not set. "
                "Pass --depth_anything3_model or set DEPTH_ANYTHING3_MODEL."
            )
        if self.model_id == "YOUR_DEPTH_ANYTHING3_MODEL_ID":
            raise RuntimeError(
                "Replace YOUR_DEPTH_ANYTHING3_MODEL_ID with a real Depth Anything 3 model id, "
                "for example depth-anything/DA3-LARGE-1.1."
            )

        if self.command:
            self.model = None
            return

        try:
            from depth_anything_3.api import DepthAnything3
        except Exception as exc:
            raise RuntimeError(
                "Depth Anything 3 predictor needs the official depth_anything_3 package. "
                "Install it from https://github.com/ByteDance-Seed/depth-anything-3."
            ) from exc

        self.model = DepthAnything3.from_pretrained(self.model_id).cuda()

    @torch.no_grad()
    def predict_depth(self, img, **kwargs):
        return self.predict_depth_batch(img, intrinsics=[kwargs.get("intri", {})])

    @torch.no_grad()
    def predict_depth_batch(self, imgs, intrinsics=None):
        if imgs.dim() != 4:
            raise ValueError("DepthAnything3Predictor expects a [N, 3, H, W] tensor.")
        if self.command:
            return self._predict_depth_batch_external(imgs)

        target_hw = imgs.shape[-2:]
        pil_imgs = [
            to_pil_image(image.detach().clamp(0.0, 1.0).cpu())
            for image in imgs
        ]
        prediction = self.model.inference(pil_imgs)
        depth = prediction.depth
        if not torch.is_tensor(depth):
            depth = torch.from_numpy(depth)
        depth = depth.float()
        if depth.dim() == 3:
            depth = depth[:, None]
        elif depth.dim() == 2:
            depth = depth[None, None]

        depth = depth.cuda().clip(0.0, None)
        if depth.shape[-2:] != target_hw:
            depth = F.interpolate(depth, target_hw, mode="bilinear", align_corners=False)
        return depth

    def _predict_depth_batch_external(self, imgs):
        target_hw = imgs.shape[-2:]
        with tempfile.TemporaryDirectory(prefix="depth_anything3_") as tmp:
            tmp_dir = Path(tmp)
            input_dir = tmp_dir / "inputs"
            output_dir = tmp_dir / "outputs"
            input_dir.mkdir()
            output_dir.mkdir()

            for index, image in enumerate(imgs):
                path = input_dir / f"{index:06d}.png"
                to_pil_image(image.detach().clamp(0.0, 1.0).cpu()).save(path)

            rendered = self.command.format(
                input_dir=str(input_dir),
                output_dir=str(output_dir),
                model_id=self.model_id,
            )
            subprocess.run(shlex.split(rendered), check=True)

            depths = []
            for index in range(imgs.shape[0]):
                depth_path = output_dir / f"{index:06d}.npy"
                if not depth_path.exists():
                    raise RuntimeError(f"Depth Anything 3 command did not write {depth_path}.")
                depth = np.load(depth_path).squeeze().astype(np.float32)
                depths.append(torch.from_numpy(depth)[None, None])

        depth = torch.cat(depths, dim=0).float().clip(0.0, None)
        if depth.shape[-2:] != target_hw:
            depth = F.interpolate(depth, target_hw, mode="bilinear", align_corners=False)
        return depth.cuda()


class HuggingFaceDepthPredictor(GeoPredictor):
    def __init__(self, model_id):
        super().__init__()
        if not model_id:
            raise RuntimeError("HuggingFaceDepthPredictor requires a model id.")
        try:
            from transformers import AutoImageProcessor, AutoModelForDepthEstimation
        except Exception as exc:
            raise RuntimeError("This predictor needs transformers depth-estimation support.") from exc

        self.processor = AutoImageProcessor.from_pretrained(model_id, trust_remote_code=True)
        self.model = AutoModelForDepthEstimation.from_pretrained(model_id, trust_remote_code=True).cuda().eval()

    @torch.no_grad()
    def predict_depth(self, img, **kwargs):
        target_hw = img.shape[-2:]
        pil_img = to_pil_image(img[0].detach().clamp(0.0, 1.0).cpu())
        inputs = self.processor(images=pil_img, return_tensors="pt")
        inputs = {key: value.cuda() for key, value in inputs.items()}
        outputs = self.model(**inputs)

        depth = getattr(outputs, "predicted_depth", None)
        if depth is None:
            if isinstance(outputs, dict) and "predicted_depth" in outputs:
                depth = outputs["predicted_depth"]
            else:
                raise RuntimeError("Depth Anything 3 output does not contain predicted_depth.")

        if depth.dim() == 3:
            depth = depth[:, None]
        elif depth.dim() == 2:
            depth = depth[None, None]

        depth = depth.float().clip(0.0, None)
        if depth.shape[-2:] != target_hw:
            depth = F.interpolate(depth, target_hw, mode="bilinear", align_corners=False)
        return depth

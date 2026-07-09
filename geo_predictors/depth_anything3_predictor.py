import os

import torch
import torch.nn.functional as F
from torchvision.transforms.functional import to_pil_image

from .geo_predictor import GeoPredictor


class DepthAnything3Predictor(GeoPredictor):
    def __init__(self, model_id=None):
        super().__init__()
        self.model_id = model_id or os.getenv("DEPTH_ANYTHING3_MODEL")
        if not self.model_id:
            raise RuntimeError(
                "Depth Anything 3 model id is not set. "
                "Pass --depth_anything3_model or set DEPTH_ANYTHING3_MODEL."
            )

        try:
            from transformers import AutoImageProcessor, AutoModelForDepthEstimation
        except Exception as exc:
            raise RuntimeError(
                "Depth Anything 3 predictor needs transformers with depth-estimation model support."
            ) from exc

        self.processor = AutoImageProcessor.from_pretrained(self.model_id, trust_remote_code=True)
        self.model = AutoModelForDepthEstimation.from_pretrained(
            self.model_id,
            trust_remote_code=True,
        ).cuda().eval()

    @torch.no_grad()
    def predict_depth(self, img, **kwargs):
        if img.dim() != 4 or img.shape[0] != 1:
            raise ValueError("DepthAnything3Predictor expects a [1, 3, H, W] tensor.")

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

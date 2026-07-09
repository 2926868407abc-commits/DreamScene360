import os

import torch
import torch.nn.functional as F
from torchvision.transforms.functional import to_pil_image

from .geo_predictor import GeoPredictor


class DepthAnything3Predictor(GeoPredictor):
    def __init__(self, model_id=None):
        super().__init__()
        self.model_id = model_id or os.getenv("DEPTH_ANYTHING3_MODEL") or "depth-anything/DA3-LARGE-1.1"
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

import os
import sys
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn.functional as F
from torchvision.transforms.functional import to_pil_image

from .geo_predictor import GeoPredictor


class G2VLMPredictor(GeoPredictor):
    def __init__(self, g2vlm_root=None, model_path=None):
        super().__init__()
        if not torch.cuda.is_available():
            raise RuntimeError("G2VLM depth prediction requires CUDA in the current G2VLM codepath.")

        repo_root = Path(__file__).resolve().parents[2]
        default_g2vlm_root = repo_root.parent / "G2VLM"
        self.g2vlm_root = Path(g2vlm_root or os.getenv("G2VLM_ROOT") or default_g2vlm_root).resolve()
        self.model_path = Path(
            model_path
            or os.getenv("G2VLM_MODEL_PATH")
            or self.g2vlm_root / "models" / "G2VLM-2B-MoT"
        ).resolve()

        if not self.g2vlm_root.exists():
            raise FileNotFoundError(f"G2VLM root not found: {self.g2vlm_root}")
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"G2VLM checkpoint not found: {self.model_path}. "
                "Pass --g2vlm_model_path or set G2VLM_MODEL_PATH."
            )

        sys.path.insert(0, str(self.g2vlm_root))
        from g2vlm_utils import load_model_and_tokenizer

        args = SimpleNamespace(model_path=str(self.model_path))
        self.model, self.tokenizer, self.new_token_ids, _, self.dino_transform = load_model_and_tokenizer(args)

    @torch.no_grad()
    def predict_depth(self, img, **kwargs):
        if img.dim() != 4 or img.shape[0] != 1:
            raise ValueError("G2VLMPredictor.predict_depth expects a [1, 3, H, W] tensor.")

        target_hw = img.shape[-2:]
        pil_img = to_pil_image(img[0].detach().clamp(0.0, 1.0).cpu())

        pred = self.model.recon(
            self.tokenizer,
            self.new_token_ids,
            self.dino_transform,
            [pil_img],
        )

        depth = pred["local_points"][0, 0, ..., -1]
        depth = depth[None, None].float().clip(0.0, None)
        if depth.shape[-2:] != target_hw:
            depth = F.interpolate(depth, target_hw, mode="bilinear", align_corners=False)
        return depth

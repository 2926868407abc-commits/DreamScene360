import os
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

from .geo_predictor import GeoPredictor


class VGGTPredictor(GeoPredictor):
    def __init__(self, vggt_root=None, model_path=None, chunk_size=8):
        super().__init__()
        if not torch.cuda.is_available():
            raise RuntimeError("VGGT depth prediction requires CUDA.")

        repo_root = Path(__file__).resolve().parents[2]
        default_vggt_root = repo_root.parent / "vggt"
        self.vggt_root = Path(vggt_root or os.getenv("VGGT_ROOT") or default_vggt_root).resolve()
        self.model_path = model_path or os.getenv("VGGT_MODEL_PATH") or "facebook/VGGT-1B"
        self.chunk_size = int(os.getenv("VGGT_CHUNK_SIZE", chunk_size or 8))

        if not self.vggt_root.exists():
            raise FileNotFoundError(f"VGGT root not found: {self.vggt_root}")

        sys.path.insert(0, str(self.vggt_root))
        from vggt.models.vggt import VGGT

        if Path(str(self.model_path)).exists():
            self.model = VGGT()
            state = torch.load(str(self.model_path), map_location="cpu")
            if isinstance(state, dict) and "model" in state:
                state = state["model"]
            self.model.load_state_dict(state)
        else:
            self.model = VGGT.from_pretrained(str(self.model_path))

        self.model.cuda().eval()
        major = torch.cuda.get_device_capability()[0]
        self.dtype = torch.bfloat16 if major >= 8 else torch.float16

    @torch.no_grad()
    def predict_depth(self, img, **kwargs):
        return self.predict_depth_batch(img, intrinsics=[kwargs.get("intri", {})])

    @torch.no_grad()
    def predict_depth_batch(self, imgs, intrinsics=None):
        if imgs.dim() != 4:
            raise ValueError("VGGTPredictor expects a [N, 3, H, W] tensor.")

        target_hw = imgs.shape[-2:]
        depths = []
        chunk_size = max(1, self.chunk_size)

        for start in range(0, imgs.shape[0], chunk_size):
            chunk = imgs[start:start + chunk_size].cuda().float().clamp(0.0, 1.0)
            model_input = F.interpolate(chunk, size=(518, 518), mode="bilinear", align_corners=False)
            with torch.cuda.amp.autocast(dtype=self.dtype):
                pred = self.model(model_input)

            depth = pred["depth"]
            if depth.dim() == 5:
                depth = depth[0].permute(0, 3, 1, 2)
            elif depth.dim() == 4 and depth.shape[-1] == 1:
                depth = depth.permute(0, 3, 1, 2)
            elif depth.dim() == 3:
                depth = depth[:, None]

            depth = depth.float().clip(0.0, None)
            if depth.shape[-2:] != target_hw:
                depth = F.interpolate(depth, target_hw, mode="bilinear", align_corners=False)
            depths.append(depth)

        return torch.cat(depths, dim=0)

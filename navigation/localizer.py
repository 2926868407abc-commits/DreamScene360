"""Visual localization: match an input photo against 3DGS training views
using DINOv2 patch features.

This lets you say "take me to where this photo was taken" — the system
finds the matching training camera and returns its 3D position + direction.
"""
import torch
import torch.nn.functional as F
from torchvision.transforms import functional as TF
from torchvision.transforms import InterpolationMode
import numpy as np
from PIL import Image


class VisualLocalizer:
    """Match input images to 3DGS training cameras via DINOv2 features.

    Usage:
        localizer = VisualLocalizer(device="cuda")
        localizer.build_index(train_cameras)
        result = localizer.localize("my_photo.jpg")
        print(f"Matched camera #{result['camera_idx']}, "
              f"position={result['position']}")
    """

    def __init__(self, device="cuda"):
        self.device = device
        self.index = []          # list of (descriptor, camera_idx, camera_obj)
        self.cameras = []

    # ------------------------------------------------------------------
    # Model lazy-load (torchvision DINOv2, no G2VLM dependency)
    # ------------------------------------------------------------------
    @property
    def model(self):
        if not hasattr(self, "_model"):
            from torchvision.models import dinov2_vits14
            self._model = dinov2_vits14(pretrained=True).to(self.device).eval()
            for p in self._model.parameters():
                p.requires_grad_(False)
        return self._model

    # ------------------------------------------------------------------
    # Feature extraction (single image -> global descriptor)
    # ------------------------------------------------------------------
    @torch.no_grad()
    def _extract(self, img_tensor):
        """Extract a global DINO descriptor from a [3, H, W] image tensor.

        Args:
            img_tensor: [3, H, W] float tensor in [0, 1].
        Returns:
            descriptor: [D] float tensor.
        """
        # resize + crop to 224
        img = TF.resize(img_tensor, 224, InterpolationMode.BILINEAR)
        img = TF.center_crop(img, (224, 224))
        # ImageNet normalisation
        img = TF.normalize(img,
                           mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
        img = img.unsqueeze(0).to(self.device)  # [1, 3, 224, 224]

        feats = self.model.forward_features(img)["x_norm_patchtokens"]  # [1, N, D]
        descriptor = feats[0].mean(dim=0)  # [D]  — average-pool patch tokens
        return descriptor

    @torch.no_grad()
    def _extract_from_path(self, image_path):
        """Load an image from disk and extract its descriptor."""
        pil = Image.open(image_path).convert("RGB")
        tensor = TF.pil_to_tensor(pil).float() / 255.0  # [3, H, W] in [0, 1]
        return self._extract(tensor)

    # ------------------------------------------------------------------
    # Build feature index from 3DGS training cameras
    # ------------------------------------------------------------------
    @torch.no_grad()
    def build_index(self, cameras, verbose=True):
        """Extract and store DINO descriptors for every training camera.

        Args:
            cameras: list of scene.cameras.Camera objects.
        """
        self.cameras = list(cameras)
        self.index = []

        for idx, cam in enumerate(self.cameras):
            img = cam.original_image                 # [3, H, W] in [0, 1]
            desc = self._extract(img)
            self.index.append((desc.cpu(), idx, cam))

            if verbose and idx % 20 == 0:
                print(f"  [{idx}/{len(self.cameras)}] indexed camera #{idx}")

        print(f"[VisualLocalizer] indexed {len(self.index)} cameras")
        return self

    # ------------------------------------------------------------------
    # Match an input image -> 3D position
    # ------------------------------------------------------------------
    @torch.no_grad()
    def localize(self, image_path, forward_dist=2.0, top_k=3):
        """Match an input photo to the nearest training camera.

        Args:
            image_path: path to input photo (jpg/png).
            forward_dist: how far (in scene units) to place the position
                          along the camera's look-at direction.
            top_k: return top-K matches for debugging.

        Returns:
            dict with keys:
                position    - [3] 3D world position for path planning.
                direction   - [3] forward direction of the matched camera.
                camera_idx  - index of best-match camera.
                similarity  - cosine similarity score.
                top_k       - list of (idx, score) for the top K matches.
        """
        assert len(self.index) > 0, "call build_index() first"

        query = self._extract_from_path(image_path).cpu()   # [D]
        query = query / (query.norm() + 1e-10)

        scores = []
        for desc, idx, cam in self.index:
            desc_n = desc / (desc.norm() + 1e-10)
            sim = (query * desc_n).sum().item()
            scores.append((sim, idx, cam))

        scores.sort(key=lambda x: -x[0])
        best_sim, best_idx, best_cam = scores[0]

        # camera forward direction in world space
        direction = self._camera_forward(best_cam)

        # position = look-at direction * distance, snapped to a reasonable height
        position = direction * forward_dist
        position[1] = 0.0   # will be overridden by ground height in planner

        return {
            "position": position,
            "direction": direction,
            "camera_idx": best_idx,
            "similarity": best_sim,
            "top_k": [(s[0], s[1]) for s in scores[:top_k]],
        }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _camera_forward(cam):
        """Extract the camera's forward (look-at) direction in world space."""
        # cam.R is the world-to-camera rotation (OpenCV convention).
        # Camera's forward in world = third row of W2C = third column of R^T.
        R = torch.tensor(cam.R, dtype=torch.float)
        forward = R[:, 2]  # third column of world-to-camera = forward in world
        return forward / (forward.norm() + 1e-10)

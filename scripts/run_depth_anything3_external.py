import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image


def load_images(input_dir):
    paths = sorted(
        path for path in Path(input_dir).iterdir()
        if path.suffix.lower() in [".png", ".jpg", ".jpeg", ".webp"]
    )
    if not paths:
        raise RuntimeError(f"No input images found in {input_dir}.")
    images = [Image.open(path).convert("RGB") for path in paths]
    return paths, images


def to_numpy_depth(depth):
    if torch.is_tensor(depth):
        depth = depth.detach().cpu().numpy()
    else:
        depth = np.asarray(depth)
    return depth.astype(np.float32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model", default="depth-anything/DA3-LARGE-1.1")
    args = parser.parse_args()

    from depth_anything_3.api import DepthAnything3

    input_paths, images = load_images(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model = DepthAnything3.from_pretrained(args.model)
    if torch.cuda.is_available():
        model = model.cuda()

    with torch.no_grad():
        prediction = model.inference(images)

    depths = to_numpy_depth(prediction.depth)
    if depths.ndim == 2:
        depths = depths[None]
    if depths.ndim == 4 and depths.shape[1] == 1:
        depths = depths[:, 0]
    if depths.shape[0] != len(input_paths):
        raise RuntimeError(
            f"Depth Anything 3 returned {depths.shape[0]} depths for {len(input_paths)} images."
        )

    for index, depth in enumerate(depths):
        np.save(output_dir / f"{index:06d}.npy", np.asarray(depth, dtype=np.float32))


if __name__ == "__main__":
    main()

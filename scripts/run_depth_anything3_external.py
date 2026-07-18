import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image


def list_images(input_dir):
    paths = sorted(
        path for path in Path(input_dir).iterdir()
        if path.suffix.lower() in [".png", ".jpg", ".jpeg", ".webp"]
    )
    if not paths:
        raise RuntimeError(f"No input images found in {input_dir}.")
    return paths


def load_images(paths):
    return [Image.open(path).convert("RGB") for path in paths]


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
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()
    if args.batch_size <= 0:
        raise ValueError(f"--batch-size must be positive, got {args.batch_size}")

    from depth_anything_3.api import DepthAnything3

    input_paths = list_images(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model = DepthAnything3.from_pretrained(args.model)
    if torch.cuda.is_available():
        model = model.cuda()

    with torch.no_grad():
        for start in range(0, len(input_paths), args.batch_size):
            batch_paths = input_paths[start:start + args.batch_size]
            images = load_images(batch_paths)
            prediction = model.inference(images)
            depths = to_numpy_depth(prediction.depth)
            if depths.ndim == 2:
                depths = depths[None]
            if depths.ndim == 4 and depths.shape[1] == 1:
                depths = depths[:, 0]
            if depths.shape[0] != len(batch_paths):
                raise RuntimeError(
                    f"Depth Anything 3 returned {depths.shape[0]} depths for {len(batch_paths)} images."
                )

            for offset, depth in enumerate(depths):
                index = start + offset
                np.save(output_dir / f"{index:06d}.npy", np.asarray(depth, dtype=np.float32))

            del images, prediction, depths
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


if __name__ == "__main__":
    main()

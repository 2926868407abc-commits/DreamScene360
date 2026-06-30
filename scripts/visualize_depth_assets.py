from pathlib import Path
import argparse

import cv2 as cv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw
import torch
import trimesh

from geo_predictors.omnidata_predictor import OmnidataPredictor


def save_depth_vis(depth, out_path):
    depth = np.squeeze(depth).astype(np.float32)
    valid = np.isfinite(depth)
    lo, hi = np.percentile(depth[valid], [2, 98])
    depth_norm = np.clip((depth - lo) / (hi - lo + 1e-6), 0, 1)
    rgb = (plt.get_cmap("magma")(depth_norm)[..., :3] * 255).astype(np.uint8)
    Image.fromarray(rgb).save(out_path)


def draw_pano_boxes(pano_path, out_path, labels):
    img = cv.imread(str(pano_path), cv.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(pano_path)

    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    img = cv.resize(img, (2048, 1024), cv.INTER_AREA)

    canvas = Image.fromarray(img)
    draw = ImageDraw.Draw(canvas)
    width, height = canvas.size

    centers = [width // 6, width // 2, 5 * width // 6]
    box_w, box_h = int(width * 0.16), int(height * 0.55)
    colors = [(235, 70, 70), (40, 120, 235), (40, 170, 90)]

    for cx, label, color in zip(centers, labels, colors):
        x0, y0 = cx - box_w // 2, height // 2 - box_h // 2
        x1, y1 = cx + box_w // 2, height // 2 + box_h // 2
        draw.rectangle([x0, y0, x1, y1], outline=color, width=8)
        draw.text((x0 + 10, y0 + 10), label, fill=color)

    canvas.save(out_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pano", required=True)
    parser.add_argument("--slice_dir", required=True)
    parser.add_argument("--ply", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--indices", default="1,81,161")
    args = parser.parse_args()

    out = Path(args.out)
    selected_dir = out / "selected_slices"
    local_depth_dir = out / "local_depth"
    selected_dir.mkdir(parents=True, exist_ok=True)
    local_depth_dir.mkdir(parents=True, exist_ok=True)

    indices = [int(x) for x in args.indices.split(",")]
    labels = [f"view {i}" for i in indices]
    draw_pano_boxes(args.pano, out / "pano_three_slices.png", labels)

    predictor = OmnidataPredictor()

    for idx in indices:
        img_path = Path(args.slice_dir) / f"image_{idx}.png"
        img = cv.imread(str(img_path), cv.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(img_path)

        cv.imwrite(str(selected_dir / f"slice_{idx:03d}.png"), img)

        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        img_t = torch.from_numpy(img.astype(np.float32) / 255.0)
        img_t = img_t.permute(2, 0, 1)[None].cuda()

        with torch.no_grad():
            depth = predictor.predict_depth(img_t).detach().cpu().numpy()[0, 0]

        np.save(local_depth_dir / f"local_depth_{idx:03d}.npy", depth)
        save_depth_vis(depth, local_depth_dir / f"local_depth_{idx:03d}.png")

    pc = trimesh.load(args.ply, process=False)
    pts = np.asarray(pc.vertices)
    pano_h, pano_w = 1024, 2048

    if pts.shape[0] != pano_h * pano_w:
        raise ValueError(
            f"point count {pts.shape[0]} != {pano_h * pano_w}; "
            "cannot reshape to panorama depth"
        )

    pano_depth = np.linalg.norm(pts, axis=1).reshape(pano_h, pano_w)
    np.save(out / "panoramic_depth.npy", pano_depth)
    save_depth_vis(pano_depth, out / "panoramic_depth.png")

    print("Saved assets to:", out)


if __name__ == "__main__":
    main()

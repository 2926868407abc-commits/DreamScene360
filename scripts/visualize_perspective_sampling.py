from pathlib import Path
import argparse
import re

import cv2 as cv
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.spatial.transform import Rotation


COLORS = [
    (235, 70, 70),
    (40, 120, 235),
    (40, 170, 90),
    (222, 150, 40),
    (155, 85, 210),
]


def parse_camera_params(line):
    parts = line.strip().split(maxsplit=4)
    cam_id = int(parts[0])
    width = int(parts[2])
    height = int(parts[3])
    params = parts[4]
    params = re.sub(r",\s*device='[^']+'", "", params)
    params = params.replace("tensor(", "").replace(")", "")
    nums = [float(x) for x in re.findall(r"[-+]?(?:\d*\.\d+|\d+)(?:[eE][-+]?\d+)?", params)]
    fx, fy, cx, cy = nums[:4]
    return cam_id, {"width": width, "height": height, "fx": fx, "fy": fy, "cx": cx, "cy": cy}


def read_cameras(path):
    cameras = {}
    for line in Path(path).read_text().splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        cam_id, cam = parse_camera_params(line)
        cameras[cam_id] = cam
    return cameras


def read_images(path):
    images = {}
    for line in Path(path).read_text().splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 10:
            continue
        image_id = int(parts[0])
        qw, qx, qy, qz = map(float, parts[1:5])
        camera_id = int(parts[8])
        name = parts[9]
        rot_w2c = Rotation.from_quat([qx, qy, qz, qw]).as_matrix()
        images[image_id] = {"rot_w2c": rot_w2c, "camera_id": camera_id, "name": name}
    return images


def world_dirs_to_pano_xy(dirs, pano_w, pano_h):
    dirs = dirs / np.linalg.norm(dirs, axis=-1, keepdims=True)
    beta = np.arcsin(np.clip(dirs[:, 2], -1.0, 1.0))
    alpha = np.arctan2(dirs[:, 1], dirs[:, 0])
    x = (-(alpha / (2.0 * np.pi)) + 0.5) * pano_w
    y = (-beta / np.pi + 0.5) * pano_h
    return np.stack([x % pano_w, y], axis=-1)


def unwrap_polyline(points, pano_w):
    points = points.copy()
    for i in range(1, len(points)):
        while points[i, 0] - points[i - 1, 0] > pano_w / 2:
            points[i, 0] -= pano_w
        while points[i - 1, 0] - points[i, 0] > pano_w / 2:
            points[i, 0] += pano_w
    return points


def camera_footprint(image_info, camera, pano_w, pano_h, samples=80):
    w, h = camera["width"], camera["height"]
    fx, fy, cx, cy = camera["fx"], camera["fy"], camera["cx"], camera["cy"]

    top = np.stack([np.linspace(0, w - 1, samples), np.zeros(samples)], axis=-1)
    right = np.stack([np.full(samples, w - 1), np.linspace(0, h - 1, samples)], axis=-1)
    bottom = np.stack([np.linspace(w - 1, 0, samples), np.full(samples, h - 1)], axis=-1)
    left = np.stack([np.zeros(samples), np.linspace(h - 1, 0, samples)], axis=-1)
    border = np.concatenate([top, right, bottom, left], axis=0)

    cam_dirs = np.stack([(border[:, 0] - cx) / fx, (border[:, 1] - cy) / fy, np.ones(len(border))], axis=-1)
    cam_dirs = cam_dirs / np.linalg.norm(cam_dirs, axis=-1, keepdims=True)
    rot_c2w = image_info["rot_w2c"].T
    world_dirs = cam_dirs @ rot_c2w.T
    return world_dirs_to_pano_xy(world_dirs, pano_w, pano_h)


def draw_footprint(draw, points, pano_w, color, label=None):
    points = unwrap_polyline(points, pano_w)
    closed = np.concatenate([points, points[:1]], axis=0)
    for shift in (-pano_w, 0, pano_w):
        shifted = closed.copy()
        shifted[:, 0] += shift
        draw.line([tuple(p) for p in shifted], fill=color, width=8)

    if label is not None:
        label_xy = points.mean(axis=0)
        label_xy[0] %= pano_w
        draw.rectangle([label_xy[0] - 4, label_xy[1] - 24, label_xy[0] + 128, label_xy[1] + 10], fill=(255, 255, 255))
        draw.text((label_xy[0], label_xy[1] - 22), label, fill=color)


def make_contact_sheet(pano_img, slice_dir, indices, out_path, show_labels=True):
    pano_w, pano_h = pano_img.size
    thumb_h = 260
    margin = 30
    sheet = Image.new("RGB", (pano_w, pano_h + thumb_h + margin * 2), "white")
    sheet.paste(pano_img, (0, 0))

    draw = ImageDraw.Draw(sheet)
    n = len(indices)
    thumb_w = 260
    gap = (pano_w - n * thumb_w) // (n + 1)
    y = pano_h + margin

    for i, idx in enumerate(indices):
        color = COLORS[i % len(COLORS)]
        x = gap + i * (thumb_w + gap)
        img_path = Path(slice_dir) / f"image_{idx}.png"
        thumb = Image.open(img_path).convert("RGB").resize((thumb_w, thumb_w), Image.Resampling.LANCZOS)
        sheet.paste(thumb, (x, y))
        draw.rectangle([x, y, x + thumb_w, y + thumb_w], outline=color, width=8)
        if show_labels:
            draw.text((x, y - 24), f"image_{idx}.png", fill=color)

    sheet.save(out_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pano", required=True)
    parser.add_argument("--slice_dir", required=True)
    parser.add_argument("--sparse_dir", required=True)
    parser.add_argument("--indices", default="3,11,57")
    parser.add_argument("--out", required=True)
    parser.add_argument("--no_labels", action="store_true")
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    indices = [int(x) for x in args.indices.split(",")]

    pano = cv.imread(args.pano, cv.IMREAD_COLOR)
    if pano is None:
        raise FileNotFoundError(args.pano)
    pano = cv.cvtColor(pano, cv.COLOR_BGR2RGB)
    pano = cv.resize(pano, (2048, 1024), cv.INTER_AREA)
    pano_img = Image.fromarray(pano)
    draw = ImageDraw.Draw(pano_img)
    pano_w, pano_h = pano_img.size

    cameras = read_cameras(Path(args.sparse_dir) / "cameras.txt")
    images = read_images(Path(args.sparse_dir) / "images.txt")

    for i, idx in enumerate(indices):
        info = images[idx]
        cam = cameras[info["camera_id"]]
        footprint = camera_footprint(info, cam, pano_w, pano_h)
        label = None if args.no_labels else f"image {idx}"
        draw_footprint(draw, footprint, pano_w, COLORS[i % len(COLORS)], label)

    pano_only = out / f"pano_sampling_{'_'.join(f'{i:03d}' for i in indices)}.png"
    sheet = out / f"pano_sampling_with_views_{'_'.join(f'{i:03d}' for i in indices)}.png"
    pano_img.save(pano_only)
    make_contact_sheet(pano_img, args.slice_dir, indices, sheet, show_labels=not args.no_labels)

    print("Saved:", pano_only)
    print("Saved:", sheet)


if __name__ == "__main__":
    main()

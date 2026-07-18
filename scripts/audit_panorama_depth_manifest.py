"""Audit panorama-depth manifests for shape and aspect-ratio issues."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from struct import unpack

import numpy as np
from PIL import Image


def read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def image_shape(path: Path) -> tuple[int, int]:
    with Image.open(path) as image:
        width, height = image.size
    return height, width


def dpt_shape(path: Path) -> tuple[int, int]:
    with path.open("rb") as f:
        tag = unpack("f", f.read(4))[0]
        width = unpack("i", f.read(4))[0]
        height = unpack("i", f.read(4))[0]
    if tag != 202021.25:
        raise ValueError(f"{path} has invalid .dpt tag: {tag}")
    return height, width


def array_shape(path: Path) -> tuple[int, int]:
    suffix = path.suffix.lower()
    if suffix == ".dpt":
        return dpt_shape(path)
    if suffix == ".npy":
        array = np.load(path, mmap_mode="r")
    elif suffix == ".npz":
        data = np.load(path)
        key = "depth" if "depth" in data.files else data.files[0]
        array = data[key]
    else:
        return image_shape(path)

    shape = tuple(int(v) for v in np.asarray(array).squeeze().shape)
    if len(shape) < 2:
        raise ValueError(f"{path} has invalid depth shape: {shape}")
    return shape[-2], shape[-1]


def parse_dataset_filter(text: str) -> set[str]:
    return {item.strip() for item in text.split(",") if item.strip()}


def has_pano_hint(path: Path) -> bool:
    lower_parts = {part.lower() for part in path.parts}
    lower = str(path).lower()
    return "pano" in lower_parts or "equirectangular" in lower or "panorama" in lower


def fmt(value: float) -> str:
    if not math.isfinite(value):
        return "NA"
    return f"{value:.4f}"


def write_csv(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def markdown(rows: list[dict[str, object]]) -> str:
    lines = [
        "| Dataset | Scene | RGB HxW | Depth HxW | Aspect | Shape Match | Aspect OK | Pano Path Hint | RGB Path |",
        "|---|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['dataset']} | {row['scene']} | {row['rgb_h']}x{row['rgb_w']} | "
            f"{row['depth_h']}x{row['depth_w']} | {fmt(float(row['aspect']))} | "
            f"{row['shape_match']} | {row['aspect_ok']} | {row['pano_path_hint']} | {row['rgb_path']} |"
        )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit panorama depth manifest image shapes.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("panorama_depth_manifest_audit"))
    parser.add_argument("--datasets", default="", help="Optional comma-separated dataset filter.")
    parser.add_argument(
        "--expected-aspect",
        type=float,
        default=0.0,
        help="Expected width/height ratio. Use 0 to skip aspect-ratio validation.",
    )
    parser.add_argument("--aspect-tolerance", type=float, default=0.05)
    parser.add_argument("--fail-on-invalid", action="store_true")
    args = parser.parse_args()

    dataset_filter = parse_dataset_filter(args.datasets)
    rows: list[dict[str, object]] = []
    invalid_rows: list[dict[str, object]] = []

    for row in read_manifest(args.manifest):
        dataset = row["dataset"].strip()
        if dataset_filter and dataset not in dataset_filter:
            continue

        rgb_path = Path(row["rgb_path"]).expanduser()
        depth_path = Path(row["depth_path"]).expanduser()
        rgb_h, rgb_w = image_shape(rgb_path)
        depth_h, depth_w = array_shape(depth_path)
        aspect = rgb_w / max(rgb_h, 1)
        shape_match = (rgb_h, rgb_w) == (depth_h, depth_w)
        aspect_ok = args.expected_aspect <= 0 or abs(aspect - args.expected_aspect) <= args.aspect_tolerance
        audit_row = {
            "dataset": dataset,
            "scene": row.get("scene", "").strip() or rgb_path.stem,
            "rgb_h": rgb_h,
            "rgb_w": rgb_w,
            "depth_h": depth_h,
            "depth_w": depth_w,
            "aspect": aspect,
            "shape_match": shape_match,
            "aspect_ok": aspect_ok,
            "pano_path_hint": has_pano_hint(rgb_path),
            "rgb_path": str(rgb_path),
            "depth_path": str(depth_path),
        }
        rows.append(audit_row)
        if not shape_match or not aspect_ok:
            invalid_rows.append(audit_row)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    fields = [
        "dataset",
        "scene",
        "rgb_h",
        "rgb_w",
        "depth_h",
        "depth_w",
        "aspect",
        "shape_match",
        "aspect_ok",
        "pano_path_hint",
        "rgb_path",
        "depth_path",
    ]
    write_csv(args.output_dir / "audit.csv", rows, fields)
    (args.output_dir / "audit.md").write_text(markdown(rows), encoding="utf-8")

    by_dataset: dict[str, dict[str, int]] = {}
    for row in rows:
        stats = by_dataset.setdefault(str(row["dataset"]), {"total": 0, "valid": 0})
        stats["total"] += 1
        if bool(row["shape_match"]) and bool(row["is_equirectangular"]):
            stats["valid"] += 1

    print("| Dataset | Shape/Aspect Valid | Pano Path Hint | Total |")
    print("|---|---:|---:|---:|")
    for dataset, stats in sorted(by_dataset.items()):
        pano_count = sum(1 for row in rows if row["dataset"] == dataset and row["pano_path_hint"])
        print(f"| {dataset} | {stats['valid']} | {pano_count} | {stats['total']} |")
    print(f"[done] {args.output_dir / 'audit.csv'}")
    print(f"[done] {args.output_dir / 'audit.md'}")

    if invalid_rows and args.fail_on_invalid:
        raise SystemExit(1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

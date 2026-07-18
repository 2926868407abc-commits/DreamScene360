"""Build a CSV manifest for panorama depth fusion evaluation.

The evaluator expects rows in this format:

    dataset,scene,rgb_path,depth_path,mask_path,depth_scale

This helper scans unpacked benchmark folders and pairs panorama RGB files with
their corresponding depth files. It is intentionally conservative: if your
local dataset layout is unusual, run with --verbose and inspect the reported
unpaired candidates.
"""

from __future__ import annotations

import argparse
import csv
import re
from dataclasses import dataclass
from pathlib import Path


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}
DEPTH_SUFFIXES = {".dpt", ".npy", ".npz", ".exr"}
RGB_HINTS = ("rgb", "color", "image")
DEPTH_HINTS = ("depth", "dpt")
DROP_TOKENS = (
    "rgb",
    "color",
    "image",
    "depth",
    "dpt",
    "pano",
    "panorama",
    "erp",
    "equirectangular",
    "domain",
)


@dataclass(frozen=True)
class DatasetSpec:
    label: str
    default_dirs: tuple[str, ...]
    depth_scale: str
    pano_only: bool = False


DATASETS = {
    "matterport3d": DatasetSpec(
        label="Matterport3D",
        default_dirs=("Matterport3D360", "Matterport3D"),
        depth_scale="",
    ),
    "stanford2d3d": DatasetSpec(
        label="Stanford2D3D",
        default_dirs=("Stanford2D3D", "2D-3D-S", "2D3DS"),
        depth_scale="512",
        pano_only=True,
    ),
    "deep360": DatasetSpec(
        label="Deep360",
        default_dirs=("Deep360",),
        depth_scale="",
    ),
}


def normalize_text(text: str) -> str:
    text = text.lower()
    for token in DROP_TOKENS:
        text = re.sub(rf"(^|[^a-z0-9]){re.escape(token)}([^a-z0-9]|$)", " ", text)
    return re.sub(r"[^a-z0-9]+", "", text)


def file_key(path: Path, root: Path) -> str:
    rel = path.relative_to(root).with_suffix("")
    return normalize_text(str(rel))


def stem_key(path: Path) -> str:
    return normalize_text(path.stem)


def is_rgb_file(path: Path) -> bool:
    if path.suffix.lower() not in IMAGE_SUFFIXES:
        return False
    lower = str(path).lower()
    if any(hint in lower for hint in DEPTH_HINTS):
        return False
    return any(hint in lower for hint in RGB_HINTS)


def is_depth_file(path: Path) -> bool:
    lower = str(path).lower()
    if path.suffix.lower() in DEPTH_SUFFIXES:
        return True
    return path.suffix.lower() in IMAGE_SUFFIXES and any(hint in lower for hint in DEPTH_HINTS)


def is_pano_file(path: Path) -> bool:
    lower_parts = {part.lower() for part in path.parts}
    lower = str(path).lower()
    return "pano" in lower_parts or "equirectangular" in lower or "panorama" in lower


def score_pair(rgb: Path, depth: Path, root: Path) -> tuple[int, int]:
    rgb_parts = set(p.lower() for p in rgb.relative_to(root).parts[:-1])
    depth_parts = set(p.lower() for p in depth.relative_to(root).parts[:-1])
    shared_dirs = len(rgb_parts & depth_parts)
    same_parent = int(rgb.parent == depth.parent)
    same_grandparent = int(rgb.parent.parent == depth.parent.parent)
    same_stem = int(stem_key(rgb) == stem_key(depth))
    same_file_key = int(file_key(rgb, root) == file_key(depth, root))
    score = same_file_key * 100 + same_stem * 40 + same_parent * 20 + same_grandparent * 10 + shared_dirs
    distance = abs(len(rgb.parts) - len(depth.parts))
    return score, -distance


def find_dataset_root(repo_root: Path, spec: DatasetSpec) -> Path | None:
    datasets_root = repo_root / "datasets"
    for name in spec.default_dirs:
        candidate = datasets_root / name
        if candidate.exists():
            return candidate
    return None


def pair_dataset(root: Path, limit: int, pano_only: bool = False) -> list[tuple[Path, Path]]:
    files = [p for p in root.rglob("*") if p.is_file()]
    if pano_only:
        files = [p for p in files if is_pano_file(p)]
    rgbs = sorted(p for p in files if is_rgb_file(p))
    depths = sorted(p for p in files if is_depth_file(p))

    pairs: list[tuple[Path, Path]] = []
    used_rgbs: set[Path] = set()
    for depth in depths:
        candidates = [rgb for rgb in rgbs if rgb not in used_rgbs]
        if not candidates:
            break
        best = max(candidates, key=lambda rgb: score_pair(rgb, depth, root))
        if score_pair(best, depth, root)[0] <= 0:
            continue
        pairs.append((best, depth))
        used_rgbs.add(best)
        if limit > 0 and len(pairs) >= limit:
            break
    return pairs


def main() -> int:
    parser = argparse.ArgumentParser(description="Create panorama depth evaluation manifest.")
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path, default=Path("panorama_depth_manifest.csv"))
    parser.add_argument(
        "--datasets",
        default="Matterport3D,Stanford2D3D",
        help="Comma-separated dataset names: Matterport3D, Stanford2D3D, Deep360",
    )
    parser.add_argument("--max-per-dataset", type=int, default=5, help="Use 0 for all pairs.")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    repo_root = args.repo_root.resolve()
    output = args.output if args.output.is_absolute() else repo_root / args.output
    requested = [name.strip().lower() for name in args.datasets.split(",") if name.strip()]

    rows: list[dict[str, str]] = []
    for name in requested:
        key = name.replace("-", "").replace("_", "")
        spec = DATASETS.get(key)
        if spec is None:
            print(f"[warn] unknown dataset name: {name}")
            continue

        root = find_dataset_root(repo_root, spec)
        if root is None:
            print(f"[warn] {spec.label}: dataset folder not found under {repo_root / 'datasets'}")
            continue

        pairs = pair_dataset(root, args.max_per_dataset, pano_only=spec.pano_only)
        print(f"[info] {spec.label}: paired {len(pairs)} samples from {root}")
        if args.verbose:
            for rgb, depth in pairs[:10]:
                print(f"  rgb={rgb}")
                print(f"  dep={depth}")

        for idx, (rgb, depth) in enumerate(pairs):
            rows.append(
                {
                    "dataset": spec.label,
                    "scene": f"{spec.label}_{idx:05d}",
                    "rgb_path": str(rgb),
                    "depth_path": str(depth),
                    "mask_path": "",
                    "depth_scale": spec.depth_scale,
                }
            )

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["dataset", "scene", "rgb_path", "depth_path", "mask_path", "depth_scale"],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"[done] wrote {output} ({len(rows)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

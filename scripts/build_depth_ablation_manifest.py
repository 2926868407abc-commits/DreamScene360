from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path


LABELS = {
    "omnidata": "Ours-Omnidata",
    "depth_anything3": "DepthAnything3",
    "dap": "DAP",
    "vggt_omega": "VGGT-Omega",
    "g2vlm": "G2VLM",
}


def canonical_method(name: str) -> str:
    key = name.lower().replace("-", "_")
    aliases = {
        "da3": "depth_anything3",
        "depthanything3": "depth_anything3",
        "depth_anything_3": "depth_anything3",
        "vggt": "vggt_omega",
        "vggtomega": "vggt_omega",
    }
    return aliases.get(key, key)


def safe_name(name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_]+", "_", canonical_method(name)).strip("_")


def read_prompt(source_path: Path) -> str:
    if source_path.is_file() and source_path.suffix == ".txt":
        return source_path.read_text(encoding="utf-8").strip()
    if source_path.is_dir():
        prompt_files = sorted(source_path.glob("*.txt"))
        if prompt_files:
            return prompt_files[0].read_text(encoding="utf-8").strip()
    return ""


def read_runtime(output_root: Path, output_prefix: str, method: str, scene: str) -> str:
    path = output_root / f"{output_prefix}_{safe_name(method)}_{scene}" / "runtime_sec.txt"
    if path.exists():
        return path.read_text(encoding="utf-8").strip()
    return ""


def main() -> int:
    parser = argparse.ArgumentParser(description="Build depth ablation evaluation manifest")
    parser.add_argument("--methods", nargs="+", required=True)
    parser.add_argument("--scene", required=True)
    parser.add_argument("--source-path", type=Path, required=True)
    parser.add_argument("--output-prefix", default="ablation_t01")
    parser.add_argument("--view-tag", default="paper_t01")
    parser.add_argument("--output-root", type=Path, default=Path("output"))
    parser.add_argument("--render-root", type=Path, default=Path("table1_inputs"))
    parser.add_argument("--output", type=Path, default=Path("table1_manifest_ablation_t01.csv"))
    args = parser.parse_args()

    prompt = read_prompt(args.source_path)
    rows = []
    for raw_method in args.methods:
        method = canonical_method(raw_method)
        name = safe_name(method)
        rows.append(
            {
                "method": LABELS.get(method, method),
                "scene": args.scene,
                "prompt": prompt,
                "image_path": "",
                "image_dir": str(args.render_root / name / f"{args.scene}_{args.view_tag}" / "renders"),
                "image_glob": "*.png",
                "runtime_sec": read_runtime(args.output_root, args.output_prefix, method, args.scene),
            }
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "method",
                "scene",
                "prompt",
                "image_path",
                "image_dir",
                "image_glob",
                "runtime_sec",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

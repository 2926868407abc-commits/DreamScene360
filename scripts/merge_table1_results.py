"""
Merge separately computed Table 1 metric summaries into one table.

Typical use:

python scripts/merge_table1_results.py \
  --iqa table1_eval_iqa_only/summary.csv \
  --clip table1_eval_clip_only/summary.csv \
  --qalign table1_eval_qalign/summary.csv \
  --output-dir table1_eval_full
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path


FIELDS = [
    "method",
    "num_images",
    "clip_distance",
    "q_align",
    "niqe",
    "brisque",
    "runtime_sec",
    "runtime",
]


def read_summary(path: Path | None) -> dict[str, dict[str, str]]:
    if path is None or not str(path).strip():
        return {}
    if not path.exists():
        print(f"[warn] missing summary, skipping: {path}")
        return {}
    with path.open(newline="", encoding="utf-8-sig") as f:
        return {row["method"]: row for row in csv.DictReader(f)}


def first_value(method: str, key: str, sources: list[dict[str, dict[str, str]]]) -> str:
    for source in sources:
        value = source.get(method, {}).get(key, "")
        text = str(value).strip()
        if text and text.lower() not in {"nan", "na", "none", "null"}:
            return value
    return ""


def metric_text(value: object) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "NA"
    if math.isnan(number):
        return "NA"
    return f"{number:.4f}"


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path: Path, rows: list[dict[str, str]]) -> None:
    lines = [
        "| Method | CLIP Distance↓ | Q-Align↑ | NIQE↓ | BRISQUE↓ | Runtime |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {method} | {clip} | {qalign} | {niqe} | {brisque} | {runtime} |".format(
                method=row["method"],
                clip=metric_text(row.get("clip_distance")),
                qalign=metric_text(row.get("q_align")),
                niqe=metric_text(row.get("niqe")),
                brisque=metric_text(row.get("brisque")),
                runtime=row.get("runtime") or "NA",
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Merge DreamScene360 Table 1 metric summaries")
    parser.add_argument("--iqa", type=Path, default=Path("table1_eval_iqa_only/summary.csv"))
    parser.add_argument("--clip", type=Path, default=Path("table1_eval_clip_only/summary.csv"))
    parser.add_argument("--qalign", type=Path, default=Path("table1_eval_qalign/summary.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("table1_eval_full"))
    args = parser.parse_args()

    iqa = read_summary(args.iqa)
    clip = read_summary(args.clip)
    qalign = read_summary(args.qalign)
    sources = [clip, qalign, iqa]

    methods = sorted(set(iqa) | set(clip) | set(qalign))
    if not methods:
        raise ValueError("No methods found. Check the input summary.csv paths.")

    rows = []
    for method in methods:
        rows.append({key: first_value(method, key, sources) for key in FIELDS})
        rows[-1]["method"] = method

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "summary.csv", rows)
    write_markdown(args.output_dir / "table1.md", rows)

    print(f"merged -> {args.output_dir / 'summary.csv'}")
    print(f"merged -> {args.output_dir / 'table1.md'}")
    print((args.output_dir / "table1.md").read_text(encoding="utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

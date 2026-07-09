"""
Evaluate Table 1 style metrics for DreamScene360 comparisons.

The manifest is a CSV with one row per image or image directory:

method,scene,prompt,image_path,image_dir,image_glob,runtime_sec
Ours,alley,"a narrow alley ...",output/ours/alley/00000.png,,,440
LucidDreamer,alley,"a narrow alley ...",,output/lucid/alley,*.png,375

Metrics:
  CLIP Distance: 1 - cosine(image, text), lower is better.
  Q-Align: no-reference image quality score, higher is better.
  NIQE / BRISQUE: no-reference image quality scores, lower is better.

Optional dependencies are loaded lazily:
  transformers or open_clip_torch for CLIP Distance
  pyiqa for NIQE / BRISQUE
  Q-Align code for Q-Align, either installed or from VideoScore2's vendored copy
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

import torch
from PIL import Image


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass(frozen=True)
class EvalItem:
    method: str
    scene: str
    prompt: str
    image_path: Path
    runtime_sec: float | None


def read_manifest(path: Path) -> list[EvalItem]:
    items: list[EvalItem] = []
    with path.open("r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        required = {"method", "scene"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Manifest missing required columns: {sorted(missing)}")

        for line_no, row in enumerate(reader, start=2):
            method = (row.get("method") or "").strip()
            scene = (row.get("scene") or "").strip()
            prompt = (row.get("prompt") or "").strip()
            runtime_sec = parse_optional_float(row.get("runtime_sec"))
            image_paths = expand_image_paths(row, manifest_dir=path.parent)

            if not method or not scene:
                raise ValueError(f"Manifest line {line_no}: method and scene are required")
            if not image_paths:
                raise ValueError(f"Manifest line {line_no}: no images matched")

            for image_path in image_paths:
                items.append(
                    EvalItem(
                        method=method,
                        scene=scene,
                        prompt=prompt,
                        image_path=image_path,
                        runtime_sec=runtime_sec,
                    )
                )
    return items


def parse_optional_float(value: str | None) -> float | None:
    if value is None or value.strip() == "":
        return None
    return float(value)


def expand_image_paths(row: dict[str, str], manifest_dir: Path) -> list[Path]:
    image_path = (row.get("image_path") or "").strip()
    image_dir = (row.get("image_dir") or "").strip()
    image_glob = (row.get("image_glob") or "*.png").strip() or "*.png"

    if image_path and image_dir:
        raise ValueError("Use either image_path or image_dir, not both")

    if image_path:
        path = resolve_manifest_path(image_path, manifest_dir)
        return [path] if path.suffix.lower() in IMAGE_SUFFIXES else []

    if image_dir:
        root = resolve_manifest_path(image_dir, manifest_dir)
        return sorted(p for p in root.glob(image_glob) if p.suffix.lower() in IMAGE_SUFFIXES)

    return []


def resolve_manifest_path(value: str, manifest_dir: Path) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = manifest_dir / path
    return path.resolve()


def load_rgb(path: Path) -> Image.Image:
    return Image.open(path).convert("RGB")


def pil_to_tensor(image: Image.Image, device: torch.device) -> torch.Tensor:
    import numpy as np

    arr = np.asarray(image).astype("float32") / 255.0
    tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
    return tensor.to(device)


class ClipScorer:
    def __init__(self, backend: str, model_name: str, device: torch.device):
        self.backend = backend
        self.model_name = model_name
        self.device = device
        self.score: Callable[[Image.Image, str], float]

        if backend == "auto":
            last_error: Exception | None = None
            for candidate in ("transformers", "open_clip"):
                try:
                    self._init_backend(candidate)
                    return
                except Exception as exc:  # noqa: BLE001 - report all backend failures
                    last_error = exc
            raise RuntimeError(
                "CLIP metric requested but no backend could be loaded. "
                "Install transformers or open_clip_torch, and make sure the model is cached."
            ) from last_error

        self._init_backend(backend)

    def _init_backend(self, backend: str) -> None:
        if backend == "transformers":
            from transformers import CLIPModel, CLIPProcessor

            model = CLIPModel.from_pretrained(self.model_name).to(self.device).eval()
            processor = CLIPProcessor.from_pretrained(self.model_name)

            def score(image: Image.Image, prompt: str) -> float:
                inputs = processor(text=[prompt], images=[image], return_tensors="pt", padding=True)
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                with torch.no_grad():
                    out = model(**inputs)
                    image_embeds = torch.nn.functional.normalize(out.image_embeds, dim=-1)
                    text_embeds = torch.nn.functional.normalize(out.text_embeds, dim=-1)
                    sim = (image_embeds * text_embeds).sum(dim=-1)
                return float(1.0 - sim.item())

            self.score = score
            self.backend = backend
            return

        if backend == "open_clip":
            import open_clip

            if "/" in self.model_name:
                model_name, pretrained = self.model_name.split("/", 1)
            else:
                model_name, pretrained = self.model_name, "openai"
            model, _, preprocess = open_clip.create_model_and_transforms(
                model_name, pretrained=pretrained, device=self.device
            )
            tokenizer = open_clip.get_tokenizer(model_name)
            model.eval()

            def score(image: Image.Image, prompt: str) -> float:
                image_tensor = preprocess(image).unsqueeze(0).to(self.device)
                text_tensor = tokenizer([prompt]).to(self.device)
                with torch.no_grad():
                    image_features = torch.nn.functional.normalize(model.encode_image(image_tensor), dim=-1)
                    text_features = torch.nn.functional.normalize(model.encode_text(text_tensor), dim=-1)
                    sim = (image_features * text_features).sum(dim=-1)
                return float(1.0 - sim.item())

            self.score = score
            self.backend = backend
            return

        raise ValueError(f"Unknown CLIP backend: {backend}")


class PyiqaScorer:
    def __init__(self, metric_name: str, device: torch.device):
        import pyiqa

        self.metric_name = metric_name
        self.metric = pyiqa.create_metric(metric_name).to(device).eval()
        self.device = device

    def score(self, image: Image.Image) -> float:
        tensor = pil_to_tensor(image, self.device)
        with torch.no_grad():
            value = self.metric(tensor)
        return float(value.reshape(-1)[0].item())


class QAlignScorer:
    def __init__(self, model_path: str, device: torch.device, vendored_root: Path | None, scale: float):
        if vendored_root is not None:
            sys.path.insert(0, str(vendored_root.resolve()))

        try:
            from q_align.evaluate.scorer import QAlignScorer as _QAlignScorer
        except Exception:
            try:
                from q_align import QAlignScorer as _QAlignScorer
            except Exception as exc:  # noqa: BLE001
                raise RuntimeError(
                    "Q-Align metric requested but q_align could not be imported. "
                    "Install Q-Align or pass --qalign-vendored-root to VideoScore2's utils_q_align."
                ) from exc

        self.model = _QAlignScorer(pretrained=model_path, device=str(device)).eval()
        self.scale = scale

    def score(self, image: Image.Image) -> float:
        with torch.no_grad():
            value = self.model([image])
        return float(value.reshape(-1)[0].item() * self.scale)


def mean(values: Iterable[float]) -> float:
    values = list(values)
    if not values:
        return math.nan
    return float(sum(values) / len(values))


def format_runtime(seconds: float) -> str:
    minutes = int(seconds // 60)
    secs = int(round(seconds - minutes * 60))
    return f"{minutes}min.{secs:02d}sec."


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def safe_float(value: object) -> float | None:
    if value is None:
        return None
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(f) else f


def build_markdown_table(summary_rows: list[dict[str, object]]) -> str:
    lines = [
        "| Method | CLIP Distance↓ | Q-Align↑ | NIQE↓ | BRISQUE↓ | Runtime |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in summary_rows:
        runtime = row.get("runtime")
        lines.append(
            "| {method} | {clip} | {qalign} | {niqe} | {brisque} | {runtime} |".format(
                method=row["method"],
                clip=format_metric(row.get("clip_distance")),
                qalign=format_metric(row.get("q_align")),
                niqe=format_metric(row.get("niqe")),
                brisque=format_metric(row.get("brisque")),
                runtime=runtime if runtime else "NA",
            )
        )
    return "\n".join(lines) + "\n"


def build_latex_table(summary_rows: list[dict[str, object]]) -> str:
    lines = [
        r"\begin{tabular}{lccccc}",
        r"\toprule",
        r"Method & CLIP Distance$\downarrow$ & Q-Align$\uparrow$ & NIQE$\downarrow$ & BRISQUE$\downarrow$ & Runtime \\",
        r"\midrule",
    ]
    for row in summary_rows:
        runtime = row.get("runtime") or "NA"
        lines.append(
            "{method} & {clip} & {qalign} & {niqe} & {brisque} & {runtime} \\\\".format(
                method=latex_escape(str(row["method"])),
                clip=format_metric(row.get("clip_distance")),
                qalign=format_metric(row.get("q_align")),
                niqe=format_metric(row.get("niqe")),
                brisque=format_metric(row.get("brisque")),
                runtime=latex_escape(str(runtime)),
            )
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", ""])
    return "\n".join(lines)


def format_metric(value: object) -> str:
    f = safe_float(value)
    return "NA" if f is None else f"{f:.4f}"


def latex_escape(text: str) -> str:
    return text.replace("_", r"\_").replace("%", r"\%").replace("&", r"\&")


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate DreamScene360 Table 1 style metrics")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("table1_eval"))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--metrics", nargs="+", default=["clip", "qalign", "niqe", "brisque"],
                        choices=["clip", "qalign", "niqe", "brisque"])
    parser.add_argument("--clip-backend", default="auto", choices=["auto", "transformers", "open_clip"])
    parser.add_argument("--clip-model", default="openai/clip-vit-base-patch32",
                        help="Transformers model id, or open_clip model/pretrained like ViT-B-32/openai")
    parser.add_argument("--qalign-model", default="q-future/one-align")
    parser.add_argument("--qalign-vendored-root", type=Path,
                        default=Path("../VideoScore2/eval/eval_methods/utils_q_align"),
                        help="Path containing the q_align package; set empty string to disable vendored path")
    parser.add_argument("--qalign-scale", type=float, default=5.0,
                        help="Q-Align scorer returns 0-1 in the vendored implementation; Table 1 uses 0-5")
    parser.add_argument("--fail-on-missing-metric", action="store_true")
    parser.add_argument("--skip-metric-errors", action="store_true",
                        help="Skip per-image metric failures and record them in metric_errors.json")
    args = parser.parse_args()

    device = torch.device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    items = read_manifest(args.manifest)
    if not items:
        raise ValueError("Manifest did not produce any evaluation items")

    scorers: dict[str, object] = {}
    metric_errors: dict[str, str] = {}

    def register(name: str, factory: Callable[[], object]) -> None:
        if name not in args.metrics:
            return
        try:
            scorers[name] = factory()
        except Exception as exc:  # noqa: BLE001
            metric_errors[name] = str(exc)
            if args.fail_on_missing_metric:
                raise

    register("clip", lambda: ClipScorer(args.clip_backend, args.clip_model, device))
    register("niqe", lambda: PyiqaScorer("niqe", device))
    register("brisque", lambda: PyiqaScorer("brisque", device))

    qalign_root = args.qalign_vendored_root
    if qalign_root is not None and str(qalign_root).strip() == "":
        qalign_root = None
    elif qalign_root is not None and not qalign_root.is_absolute():
        qalign_root = (Path.cwd() / qalign_root).resolve()
    register("qalign", lambda: QAlignScorer(args.qalign_model, device, qalign_root, args.qalign_scale))

    per_image_rows: list[dict[str, object]] = []
    for item in items:
        image = load_rgb(item.image_path)
        row: dict[str, object] = {
            "method": item.method,
            "scene": item.scene,
            "prompt": item.prompt,
            "image_path": str(item.image_path),
            "runtime_sec": item.runtime_sec if item.runtime_sec is not None else "",
        }

        def score_or_record(metric_name: str, row_key: str, scorer_call: Callable[[], float]) -> None:
            try:
                row[row_key] = scorer_call()
            except Exception as exc:  # noqa: BLE001
                if not args.skip_metric_errors:
                    raise
                error_key = f"{metric_name}:{item.image_path}"
                metric_errors[error_key] = str(exc)
                row[row_key] = ""

        if "clip" in scorers and item.prompt:
            score_or_record(
                "clip",
                "clip_distance",
                lambda: scorers["clip"].score(image, item.prompt),  # type: ignore[attr-defined]
            )
        if "qalign" in scorers:
            score_or_record(
                "qalign",
                "q_align",
                lambda: scorers["qalign"].score(image),  # type: ignore[attr-defined]
            )
        if "niqe" in scorers:
            score_or_record(
                "niqe",
                "niqe",
                lambda: scorers["niqe"].score(image),  # type: ignore[attr-defined]
            )
        if "brisque" in scorers:
            score_or_record(
                "brisque",
                "brisque",
                lambda: scorers["brisque"].score(image),  # type: ignore[attr-defined]
            )

        per_image_rows.append(row)

    per_image_fields = [
        "method", "scene", "prompt", "image_path", "runtime_sec",
        "clip_distance", "q_align", "niqe", "brisque",
    ]
    write_csv(args.output_dir / "metrics_per_image.csv", per_image_rows, per_image_fields)

    methods = sorted({item.method for item in items})
    summary_rows: list[dict[str, object]] = []
    for method in methods:
        method_rows = [row for row in per_image_rows if row["method"] == method]
        runtime_values = {
            float(row["runtime_sec"])
            for row in method_rows
            if str(row.get("runtime_sec", "")).strip() != ""
        }
        runtime_mean = mean(runtime_values)
        summary_rows.append(
            {
                "method": method,
                "num_images": len(method_rows),
                "clip_distance": mean_metric(method_rows, "clip_distance"),
                "q_align": mean_metric(method_rows, "q_align"),
                "niqe": mean_metric(method_rows, "niqe"),
                "brisque": mean_metric(method_rows, "brisque"),
                "runtime_sec": runtime_mean if not math.isnan(runtime_mean) else "",
                "runtime": format_runtime(runtime_mean) if not math.isnan(runtime_mean) else "",
            }
        )

    summary_fields = [
        "method", "num_images", "clip_distance", "q_align", "niqe",
        "brisque", "runtime_sec", "runtime",
    ]
    write_csv(args.output_dir / "summary.csv", summary_rows, summary_fields)
    (args.output_dir / "table1.md").write_text(build_markdown_table(summary_rows), encoding="utf-8")
    (args.output_dir / "table1.tex").write_text(build_latex_table(summary_rows), encoding="utf-8")
    (args.output_dir / "metric_errors.json").write_text(
        json.dumps(metric_errors, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    if metric_errors:
        print("Some metrics were skipped because their dependencies/models are missing:")
        for name, error in metric_errors.items():
            print(f"  - {name}: {error}")
    print(f"Wrote results to {args.output_dir}")
    return 0


def mean_metric(rows: list[dict[str, object]], key: str) -> float:
    values = []
    for row in rows:
        value = safe_float(row.get(key))
        if value is not None:
            values.append(value)
    return mean(values)


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

import argparse
import shutil
import subprocess
import tempfile
from pathlib import Path

import yaml


def make_config(config_path: Path, weights_dir: str | None, output_path: Path) -> Path:
    if not weights_dir:
        return config_path

    with config_path.open("r", encoding="utf-8") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    config["load_weights_dir"] = weights_dir

    with output_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, sort_keys=False)
    return output_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Run DAP on one image and write a depth .npy")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--root", default="/mnt/data/wangqq/DAP")
    parser.add_argument("--config", default="")
    parser.add_argument("--weights-dir", default="")
    parser.add_argument("--gpu", default="0")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    config_path = Path(args.config).resolve() if args.config else root / "config" / "infer.yaml"
    infer_script = root / "test" / "infer.py"
    output_path = Path(args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not infer_script.exists():
        raise FileNotFoundError(f"DAP infer.py not found: {infer_script}")
    if not config_path.exists():
        raise FileNotFoundError(f"DAP config not found: {config_path}")

    with tempfile.TemporaryDirectory(prefix="dap_external_") as tmp:
        tmp_dir = Path(tmp)
        txt_path = tmp_dir / "input.txt"
        out_dir = tmp_dir / "dap_output"
        local_config = make_config(config_path, args.weights_dir or None, tmp_dir / "infer_local.yaml")

        txt_path.write_text(str(Path(args.input).resolve()) + "\n", encoding="utf-8")
        subprocess.run(
            [
                "python",
                str(infer_script),
                "--config",
                str(local_config),
                "--txt",
                str(txt_path),
                "--output",
                str(out_dir),
                "--gpu",
                args.gpu,
            ],
            cwd=str(root),
            check=True,
        )

        depth_path = out_dir / "depth_npy" / "000001.npy"
        if not depth_path.exists():
            raise FileNotFoundError(f"DAP did not write expected depth file: {depth_path}")
        shutil.copyfile(depth_path, output_path)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

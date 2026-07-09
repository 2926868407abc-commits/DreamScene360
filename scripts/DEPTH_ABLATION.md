# Depth Predictor Ablation

DreamScene360 now accepts these `--depth_predictor` values:

- `omnidata`: original DreamScene360 default.
- `g2vlm`: existing G2VLM adapter.
- `depth_anything3`: official Depth Anything 3 adapter.
- `dap`: external command adapter.
- `vggt_omega`: local VGGT adapter.

## One-command ablation runner

First prepare the external environments:

```bash
conda activate /mnt/data/wangqq/conda_envs/dreamscene360
cd /mnt/data/wangqq/DreamScene360

bash scripts/setup_depth_ablation_envs.sh
```

Then run the full pipeline:

```bash
conda activate /mnt/data/wangqq/conda_envs/dreamscene360
cd /mnt/data/wangqq/DreamScene360

bash scripts/run_depth_ablation_t01.sh
```

The runner first smoke-tests all requested depth predictors, then trains,
renders paper-style views, evaluates metrics, and merges the final table. It
uses this shared view setting for every method:

```bash
--view-mode paper \
--seed 0 \
--paper-pitch-degrees 10 \
--translation-radius 0.1
```

By default `AUTO_SKIP_UNAVAILABLE=1`, so unavailable methods are skipped with a
message instead of stopping the whole run. For example, if DAP has no
`DAP_DEPTH_COMMAND` or VGGT has no local repo, the runner will still evaluate
the ready methods such as `omnidata` and `depth_anything3`.

Useful switches:

```bash
# Only test whether predictors can run.
RUN_TRAIN=0 RUN_RENDER=0 RUN_METRICS=0 bash scripts/run_depth_ablation_t01.sh

# Run a subset.
METHODS="omnidata depth_anything3" bash scripts/run_depth_ablation_t01.sh

# Require every requested method to be available.
AUTO_SKIP_UNAVAILABLE=0 bash scripts/run_depth_ablation_t01.sh

# Quick debugging run.
ITERATIONS=100 RUN_METRICS=0 bash scripts/run_depth_ablation_t01.sh
```

## Depth Anything 3

The original DreamScene360 environment uses Python 3.8, while the official
Depth Anything 3 package requires Python >= 3.9. Keep the DreamScene360
environment unchanged and install Depth Anything 3 in a separate environment:

```bash
conda create -p /mnt/data/wangqq/conda_envs/depth_anything3 python=3.10 -y
conda activate /mnt/data/wangqq/conda_envs/depth_anything3

python -m pip install --upgrade pip
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

cd /mnt/data/wangqq

git clone https://github.com/ByteDance-Seed/depth-anything-3.git
cd depth-anything-3
python -m pip install -e .
```

Then run DreamScene360 in its original environment and call the Depth Anything 3
environment through `--depth_anything3_command`. Do not pass the placeholder
`YOUR_DEPTH_ANYTHING3_MODEL_ID`.

```bash
conda activate /mnt/data/wangqq/conda_envs/dreamscene360
cd /mnt/data/wangqq/DreamScene360

export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME=/mnt/data/wangqq/hf_cache
export TORCH_HOME=/mnt/data/wangqq/torch_cache

python train.py \
  -s data/YOUR_SCENE \
  -m output/YOUR_SCENE_da3 \
  --depth_predictor depth_anything3 \
  --depth_anything3_model depth-anything/DA3-LARGE-1.1 \
  --depth_anything3_command "/mnt/data/wangqq/conda_envs/depth_anything3/bin/python /mnt/data/wangqq/DreamScene360/scripts/run_depth_anything3_external.py --input-dir {input_dir} --output-dir {output_dir} --model {model_id}" \
  --iterations 10000
```

You can also set `DEPTH_ANYTHING3_MODEL` instead of passing
`--depth_anything3_model`.

## DAP

DAP is connected as an external command because the exact DAP repository/API is
not fixed here. The command must write a depth file to `{output}`. The output
path is currently `depth.npy`.

```bash
export DAP_DEPTH_COMMAND='python /path/to/dap/infer.py --image {input} --output {output}'

python train.py \
  -s data/YOUR_SCENE \
  -m output/YOUR_SCENE_dap \
  --depth_predictor dap \
  --iterations 10000
```

The command template can use:

- `{input}`: temporary RGB PNG.
- `{output}`: temporary output path, usually `depth.npy`.
- `{intrinsics}`: JSON file with `fx/fy/cx/cy`.
- `{root}`: value of `--dap_root` or `DAP_ROOT`.
- `{model_path}`: value of `--dap_model_path` or `DAP_MODEL_PATH`.

## VGGT / VGGT-Omega

By default this expects a local VGGT repo at `/mnt/data/wangqq/vggt` or next to
DreamScene360. Set `VGGT_ROOT` or pass `--vggt_root` if needed.

```bash
export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME=/mnt/data/wangqq/hf_cache
export VGGT_ROOT=/mnt/data/wangqq/vggt
export VGGT_CHUNK_SIZE=8

python train.py \
  -s data/YOUR_SCENE \
  -m output/YOUR_SCENE_vggt_omega \
  --depth_predictor vggt_omega \
  --vggt_model_path facebook/VGGT-1B \
  --iterations 10000
```

For a local VGGT checkpoint:

```bash
python train.py \
  -s data/YOUR_SCENE \
  -m output/YOUR_SCENE_vggt_omega \
  --depth_predictor vggt_omega \
  --vggt_root /mnt/data/wangqq/vggt \
  --vggt_model_path /path/to/model.pt \
  --vggt_chunk_size 8 \
  --iterations 10000
```

Set `--vggt_chunk_size 20` to let VGGT process all 20 perspective supervision
views jointly. This is closer to a multi-view setting but uses much more VRAM.

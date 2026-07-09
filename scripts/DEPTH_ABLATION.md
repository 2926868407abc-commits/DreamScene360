# Depth Predictor Ablation

DreamScene360 now accepts these `--depth_predictor` values:

- `omnidata`: original DreamScene360 default.
- `g2vlm`: existing G2VLM adapter.
- `depth_anything3`: Hugging Face depth-estimation adapter.
- `dap`: external command adapter.
- `vggt_omega`: local VGGT adapter.

## Depth Anything 3

Set the exact Hugging Face model id used by your Depth Anything 3 checkout:

```bash
export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME=/mnt/data/wangqq/hf_cache

python train.py \
  -s data/YOUR_SCENE \
  -m output/YOUR_SCENE_da3 \
  --depth_predictor depth_anything3 \
  --depth_anything3_model YOUR_DEPTH_ANYTHING3_MODEL_ID \
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

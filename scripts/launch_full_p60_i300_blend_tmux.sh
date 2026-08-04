#!/usr/bin/env bash
set -euo pipefail

cd /mnt/data/wangqq/DreamScene360

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-3}"
export PYTHONUNBUFFERED=1
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HOME="${HF_HOME:-/mnt/data/wangqq/hf_cache}"
export TORCH_HOME="${TORCH_HOME:-/mnt/data/wangqq/torch_cache}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

bash scripts/run_full_p60_i300_blend.sh > full_p60_i300_blend.log 2>&1

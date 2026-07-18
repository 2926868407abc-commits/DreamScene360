#!/usr/bin/env bash
set -euo pipefail

# Run a local same-sample comparison:
#   direct panoramic depth baseline vs DreamScene360 perspective/PanoGeo fusion.
#
# This avoids comparing local 5-sample fusion results against paper Table 3
# numbers measured on a different evaluation set.

PROJECT_DIR="${PROJECT_DIR:-/mnt/data/wangqq/DreamScene360}"
DREAM_ENV="${DREAM_ENV:-/mnt/data/wangqq/conda_envs/dreamscene360}"
DA3_ENV="${DA3_ENV:-/mnt/data/wangqq/conda_envs/depth_anything3}"
DAP_ENV="${DAP_ENV:-/mnt/data/wangqq/conda_envs/dap}"

MANIFEST="${MANIFEST:-panorama_depth_manifest_check.csv}"
DATASETS="${DATASETS:-Matterport3D,Stanford2D3D}"
MAX_PER_DATASET="${MAX_PER_DATASET:-5}"
MAX_DEPTH="${MAX_DEPTH:-100}"

DIRECT_METHOD="${DIRECT_METHOD:-dap}"
DIRECT_LABEL="${DIRECT_LABEL:-DAP-Direct-LocalSamples}"
DIRECT_OUTPUT_DIR="${DIRECT_OUTPUT_DIR:-panorama_depth_eval_dap_direct_balanced5}"

FUSION_DIR="${FUSION_DIR:-panorama_depth_eval_dreamscene360_pano_geo_balanced5_depth_anything3_none}"
FUSION_LABEL="${FUSION_LABEL:-DreamScene360-PanoGeo-DA3}"
COMPARE_OUTPUT_DIR="${COMPARE_OUTPUT_DIR:-panorama_depth_compare_da3_pano_geo_vs_dap_direct_local}"

RUN_FUSION="${RUN_FUSION:-0}"
RUN_DIRECT="${RUN_DIRECT:-1}"
RUN_COMPARE="${RUN_COMPARE:-1}"
SAVE_PREDICTIONS="${SAVE_PREDICTIONS:-1}"

FUSION_INNER_DEPTH_PREDICTOR="${FUSION_INNER_DEPTH_PREDICTOR:-depth_anything3}"
FUSION_EVAL_ALIGN="${FUSION_EVAL_ALIGN:-none}"
FUSION_PREDICTION_SCALE="${FUSION_PREDICTION_SCALE:-1.0}"
PANO_GEO_GEN_RES="${PANO_GEO_GEN_RES:-512}"
PANO_GEO_REG_LOSS_WEIGHT="${PANO_GEO_REG_LOSS_WEIGHT:-0.1}"
PANO_GEO_DEPTH_NORMALIZE="${PANO_GEO_DEPTH_NORMALIZE:-none}"
PANO_GEO_ITERS="${PANO_GEO_ITERS:-1500}"
PANO_GEO_NUM_PERSPECTIVES="${PANO_GEO_NUM_PERSPECTIVES:-20}"
SEED="${SEED:-0}"

DAP_ROOT="${DAP_ROOT:-/mnt/data/wangqq/DAP}"
DAP_WEIGHTS_DIR="${DAP_WEIGHTS_DIR:-/mnt/data/wangqq/DAP-weights}"
DAP_DEPTH_COMMAND="${DAP_DEPTH_COMMAND:-}"
if [[ -z "$DAP_DEPTH_COMMAND" && -x "${DAP_ENV}/bin/python" && -f "${DAP_ROOT}/test/infer.py" ]]; then
  DAP_DEPTH_COMMAND="${DAP_ENV}/bin/python ${PROJECT_DIR}/scripts/run_dap_external.py --input {input} --output {output} --root ${DAP_ROOT} --weights-dir ${DAP_WEIGHTS_DIR}"
fi

DEPTH_ANYTHING3_MODEL="${DEPTH_ANYTHING3_MODEL:-depth-anything/DA3-LARGE-1.1}"
DEPTH_ANYTHING3_COMMAND="${DEPTH_ANYTHING3_COMMAND:-}"
if [[ -z "$DEPTH_ANYTHING3_COMMAND" ]]; then
  DEPTH_ANYTHING3_COMMAND="${DA3_ENV}/bin/python ${PROJECT_DIR}/scripts/run_depth_anything3_external.py --input-dir {input_dir} --output-dir {output_dir} --model {model_id}"
fi

VGGT_ROOT="${VGGT_ROOT:-/mnt/data/wangqq/vggt}"
VGGT_MODEL_PATH="${VGGT_MODEL_PATH:-facebook/VGGT-1B}"
VGGT_CHUNK_SIZE="${VGGT_CHUNK_SIZE:-8}"

G2VLM_ROOT="${G2VLM_ROOT:-/mnt/data/wangqq/G2VLM}"
G2VLM_MODEL_PATH="${G2VLM_MODEL_PATH:-/mnt/data/wangqq/G2VLM/models/G2VLM-2B-MoT}"

HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
HF_HOME="${HF_HOME:-/mnt/data/wangqq/hf_cache}"
TORCH_HOME="${TORCH_HOME:-/mnt/data/wangqq/torch_cache}"
export HF_ENDPOINT HF_HOME TORCH_HOME

cd "$PROJECT_DIR"

PYTHON_BIN="${DREAM_ENV}/bin/python"
if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "[error] python not found: $PYTHON_BIN" >&2
  exit 1
fi

if [[ ! -f "$MANIFEST" ]]; then
  echo "[error] manifest not found: $MANIFEST" >&2
  exit 1
fi

direct_extra_args=()
case "$DIRECT_METHOD" in
  dap)
    if [[ -z "$DAP_DEPTH_COMMAND" ]]; then
      echo "[error] DAP_DEPTH_COMMAND is empty. Check DAP_ENV, DAP_ROOT, and DAP_WEIGHTS_DIR." >&2
      exit 1
    fi
    direct_extra_args+=(--dap-root "$DAP_ROOT" --dap-command "$DAP_DEPTH_COMMAND")
    ;;
  depth_anything3|da3)
    direct_extra_args+=(
      --depth-anything3-model "$DEPTH_ANYTHING3_MODEL"
      --depth-anything3-command "$DEPTH_ANYTHING3_COMMAND"
    )
    ;;
  vggt_omega|vggt)
    direct_extra_args+=(
      --vggt-root "$VGGT_ROOT"
      --vggt-model-path "$VGGT_MODEL_PATH"
      --vggt-chunk-size "$VGGT_CHUNK_SIZE"
    )
    ;;
  g2vlm)
    direct_extra_args+=(
      --g2vlm-root "$G2VLM_ROOT"
      --g2vlm-model-path "$G2VLM_MODEL_PATH"
    )
    ;;
  *)
    echo "[error] unknown DIRECT_METHOD: $DIRECT_METHOD" >&2
    exit 1
    ;;
esac

echo "[config] project: $PROJECT_DIR"
echo "[config] manifest: $MANIFEST"
echo "[config] datasets: $DATASETS"
echo "[config] max per dataset: $MAX_PER_DATASET"
echo "[config] direct method: $DIRECT_METHOD"
echo "[config] direct output: $DIRECT_OUTPUT_DIR"
echo "[config] fusion dir: $FUSION_DIR"
echo "[config] run fusion: $RUN_FUSION"
echo "[config] fusion inner depth predictor: $FUSION_INNER_DEPTH_PREDICTOR"
echo "[config] pano geo num perspectives: $PANO_GEO_NUM_PERSPECTIVES"
echo "[config] compare output: $COMPARE_OUTPUT_DIR"

if [[ "$RUN_FUSION" == "1" ]]; then
  mkdir -p "$FUSION_DIR"
  fusion_args=(
    scripts/evaluate_panorama_depth_fusion.py
    --manifest "$MANIFEST"
    --output-dir "$FUSION_DIR"
    --method dreamscene360
    --method-label "$FUSION_LABEL"
    --datasets "$DATASETS"
    --max-per-dataset "$MAX_PER_DATASET"
    --dreamscene360-depth-predictor "$FUSION_INNER_DEPTH_PREDICTOR"
    --pano-geo-gen-res "$PANO_GEO_GEN_RES"
    --pano-geo-reg-loss-weight "$PANO_GEO_REG_LOSS_WEIGHT"
    --pano-geo-depth-normalize "$PANO_GEO_DEPTH_NORMALIZE"
    --pano-geo-iters "$PANO_GEO_ITERS"
    --pano-geo-num-perspectives "$PANO_GEO_NUM_PERSPECTIVES"
    --eval-align "$FUSION_EVAL_ALIGN"
    --prediction-scale "$FUSION_PREDICTION_SCALE"
    --max-depth "$MAX_DEPTH"
    --seed "$SEED"
    --depth-anything3-model "$DEPTH_ANYTHING3_MODEL"
    --depth-anything3-command "$DEPTH_ANYTHING3_COMMAND"
    --dap-root "$DAP_ROOT"
    --dap-command "$DAP_DEPTH_COMMAND"
    --vggt-root "$VGGT_ROOT"
    --vggt-model-path "$VGGT_MODEL_PATH"
    --vggt-chunk-size "$VGGT_CHUNK_SIZE"
    --g2vlm-root "$G2VLM_ROOT"
    --g2vlm-model-path "$G2VLM_MODEL_PATH"
  )
  if [[ "$SAVE_PREDICTIONS" == "1" ]]; then
    fusion_args+=(--save-predictions)
  fi

  echo "[stage] run DreamScene360/PanoGeo perspective fusion"
  PYTHONUNBUFFERED=1 "$PYTHON_BIN" -u "${fusion_args[@]}" \
    2>&1 | tee "$FUSION_DIR/run.log"
fi

if [[ "$RUN_DIRECT" == "1" ]]; then
  mkdir -p "$DIRECT_OUTPUT_DIR"
  direct_args=(
    scripts/evaluate_panorama_depth_fusion.py
    --manifest "$MANIFEST"
    --output-dir "$DIRECT_OUTPUT_DIR"
    --method "$DIRECT_METHOD"
    --eval-mode direct_panorama
    --method-label "$DIRECT_LABEL"
    --datasets "$DATASETS"
    --max-per-dataset "$MAX_PER_DATASET"
    --eval-align none
    --max-depth "$MAX_DEPTH"
    "${direct_extra_args[@]}"
  )
  if [[ "$SAVE_PREDICTIONS" == "1" ]]; then
    direct_args+=(--save-predictions)
  fi

  echo "[stage] run direct panorama baseline"
  PYTHONUNBUFFERED=1 "$PYTHON_BIN" -u "${direct_args[@]}" \
    2>&1 | tee "$DIRECT_OUTPUT_DIR/run.log"
fi

if [[ "$RUN_COMPARE" == "1" ]]; then
  if [[ ! -f "$DIRECT_OUTPUT_DIR/metrics_per_image.csv" ]]; then
    echo "[error] missing direct metrics: $DIRECT_OUTPUT_DIR/metrics_per_image.csv" >&2
    exit 1
  fi
  if [[ ! -f "$FUSION_DIR/metrics_per_image.csv" ]]; then
    echo "[error] missing fusion metrics: $FUSION_DIR/metrics_per_image.csv" >&2
    exit 1
  fi

  echo "[stage] compare fusion vs direct"
  "$PYTHON_BIN" scripts/compare_panorama_depth_runs.py \
    --candidate-dir "$FUSION_DIR" \
    --baseline-dir "$DIRECT_OUTPUT_DIR" \
    --output-dir "$COMPARE_OUTPUT_DIR" \
    --candidate-label "$FUSION_LABEL" \
    --baseline-label "$DIRECT_LABEL"

  echo "[done] $COMPARE_OUTPUT_DIR/coverage.md"
  echo "[done] $COMPARE_OUTPUT_DIR/comparison_summary.md"
fi

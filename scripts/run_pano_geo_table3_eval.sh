#!/usr/bin/env bash
set -euo pipefail

# Evaluate DreamScene360's PanoGeo panorama-depth optimization on Table 3 style
# panoramic depth metrics. This is different from the simple splat fusion path:
# it calls geo_predictors.PanoGeoPredictor, which is the depth alignment and
# panorama-depth optimization used by DreamScene360.

PROJECT_DIR="${PROJECT_DIR:-/mnt/data/wangqq/DreamScene360}"
DREAM_ENV="${DREAM_ENV:-/mnt/data/wangqq/conda_envs/dreamscene360}"
DA3_ENV="${DA3_ENV:-/mnt/data/wangqq/conda_envs/depth_anything3}"
DAP_ENV="${DAP_ENV:-/mnt/data/wangqq/conda_envs/dap}"

MANIFEST="${MANIFEST:-panorama_depth_manifest_check.csv}"
OUTPUT_PREFIX="${OUTPUT_PREFIX:-panorama_depth_eval_dreamscene360_pano_geo}"

INNER_DEPTH_PREDICTOR="${INNER_DEPTH_PREDICTOR:-omnidata}"
METHOD_LABEL="${METHOD_LABEL:-DreamScene360-PanoGeo-${INNER_DEPTH_PREDICTOR}-Metric}"

GEN_RES="${GEN_RES:-512}"
REG_LOSS_WEIGHT="${REG_LOSS_WEIGHT:-0.1}"
PANO_GEO_DEPTH_NORMALIZE="${PANO_GEO_DEPTH_NORMALIZE:-mean}"
PANO_GEO_ITERS="${PANO_GEO_ITERS:-1500}"
PANO_GEO_NUM_PERSPECTIVES="${PANO_GEO_NUM_PERSPECTIVES:-20}"
EVAL_ALIGN="${EVAL_ALIGN:-none}"
PREDICTION_SCALE="${PREDICTION_SCALE:-1.0}"
MAX_DEPTH="${MAX_DEPTH:-100}"
SEED="${SEED:-0}"

# Default to the current small check set. Use MAX_ITEMS=0 for the full manifest.
MAX_ITEMS="${MAX_ITEMS:-5}"
MAX_PER_DATASET="${MAX_PER_DATASET:-0}"
DATASETS="${DATASETS:-}"

RUN_SMOKE="${RUN_SMOKE:-1}"
RUN_EVAL="${RUN_EVAL:-1}"
SAVE_PREDICTIONS="${SAVE_PREDICTIONS:-1}"
INCLUDE_TABLE3_BASELINES="${INCLUDE_TABLE3_BASELINES:-1}"

HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
HF_HOME="${HF_HOME:-/mnt/data/wangqq/hf_cache}"
TORCH_HOME="${TORCH_HOME:-/mnt/data/wangqq/torch_cache}"

export HF_ENDPOINT HF_HOME TORCH_HOME

DEPTH_ANYTHING3_MODEL="${DEPTH_ANYTHING3_MODEL:-depth-anything/DA3-LARGE-1.1}"
if [[ -z "${DEPTH_ANYTHING3_COMMAND:-}" ]]; then
  DEPTH_ANYTHING3_COMMAND="${DA3_ENV}/bin/python ${PROJECT_DIR}/scripts/run_depth_anything3_external.py --input-dir {input_dir} --output-dir {output_dir} --model {model_id}"
fi

DAP_ROOT="${DAP_ROOT:-/mnt/data/wangqq/DAP}"
DAP_MODEL_PATH="${DAP_MODEL_PATH:-}"
DAP_WEIGHTS_DIR="${DAP_WEIGHTS_DIR:-/mnt/data/wangqq/DAP-weights}"
DAP_DEPTH_COMMAND="${DAP_DEPTH_COMMAND:-}"
if [[ -z "$DAP_DEPTH_COMMAND" && -x "${DAP_ENV}/bin/python" && -f "${DAP_ROOT}/test/infer.py" ]]; then
  DAP_DEPTH_COMMAND="${DAP_ENV}/bin/python ${PROJECT_DIR}/scripts/run_dap_external.py --input {input} --output {output} --root ${DAP_ROOT} --weights-dir ${DAP_WEIGHTS_DIR}"
fi

VGGT_ROOT="${VGGT_ROOT:-/mnt/data/wangqq/vggt}"
VGGT_MODEL_PATH="${VGGT_MODEL_PATH:-facebook/VGGT-1B}"
VGGT_CHUNK_SIZE="${VGGT_CHUNK_SIZE:-8}"

G2VLM_ROOT="${G2VLM_ROOT:-/mnt/data/wangqq/G2VLM}"
G2VLM_MODEL_PATH="${G2VLM_MODEL_PATH:-/mnt/data/wangqq/G2VLM/models/G2VLM-2B-MoT}"

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

common_args=(
  scripts/evaluate_panorama_depth_fusion.py
  --manifest "$MANIFEST"
  --method dreamscene360
  --method-label "$METHOD_LABEL"
  --datasets "$DATASETS"
  --max-per-dataset "$MAX_PER_DATASET"
  --dreamscene360-depth-predictor "$INNER_DEPTH_PREDICTOR"
  --pano-geo-gen-res "$GEN_RES"
  --pano-geo-reg-loss-weight "$REG_LOSS_WEIGHT"
  --pano-geo-depth-normalize "$PANO_GEO_DEPTH_NORMALIZE"
  --pano-geo-iters "$PANO_GEO_ITERS"
  --pano-geo-num-perspectives "$PANO_GEO_NUM_PERSPECTIVES"
  --eval-align "$EVAL_ALIGN"
  --prediction-scale "$PREDICTION_SCALE"
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

if [[ -n "$DAP_MODEL_PATH" ]]; then
  common_args+=(--dap-model-path "$DAP_MODEL_PATH")
fi

if [[ "$SAVE_PREDICTIONS" == "1" ]]; then
  common_args+=(--save-predictions)
fi

if [[ "$INCLUDE_TABLE3_BASELINES" == "1" ]]; then
  common_args+=(--include-table3-direct-baselines)
fi

echo "[config] project: $PROJECT_DIR"
echo "[config] manifest: $MANIFEST"
echo "[config] inner depth predictor: $INNER_DEPTH_PREDICTOR"
echo "[config] eval align: $EVAL_ALIGN"
echo "[config] prediction scale: $PREDICTION_SCALE"
echo "[config] gen res: $GEN_RES"
echo "[config] pano geo depth normalize: $PANO_GEO_DEPTH_NORMALIZE"
echo "[config] pano geo iters: $PANO_GEO_ITERS"
echo "[config] pano geo num perspectives: $PANO_GEO_NUM_PERSPECTIVES"
echo "[config] datasets: ${DATASETS:-all}"
echo "[config] max items: $MAX_ITEMS"
echo "[config] max per dataset: $MAX_PER_DATASET"

if [[ "$RUN_SMOKE" == "1" ]]; then
  smoke_dir="${OUTPUT_PREFIX}_${INNER_DEPTH_PREDICTOR}_${EVAL_ALIGN}_smoke"
  mkdir -p "$smoke_dir"
  echo "[stage] smoke test one panorama -> $smoke_dir"
  PYTHONUNBUFFERED=1 "$PYTHON_BIN" -u "${common_args[@]}" \
    --output-dir "$smoke_dir" \
    --max-items 1 \
    2>&1 | tee "$smoke_dir/run.log"
fi

if [[ "$RUN_EVAL" == "1" ]]; then
  eval_dir="${OUTPUT_PREFIX}_${INNER_DEPTH_PREDICTOR}_${EVAL_ALIGN}"
  mkdir -p "$eval_dir"
  eval_args=("${common_args[@]}" --output-dir "$eval_dir")
  if [[ "$MAX_ITEMS" != "0" ]]; then
    eval_args+=(--max-items "$MAX_ITEMS")
  fi

  echo "[stage] evaluation -> $eval_dir"
  PYTHONUNBUFFERED=1 "$PYTHON_BIN" -u "${eval_args[@]}" \
    2>&1 | tee "$eval_dir/run.log"
  echo "[done] $eval_dir/table3_style.md"
fi

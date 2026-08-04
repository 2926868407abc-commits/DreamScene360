#!/usr/bin/env bash
set -euo pipefail

# Fixed-scale calibration for Deep360:
# 1. Run direct panorama and DreamScene360/PanoGeo fusion with saved predictions.
# 2. Estimate one fixed dataset scale from the first N predictions.
# 3. Evaluate the remaining predictions with that fixed scale.
# 4. Compare fixed-scale fusion against fixed-scale direct.

PROJECT_DIR="${PROJECT_DIR:-/mnt/data/wangqq/DreamScene360}"
DREAM_ENV="${DREAM_ENV:-/mnt/data/wangqq/conda_envs/dreamscene360}"
DA3_ENV="${DA3_ENV:-/mnt/data/wangqq/conda_envs/depth_anything3}"

MANIFEST="${MANIFEST:-panorama_depth_manifest_deep360_balanced143.csv}"
DATASETS="${DATASETS:-Deep360}"
MAX_PER_DATASET="${MAX_PER_DATASET:-143}"
CALIBRATION_COUNT_PER_DATASET="${CALIBRATION_COUNT_PER_DATASET:-10}"
TEST_COUNT_PER_DATASET="${TEST_COUNT_PER_DATASET:-0}"
MAX_DEPTH="${MAX_DEPTH:-100}"

PANO_GEO_NUM_PERSPECTIVES="${PANO_GEO_NUM_PERSPECTIVES:-24}"
PANO_GEO_ITERS="${PANO_GEO_ITERS:-100}"
PANO_GEO_GEN_RES="${PANO_GEO_GEN_RES:-512}"
PANO_GEO_REG_LOSS_WEIGHT="${PANO_GEO_REG_LOSS_WEIGHT:-0.1}"
PANO_GEO_DEPTH_NORMALIZE="${PANO_GEO_DEPTH_NORMALIZE:-none}"
DEPTH_ANYTHING3_BATCH_SIZE="${DEPTH_ANYTHING3_BATCH_SIZE:-1}"
DEPTH_ANYTHING3_MODEL="${DEPTH_ANYTHING3_MODEL:-depth-anything/DA3-LARGE-1.1}"
DEPTH_ANYTHING3_COMMAND="${DEPTH_ANYTHING3_COMMAND:-}"
if [[ -z "$DEPTH_ANYTHING3_COMMAND" ]]; then
  DEPTH_ANYTHING3_COMMAND="${DA3_ENV}/bin/python ${PROJECT_DIR}/scripts/run_depth_anything3_external.py --input-dir {input_dir} --output-dir {output_dir} --model {model_id} --batch-size ${DEPTH_ANYTHING3_BATCH_SIZE}"
fi

RAW_DIRECT_DIR="${RAW_DIRECT_DIR:-panorama_depth_eval_da3_direct_deep360_fixedscale_raw}"
RAW_FUSION_DIR="${RAW_FUSION_DIR:-panorama_depth_eval_dreamscene360_pano_geo_da3_p24_i100_deep360_fixedscale_raw}"
FIXED_DIRECT_DIR="${FIXED_DIRECT_DIR:-panorama_depth_eval_da3_direct_deep360_fixedscale_calib${CALIBRATION_COUNT_PER_DATASET}}"
FIXED_FUSION_DIR="${FIXED_FUSION_DIR:-panorama_depth_eval_dreamscene360_pano_geo_da3_p24_i100_deep360_fixedscale_calib${CALIBRATION_COUNT_PER_DATASET}}"
COMPARE_DIR="${COMPARE_DIR:-panorama_depth_compare_deep360_fixedscale_calib${CALIBRATION_COUNT_PER_DATASET}_fusion_vs_direct}"

RAW_DIRECT_LABEL="${RAW_DIRECT_LABEL:-DA3-Direct-Deep360-RawForFixedScale}"
RAW_FUSION_LABEL="${RAW_FUSION_LABEL:-DreamScene360-PanoGeo-DA3-P${PANO_GEO_NUM_PERSPECTIVES}-I${PANO_GEO_ITERS}-Deep360-RawForFixedScale}"
FIXED_DIRECT_LABEL="${FIXED_DIRECT_LABEL:-DA3-Direct-Deep360-FixedScale-Calib${CALIBRATION_COUNT_PER_DATASET}}"
FIXED_FUSION_LABEL="${FIXED_FUSION_LABEL:-DreamScene360-PanoGeo-DA3-P${PANO_GEO_NUM_PERSPECTIVES}-I${PANO_GEO_ITERS}-Deep360-FixedScale-Calib${CALIBRATION_COUNT_PER_DATASET}}"

RUN_RAW_DIRECT="${RUN_RAW_DIRECT:-1}"
RUN_RAW_FUSION="${RUN_RAW_FUSION:-1}"
RUN_RESCORE="${RUN_RESCORE:-1}"
RUN_COMPARE="${RUN_COMPARE:-1}"

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
  echo "[stage] build manifest: $MANIFEST"
  "$PYTHON_BIN" scripts/make_panorama_depth_manifest.py \
    --output "$MANIFEST" \
    --datasets "$DATASETS" \
    --max-per-dataset "$MAX_PER_DATASET" \
    --verbose
fi

if [[ "$RUN_RAW_DIRECT" == "1" ]]; then
  echo "[stage] run raw direct with saved predictions -> $RAW_DIRECT_DIR"
  mkdir -p "$RAW_DIRECT_DIR"
  PYTHONUNBUFFERED=1 "$PYTHON_BIN" -u scripts/evaluate_panorama_depth_fusion.py \
    --manifest "$MANIFEST" \
    --output-dir "$RAW_DIRECT_DIR" \
    --method depth_anything3 \
    --eval-mode direct_panorama \
    --method-label "$RAW_DIRECT_LABEL" \
    --datasets "$DATASETS" \
    --max-per-dataset "$MAX_PER_DATASET" \
    --eval-align none \
    --max-depth "$MAX_DEPTH" \
    --depth-anything3-model "$DEPTH_ANYTHING3_MODEL" \
    --depth-anything3-command "$DEPTH_ANYTHING3_COMMAND" \
    --save-predictions \
    2>&1 | tee "$RAW_DIRECT_DIR/run.log"
fi

if [[ "$RUN_RAW_FUSION" == "1" ]]; then
  echo "[stage] run raw fusion with saved predictions -> $RAW_FUSION_DIR"
  mkdir -p "$RAW_FUSION_DIR"
  PYTHONUNBUFFERED=1 "$PYTHON_BIN" -u scripts/evaluate_panorama_depth_fusion.py \
    --manifest "$MANIFEST" \
    --output-dir "$RAW_FUSION_DIR" \
    --method dreamscene360 \
    --method-label "$RAW_FUSION_LABEL" \
    --datasets "$DATASETS" \
    --max-per-dataset "$MAX_PER_DATASET" \
    --dreamscene360-depth-predictor depth_anything3 \
    --pano-geo-gen-res "$PANO_GEO_GEN_RES" \
    --pano-geo-reg-loss-weight "$PANO_GEO_REG_LOSS_WEIGHT" \
    --pano-geo-depth-normalize "$PANO_GEO_DEPTH_NORMALIZE" \
    --pano-geo-iters "$PANO_GEO_ITERS" \
    --pano-geo-num-perspectives "$PANO_GEO_NUM_PERSPECTIVES" \
    --eval-align none \
    --max-depth "$MAX_DEPTH" \
    --depth-anything3-model "$DEPTH_ANYTHING3_MODEL" \
    --depth-anything3-command "$DEPTH_ANYTHING3_COMMAND" \
    --save-predictions \
    2>&1 | tee "$RAW_FUSION_DIR/run.log"
fi

if [[ "$RUN_RESCORE" == "1" ]]; then
  echo "[stage] fixed-scale rescore direct -> $FIXED_DIRECT_DIR"
  "$PYTHON_BIN" scripts/rescore_panorama_depth_predictions.py \
    --manifest "$MANIFEST" \
    --prediction-dir "$RAW_DIRECT_DIR/predictions" \
    --output-dir "$FIXED_DIRECT_DIR" \
    --method-label "$FIXED_DIRECT_LABEL" \
    --calibration-count-per-dataset "$CALIBRATION_COUNT_PER_DATASET" \
    --test-count-per-dataset "$TEST_COUNT_PER_DATASET" \
    --max-depth "$MAX_DEPTH"

  echo "[stage] fixed-scale rescore fusion -> $FIXED_FUSION_DIR"
  "$PYTHON_BIN" scripts/rescore_panorama_depth_predictions.py \
    --manifest "$MANIFEST" \
    --prediction-dir "$RAW_FUSION_DIR/predictions" \
    --output-dir "$FIXED_FUSION_DIR" \
    --method-label "$FIXED_FUSION_LABEL" \
    --calibration-count-per-dataset "$CALIBRATION_COUNT_PER_DATASET" \
    --test-count-per-dataset "$TEST_COUNT_PER_DATASET" \
    --max-depth "$MAX_DEPTH"
fi

if [[ "$RUN_COMPARE" == "1" ]]; then
  echo "[stage] compare fixed-scale fusion vs direct -> $COMPARE_DIR"
  "$PYTHON_BIN" scripts/compare_panorama_depth_runs.py \
    --candidate-dir "$FIXED_FUSION_DIR" \
    --baseline-dir "$FIXED_DIRECT_DIR" \
    --output-dir "$COMPARE_DIR" \
    --candidate-label "$FIXED_FUSION_LABEL" \
    --baseline-label "$FIXED_DIRECT_LABEL"
fi

echo "[done] raw direct: $RAW_DIRECT_DIR/table3_style.md"
echo "[done] raw fusion: $RAW_FUSION_DIR/table3_style.md"
echo "[done] fixed direct: $FIXED_DIRECT_DIR/table3_style.md"
echo "[done] fixed fusion: $FIXED_FUSION_DIR/table3_style.md"
echo "[done] comparison: $COMPARE_DIR/comparison_summary.md"

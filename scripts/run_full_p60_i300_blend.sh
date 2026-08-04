#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/mnt/data/wangqq/DreamScene360}"
DREAM_ENV="${DREAM_ENV:-/mnt/data/wangqq/conda_envs/dreamscene360}"
DA3_ENV="${DA3_ENV:-/mnt/data/wangqq/conda_envs/depth_anything3}"

MANIFEST="${MANIFEST:-panorama_depth_manifest_mp_s2d3d_balanced143.csv}"
DATASETS="${DATASETS:-Matterport3D,Stanford2D3D}"
MAX_PER_DATASET="${MAX_PER_DATASET:-143}"
CALIBRATION_COUNT_PER_DATASET="${CALIBRATION_COUNT_PER_DATASET:-10}"
MAX_DEPTH="${MAX_DEPTH:-100}"

DIRECT_RAW_DIR="${DIRECT_RAW_DIR:-panorama_depth_eval_da3_direct_mp_s2d3d_fixedscale_raw}"
DIRECT_FIXED_DIR="${DIRECT_FIXED_DIR:-panorama_depth_eval_da3_direct_mp_s2d3d_fixedscale_calib10}"
FUSION_RAW_DIR="${FUSION_RAW_DIR:-panorama_depth_eval_dreamscene360_pano_geo_da3_p60_i300_mp_s2d3d_blend_raw}"
BLEND_DIR="${BLEND_DIR:-panorama_depth_eval_blend_direct_p60_i300_mp_s2d3d_alpha04}"
COMPARE_DIR="${COMPARE_DIR:-panorama_depth_compare_blend_direct_p60_i300_mp_s2d3d_alpha04}"

PANO_GEO_NUM_PERSPECTIVES="${PANO_GEO_NUM_PERSPECTIVES:-60}"
PANO_GEO_ITERS="${PANO_GEO_ITERS:-300}"
PANO_GEO_GEN_RES="${PANO_GEO_GEN_RES:-512}"
PANO_GEO_REG_LOSS_WEIGHT="${PANO_GEO_REG_LOSS_WEIGHT:-0.1}"
PANO_GEO_DEPTH_NORMALIZE="${PANO_GEO_DEPTH_NORMALIZE:-none}"
DEPTH_ANYTHING3_BATCH_SIZE="${DEPTH_ANYTHING3_BATCH_SIZE:-1}"
DEPTH_ANYTHING3_MODEL="${DEPTH_ANYTHING3_MODEL:-depth-anything/DA3-LARGE-1.1}"

ALPHA_MATTERPORT="${ALPHA_MATTERPORT:-0.4}"
ALPHA_STANFORD="${ALPHA_STANFORD:-0.4}"
RUN_FUSION="${RUN_FUSION:-1}"
RUN_BLEND="${RUN_BLEND:-1}"

HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
HF_HOME="${HF_HOME:-/mnt/data/wangqq/hf_cache}"
TORCH_HOME="${TORCH_HOME:-/mnt/data/wangqq/torch_cache}"
PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export HF_ENDPOINT HF_HOME TORCH_HOME PYTORCH_CUDA_ALLOC_CONF

cd "$PROJECT_DIR"
PYTHON_BIN="${DREAM_ENV}/bin/python"
DEPTH_ANYTHING3_COMMAND="${DEPTH_ANYTHING3_COMMAND:-}"
if [[ -z "$DEPTH_ANYTHING3_COMMAND" ]]; then
  DEPTH_ANYTHING3_COMMAND="${DA3_ENV}/bin/python ${PROJECT_DIR}/scripts/run_depth_anything3_external.py --input-dir {input_dir} --output-dir {output_dir} --model {model_id} --batch-size ${DEPTH_ANYTHING3_BATCH_SIZE}"
fi

expected_count=$((MAX_PER_DATASET * 2))

echo "[config] project: $PROJECT_DIR"
echo "[config] gpu: ${CUDA_VISIBLE_DEVICES:-unset}"
echo "[config] manifest: $MANIFEST"
echo "[config] fusion raw: $FUSION_RAW_DIR"
echo "[config] blend: $BLEND_DIR"
echo "[config] compare: $COMPARE_DIR"
echo "[config] p60/i300: perspectives=$PANO_GEO_NUM_PERSPECTIVES iters=$PANO_GEO_ITERS"
echo "[config] alpha: Matterport3D=$ALPHA_MATTERPORT Stanford2D3D=$ALPHA_STANFORD"

if [[ "$RUN_FUSION" == "1" ]]; then
  existing=0
  if [[ -d "$FUSION_RAW_DIR/predictions" ]]; then
    existing=$(find "$FUSION_RAW_DIR/predictions" -name "*_pred.npy" | wc -l)
  fi
  if [[ "$existing" -ge "$expected_count" ]]; then
    echo "[skip] fusion predictions already exist: $existing/$expected_count"
  else
    echo "[stage] run P60-I300 raw fusion -> $FUSION_RAW_DIR"
    mkdir -p "$FUSION_RAW_DIR"
    PYTHONUNBUFFERED=1 "$PYTHON_BIN" -u scripts/evaluate_panorama_depth_fusion.py \
      --manifest "$MANIFEST" \
      --output-dir "$FUSION_RAW_DIR" \
      --method dreamscene360 \
      --method-label "DreamScene360-PanoGeo-DA3-P60-I300-MP-S2D3D-RawForBlend" \
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
      2>&1 | tee "$FUSION_RAW_DIR/run.log"
  fi
fi

if [[ "$RUN_BLEND" == "1" ]]; then
  echo "[stage] blend direct raw + P60-I300 fusion raw -> $BLEND_DIR"
  "$PYTHON_BIN" scripts/blend_panorama_depth_predictions.py \
    --manifest "$MANIFEST" \
    --direct-prediction-dir "$DIRECT_RAW_DIR/predictions" \
    --fusion-prediction-dir "$FUSION_RAW_DIR/predictions" \
    --output-dir "$BLEND_DIR" \
    --method-label "DA3-DirectGuided-PanoGeo-P60-I300-MP-S2D3D-Alpha04" \
    --calibration-count-per-dataset "$CALIBRATION_COUNT_PER_DATASET" \
    --alpha "Matterport3D=$ALPHA_MATTERPORT" \
    --alpha "Stanford2D3D=$ALPHA_STANFORD" \
    --max-depth "$MAX_DEPTH"

  echo "[stage] compare blend vs direct fixed-scale -> $COMPARE_DIR"
  "$PYTHON_BIN" scripts/compare_panorama_depth_runs.py \
    --candidate-dir "$BLEND_DIR" \
    --baseline-dir "$DIRECT_FIXED_DIR" \
    --output-dir "$COMPARE_DIR" \
    --candidate-label "DA3-DirectGuided-PanoGeo-P60-I300-MP-S2D3D-Alpha04" \
    --baseline-label "DA3-Direct-MP-S2D3D-FixedScale-Calib10"

  echo "[done] blend table: $BLEND_DIR/table3_style.md"
  echo "[done] comparison: $COMPARE_DIR/comparison_summary.md"
  cat "$BLEND_DIR/table3_style.md"
  cat "$COMPARE_DIR/comparison_summary.md"
fi

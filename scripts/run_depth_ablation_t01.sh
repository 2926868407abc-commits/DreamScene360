#!/usr/bin/env bash
set -euo pipefail

# One-command depth-estimator ablation runner.
#
# Default methods:
#   omnidata depth_anything3 dap vggt_omega
#
# DAP is an external-command adapter. Before including dap in METHODS, set:
#   export DAP_DEPTH_COMMAND='python /path/to/dap/infer.py --image {input} --output {output}'

PROJECT_DIR="${PROJECT_DIR:-/mnt/data/wangqq/DreamScene360}"
SCENE_NAME="${SCENE_NAME:-alley}"
SOURCE_PATH="${SOURCE_PATH:-data/alley_pano}"
METHODS="${METHODS:-omnidata depth_anything3 dap vggt_omega}"

OUTPUT_PREFIX="${OUTPUT_PREFIX:-ablation_t01}"
VIEW_TAG="${VIEW_TAG:-paper_t01}"
ITERATIONS="${ITERATIONS:-10000}"

VIEW_MODE="${VIEW_MODE:-paper}"
VIEW_SEED="${VIEW_SEED:-0}"
PAPER_PITCH_DEGREES="${PAPER_PITCH_DEGREES:-10}"
TRANSLATION_RADIUS="${TRANSLATION_RADIUS:-0.1}"

RUN_SMOKE="${RUN_SMOKE:-1}"
RUN_TRAIN="${RUN_TRAIN:-1}"
RUN_RENDER="${RUN_RENDER:-1}"
RUN_METRICS="${RUN_METRICS:-1}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"
AUTO_SKIP_UNAVAILABLE="${AUTO_SKIP_UNAVAILABLE:-1}"

DREAM_ENV="${DREAM_ENV:-/mnt/data/wangqq/conda_envs/dreamscene360}"
IQA_ENV="${IQA_ENV:-/mnt/data/wangqq/conda_envs/iqa_eval}"
QALIGN_ENV="${QALIGN_ENV:-/mnt/data/wangqq/conda_envs/q_align_eval}"
DA3_ENV="${DA3_ENV:-/mnt/data/wangqq/conda_envs/depth_anything3}"
DAP_ENV="${DAP_ENV:-/mnt/data/wangqq/conda_envs/dap}"

HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
HF_HOME="${HF_HOME:-/mnt/data/wangqq/hf_cache}"
TORCH_HOME="${TORCH_HOME:-/mnt/data/wangqq/torch_cache}"

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

QALIGN_MODEL="${QALIGN_MODEL:-q-future/one-align}"

source_conda() {
  if command -v conda >/dev/null 2>&1; then
    eval "$(conda shell.bash hook)"
    return
  fi

  local candidates=(
    "$HOME/miniconda3/etc/profile.d/conda.sh"
    "$HOME/anaconda3/etc/profile.d/conda.sh"
    "/mnt/data/wangqq/miniconda3/etc/profile.d/conda.sh"
    "/mnt/data/wangqq/anaconda3/etc/profile.d/conda.sh"
  )
  for candidate in "${candidates[@]}"; do
    if [[ -f "$candidate" ]]; then
      # shellcheck source=/dev/null
      source "$candidate"
      return
    fi
  done

  echo "[error] conda not found. Activate conda first or set up conda.sh." >&2
  exit 1
}

canonical_method() {
  local method="${1,,}"
  method="${method//-/_}"
  case "$method" in
    da3|depthanything3|depth_anything_3) echo "depth_anything3" ;;
    vggt|vggtomega) echo "vggt_omega" ;;
    *) echo "$method" ;;
  esac
}

safe_method() {
  canonical_method "$1" | sed -E 's/[^a-zA-Z0-9_]+/_/g'
}

model_dir_for() {
  local method_safe
  method_safe="$(safe_method "$1")"
  echo "output/${OUTPUT_PREFIX}_${method_safe}_${SCENE_NAME}"
}

render_dir_for() {
  local method_safe
  method_safe="$(safe_method "$1")"
  echo "table1_inputs/${method_safe}/${SCENE_NAME}_${VIEW_TAG}"
}

method_train_args() {
  local method
  method="$(canonical_method "$1")"
  TRAIN_EXTRA_ARGS=(--depth_predictor "$method")

  case "$method" in
    omnidata)
      ;;
    depth_anything3)
      TRAIN_EXTRA_ARGS+=(
        --depth_anything3_model "$DEPTH_ANYTHING3_MODEL"
        --depth_anything3_command "$DEPTH_ANYTHING3_COMMAND"
      )
      ;;
    dap)
      if [[ -z "$DAP_DEPTH_COMMAND" ]]; then
        echo "[error] dap requires DAP_DEPTH_COMMAND." >&2
        echo "Example: export DAP_DEPTH_COMMAND='python /path/to/dap/infer.py --image {input} --output {output}'" >&2
        exit 1
      fi
      [[ -n "$DAP_ROOT" ]] && TRAIN_EXTRA_ARGS+=(--dap_root "$DAP_ROOT")
      [[ -n "$DAP_MODEL_PATH" ]] && TRAIN_EXTRA_ARGS+=(--dap_model_path "$DAP_MODEL_PATH")
      TRAIN_EXTRA_ARGS+=(--dap_command "$DAP_DEPTH_COMMAND")
      ;;
    vggt_omega)
      TRAIN_EXTRA_ARGS+=(
        --vggt_root "$VGGT_ROOT"
        --vggt_model_path "$VGGT_MODEL_PATH"
        --vggt_chunk_size "$VGGT_CHUNK_SIZE"
      )
      ;;
    *)
      echo "[error] unknown method: $method" >&2
      exit 1
      ;;
  esac
}

smoke_args_for_methods() {
  SMOKE_ARGS=(
    --methods ${METHODS}
    --image "$SOURCE_PATH"
    --image-size 256
    --batch-size 1
    --output-dir "output/${OUTPUT_PREFIX}_smoke"
    --depth-anything3-model "$DEPTH_ANYTHING3_MODEL"
    --depth-anything3-command "$DEPTH_ANYTHING3_COMMAND"
    --vggt-root "$VGGT_ROOT"
    --vggt-model-path "$VGGT_MODEL_PATH"
    --vggt-chunk-size "$VGGT_CHUNK_SIZE"
  )
  [[ -n "$DAP_ROOT" ]] && SMOKE_ARGS+=(--dap-root "$DAP_ROOT")
  [[ -n "$DAP_MODEL_PATH" ]] && SMOKE_ARGS+=(--dap-model-path "$DAP_MODEL_PATH")
  [[ -n "$DAP_DEPTH_COMMAND" ]] && SMOKE_ARGS+=(--dap-command "$DAP_DEPTH_COMMAND")
}

preflight_configs() {
  local active_methods=()
  for raw_method in ${METHODS}; do
    method="$(canonical_method "$raw_method")"
    skip_reason=""
    case "$method" in
      depth_anything3)
        if [[ ! -x "${DA3_ENV}/bin/python" ]]; then
          skip_reason="Depth Anything 3 env python not found: ${DA3_ENV}/bin/python"
        fi
        ;;
      dap)
        if [[ -z "$DAP_DEPTH_COMMAND" ]]; then
          skip_reason="DAP_DEPTH_COMMAND is empty"
        elif [[ "$DAP_DEPTH_COMMAND" == *"/path/to/"* ]]; then
          skip_reason="DAP_DEPTH_COMMAND still contains placeholder /path/to/"
        else
          dap_script="$(echo "$DAP_DEPTH_COMMAND" | awk '{print $2}')"
          if [[ "$dap_script" == *.py && ! -f "$dap_script" ]]; then
            skip_reason="DAP script not found: $dap_script"
          fi
        fi
        ;;
      vggt_omega)
        if [[ ! -d "$VGGT_ROOT" ]]; then
          skip_reason="VGGT_ROOT not found: $VGGT_ROOT"
        fi
        ;;
    esac

    if [[ -n "$skip_reason" ]]; then
      if [[ "$AUTO_SKIP_UNAVAILABLE" == "1" ]]; then
        echo "[skip] ${method}: ${skip_reason}"
        continue
      fi
      echo "[error] ${method}: ${skip_reason}" >&2
      echo "Set AUTO_SKIP_UNAVAILABLE=1 to skip unavailable methods." >&2
      exit 1
    fi
    active_methods+=("$method")
  done

  if [[ "${#active_methods[@]}" == "0" ]]; then
    echo "[error] no runnable methods left after preflight checks." >&2
    exit 1
  fi

  METHODS="${active_methods[*]}"
  echo "[config] runnable methods: $METHODS"
}

export HF_ENDPOINT HF_HOME TORCH_HOME

source_conda
cd "$PROJECT_DIR"

echo "[config] project: $PROJECT_DIR"
echo "[config] scene: $SCENE_NAME"
echo "[config] source: $SOURCE_PATH"
echo "[config] methods: $METHODS"
echo "[config] view: ${VIEW_MODE}, seed=${VIEW_SEED}, pitch=${PAPER_PITCH_DEGREES}, translation=${TRANSLATION_RADIUS}"
preflight_configs

if [[ "$RUN_SMOKE" == "1" ]]; then
  echo "[stage] smoke test depth predictors"
  conda activate "$DREAM_ENV"
  smoke_args_for_methods
  PYTHONUNBUFFERED=1 python -u scripts/smoke_test_depth_predictors.py "${SMOKE_ARGS[@]}"
fi

if [[ "$RUN_TRAIN" == "1" ]]; then
  echo "[stage] train methods"
  conda activate "$DREAM_ENV"
  for raw_method in ${METHODS}; do
    method="$(canonical_method "$raw_method")"
    model_dir="$(model_dir_for "$method")"
    final_ply="${model_dir}/point_cloud/iteration_${ITERATIONS}/point_cloud.ply"

    if [[ "$SKIP_EXISTING" == "1" && -f "$final_ply" ]]; then
      echo "[train] skip existing ${method}: $final_ply"
      continue
    fi

    mkdir -p "$model_dir"
    method_train_args "$method"

    echo "[train] ${method} -> ${model_dir}"
    SECONDS=0
    PYTHONUNBUFFERED=1 python -u train.py \
      -s "$SOURCE_PATH" \
      -m "$model_dir" \
      --iterations "$ITERATIONS" \
      --test_iterations "$ITERATIONS" \
      --save_iterations "$ITERATIONS" \
      "${TRAIN_EXTRA_ARGS[@]}" \
      2>&1 | tee "${model_dir}/train.log"
    runtime="$SECONDS"
    echo "$runtime" > "${model_dir}/runtime_sec.txt"
    echo "[train] ${method} runtime_sec=${runtime}"
  done
fi

if [[ "$RUN_RENDER" == "1" ]]; then
  echo "[stage] render paper-style views"
  conda activate "$DREAM_ENV"
  for raw_method in ${METHODS}; do
    method="$(canonical_method "$raw_method")"
    model_dir="$(model_dir_for "$method")"
    render_dir="$(render_dir_for "$method")"

    echo "[render] ${method} -> ${render_dir}"
    python scripts/render_table1_views.py \
      -s "${model_dir}/generated_data" \
      -m "$model_dir" \
      --iteration -1 \
      --output-dir "$render_dir" \
      --view-mode "$VIEW_MODE" \
      --seed "$VIEW_SEED" \
      --paper-pitch-degrees "$PAPER_PITCH_DEGREES" \
      --translation-radius "$TRANSLATION_RADIUS" \
      --save-depth
  done
fi

manifest="table1_manifest_${OUTPUT_PREFIX}.csv"
conda activate "$DREAM_ENV"
python scripts/build_depth_ablation_manifest.py \
  --methods ${METHODS} \
  --scene "$SCENE_NAME" \
  --source-path "$SOURCE_PATH" \
  --output-prefix "$OUTPUT_PREFIX" \
  --view-tag "$VIEW_TAG" \
  --output "$manifest"

if [[ "$RUN_METRICS" == "1" ]]; then
  echo "[stage] evaluate CLIP"
  conda activate "$IQA_ENV"
  PYTHONUNBUFFERED=1 python -u scripts/evaluate_table1.py \
    --manifest "$manifest" \
    --output-dir "table1_eval_${OUTPUT_PREFIX}_clip_only" \
    --device cuda \
    --metrics clip

  echo "[stage] evaluate NIQE/BRISQUE"
  PYTHONUNBUFFERED=1 python -u scripts/evaluate_table1.py \
    --manifest "$manifest" \
    --output-dir "table1_eval_${OUTPUT_PREFIX}_iqa_skip" \
    --device cpu \
    --metrics niqe brisque \
    --skip-metric-errors

  echo "[stage] evaluate Q-Align"
  conda activate "$QALIGN_ENV"
  PYTHONUNBUFFERED=1 python -u scripts/evaluate_table1.py \
    --manifest "$manifest" \
    --output-dir "table1_eval_${OUTPUT_PREFIX}_qalign" \
    --device cuda \
    --metrics qalign \
    --qalign-model "$QALIGN_MODEL"

  echo "[stage] merge final table"
  conda activate "$IQA_ENV"
  python scripts/merge_table1_results.py \
    --clip "table1_eval_${OUTPUT_PREFIX}_clip_only/summary.csv" \
    --iqa "table1_eval_${OUTPUT_PREFIX}_iqa_skip/summary.csv" \
    --qalign "table1_eval_${OUTPUT_PREFIX}_qalign/summary.csv" \
    --output-dir "table1_eval_${OUTPUT_PREFIX}_full"
fi

echo "[done] manifest: $manifest"
echo "[done] final table: table1_eval_${OUTPUT_PREFIX}_full/table1.md"

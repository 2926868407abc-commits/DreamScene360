#!/usr/bin/env bash
set -euo pipefail

# Server-side navigation runner for DreamScene360.
#
# Edit MODEL_PATH / SOURCE_PATH / start-goal below, then run:
#   bash scripts/run_navigation_server.sh
#
# Notes:
# - G2VLM paths are absolute and exported for any code path that needs them.
# - Current navigation collision uses DreamScene360 Gaussian geometry by default.
# - HYBRID mode is wired as an interface; G2VLM depth fusion is reserved but not
#   yet used by inference_nav.py as an active collision source.

DREAMSCENE360_ROOT="/root/autodl-tmp/DreamScene360"
G2VLM_ROOT="/root/autodl-tmp/G2VLM"
G2VLM_MODEL_PATH="/root/autodl-tmp/G2VLM/checkpoints/G2VLM-2B-MoT"

# Change these two paths to your trained DreamScene360 scene.
MODEL_PATH="/root/autodl-tmp/DreamScene360/output/Italy_output"
SOURCE_PATH="/root/autodl-tmp/DreamScene360/data/Italy_text"

# Choose one start/goal mode.
# Mode A: viewpoint indices from nav_candidates/metadata.txt
START_ARGS=(--start_vp 3)
GOAL_ARGS=(--goal_vp 42)
VP_DIR="/root/autodl-tmp/DreamScene360/nav_candidates"

# Mode B example: raw xyz
# START_ARGS=(--start_xyz "1.5 g 0.5")
# GOAL_ARGS=(--goal_xyz "-1.0 g -1.5")

# Mode C example: image localization
# START_ARGS=(--start_img "/root/autodl-tmp/start.jpg")
# GOAL_ARGS=(--goal_img "/root/autodl-tmp/goal.jpg")

OUTPUT_DIR="/root/autodl-tmp/DreamScene360/nav_output"
OCC_CKPT="/root/autodl-tmp/DreamScene360/nav_output/occ_field.pt"

export G2VLM_ROOT
export G2VLM_MODEL_PATH
export PYTHONPATH="$DREAMSCENE360_ROOT:$G2VLM_ROOT:${PYTHONPATH:-}"

cd "$DREAMSCENE360_ROOT"

python inference_nav.py \
  -m "$MODEL_PATH" \
  -s "$SOURCE_PATH" \
  --geometry_source gaussian \
  --vp_dir "$VP_DIR" \
  "${START_ARGS[@]}" \
  "${GOAL_ARGS[@]}" \
  --occ_checkpoint "$OCC_CKPT" \
  --occ_fit_steps 2000 \
  --num_waypoints 20 \
  --opt_steps 800 \
  --samples_per_segment 4 \
  --random_inits 2 \
  --output_dir "$OUTPUT_DIR" \
  --fov 60 \
  --g2vlm_root "$G2VLM_ROOT" \
  --g2vlm_model_path "$G2VLM_MODEL_PATH"

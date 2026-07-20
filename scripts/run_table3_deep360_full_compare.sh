#!/usr/bin/env bash
set -euo pipefail

# Full local same-sample evaluation for Deep360 testing split:
#   direct panorama depth
# vs
#   perspective monocular depth -> DreamScene360/PanoGeo panorama fusion.

PROJECT_DIR="${PROJECT_DIR:-/mnt/data/wangqq/DreamScene360}"
DREAM_ENV="${DREAM_ENV:-/mnt/data/wangqq/conda_envs/dreamscene360}"

DATASETS="${DATASETS:-Deep360}"
MAX_PER_DATASET="${MAX_PER_DATASET:-0}"
MANIFEST="${MANIFEST:-panorama_depth_manifest_deep360_full.csv}"
AUDIT_DIR="${AUDIT_DIR:-panorama_depth_manifest_deep360_full_audit}"

DIRECT_METHOD="${DIRECT_METHOD:-depth_anything3}"
DIRECT_LABEL="${DIRECT_LABEL:-DA3-Direct-Full-Deep360}"
DIRECT_OUTPUT_DIR="${DIRECT_OUTPUT_DIR:-panorama_depth_eval_da3_direct_deep360_full}"

FUSION_INNER_DEPTH_PREDICTOR="${FUSION_INNER_DEPTH_PREDICTOR:-depth_anything3}"
FUSION_LABEL="${FUSION_LABEL:-DreamScene360-PanoGeo-DA3-P240-Full-Deep360}"
FUSION_DIR="${FUSION_DIR:-panorama_depth_eval_dreamscene360_pano_geo_da3_p240_deep360_full}"
COMPARE_OUTPUT_DIR="${COMPARE_OUTPUT_DIR:-panorama_depth_compare_da3_p240_fusion_vs_da3_direct_deep360_full}"

PANO_GEO_NUM_PERSPECTIVES="${PANO_GEO_NUM_PERSPECTIVES:-240}"
PANO_GEO_ITERS="${PANO_GEO_ITERS:-1500}"
PANO_GEO_DEPTH_NORMALIZE="${PANO_GEO_DEPTH_NORMALIZE:-none}"
DEPTH_ANYTHING3_BATCH_SIZE="${DEPTH_ANYTHING3_BATCH_SIZE:-4}"
SAVE_PREDICTIONS="${SAVE_PREDICTIONS:-0}"
RUN_EVAL="${RUN_EVAL:-1}"
RUN_FUSION="${RUN_FUSION:-1}"
RUN_DIRECT="${RUN_DIRECT:-1}"
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

echo "[stage] build Deep360 full manifest: $MANIFEST"
"$PYTHON_BIN" scripts/make_panorama_depth_manifest.py \
  --output "$MANIFEST" \
  --datasets "$DATASETS" \
  --max-per-dataset "$MAX_PER_DATASET" \
  --verbose

echo "[stage] audit Deep360 full manifest"
"$PYTHON_BIN" scripts/audit_panorama_depth_manifest.py \
  --manifest "$MANIFEST" \
  --output-dir "$AUDIT_DIR" \
  --datasets "$DATASETS" \
  --expected-aspect 0.5 \
  --fail-on-invalid

echo "[info] raw Deep360 files are stored as 1024x512 HxW; evaluator transposes them to 512x1024."
echo "[info] full audit table: $AUDIT_DIR/audit.md"

echo "[stage] count manifest rows"
"$PYTHON_BIN" - "$MANIFEST" <<'PY'
import csv
import sys
from collections import Counter

manifest = sys.argv[1]
rows = list(csv.DictReader(open(manifest, newline="", encoding="utf-8-sig")))
counts = Counter(row["dataset"] for row in rows)
print("| Dataset | Num Images |")
print("|---|---:|")
for dataset in sorted(counts):
    print(f"| {dataset} | {counts[dataset]} |")
print(f"| TOTAL | {len(rows)} |")
PY

if [[ "$RUN_EVAL" != "1" ]]; then
  echo "[done] RUN_EVAL=$RUN_EVAL, stopped after manifest/audit/count."
  echo "[done] manifest: $MANIFEST"
  echo "[done] audit: $AUDIT_DIR/audit.md"
  exit 0
fi

echo "[stage] run Deep360 direct-vs-fusion comparison"
DATASETS="$DATASETS" \
MAX_PER_DATASET="$MAX_PER_DATASET" \
MANIFEST="$MANIFEST" \
RUN_FUSION="$RUN_FUSION" \
RUN_DIRECT="$RUN_DIRECT" \
RUN_COMPARE="$RUN_COMPARE" \
SAVE_PREDICTIONS="$SAVE_PREDICTIONS" \
DIRECT_METHOD="$DIRECT_METHOD" \
DIRECT_LABEL="$DIRECT_LABEL" \
DIRECT_OUTPUT_DIR="$DIRECT_OUTPUT_DIR" \
FUSION_INNER_DEPTH_PREDICTOR="$FUSION_INNER_DEPTH_PREDICTOR" \
FUSION_LABEL="$FUSION_LABEL" \
FUSION_DIR="$FUSION_DIR" \
COMPARE_OUTPUT_DIR="$COMPARE_OUTPUT_DIR" \
PANO_GEO_NUM_PERSPECTIVES="$PANO_GEO_NUM_PERSPECTIVES" \
PANO_GEO_ITERS="$PANO_GEO_ITERS" \
PANO_GEO_DEPTH_NORMALIZE="$PANO_GEO_DEPTH_NORMALIZE" \
DEPTH_ANYTHING3_BATCH_SIZE="$DEPTH_ANYTHING3_BATCH_SIZE" \
bash scripts/run_local_direct_vs_fusion_depth_compare.sh

echo "[done] manifest: $MANIFEST"
echo "[done] audit: $AUDIT_DIR/audit.md"
echo "[done] fusion table: $FUSION_DIR/table3_style.md"
echo "[done] direct table: $DIRECT_OUTPUT_DIR/table3_style.md"
echo "[done] comparison: $COMPARE_OUTPUT_DIR/comparison_summary.md"

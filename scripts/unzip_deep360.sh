#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/mnt/data/wangqq/DreamScene360}"
DOWNLOAD_DIR="${DOWNLOAD_DIR:-$PROJECT_DIR/downloads/Deep360}"
OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_DIR/datasets/Deep360}"

cd "$PROJECT_DIR"
mkdir -p "$OUTPUT_DIR"

echo "[start] Deep360 unzip"
echo "[download_dir] $DOWNLOAD_DIR"
echo "[output_dir] $OUTPUT_DIR"

shopt -s nullglob
zips=("$DOWNLOAD_DIR"/ep*_500frames.zip)
if [[ "${#zips[@]}" -eq 0 ]]; then
  echo "[error] no Deep360 zip files found in $DOWNLOAD_DIR" >&2
  exit 1
fi

for zip_path in "${zips[@]}"; do
  echo "[unzip] $zip_path"
  unzip -n "$zip_path" -d "$OUTPUT_DIR"
done

if [[ -f "$DOWNLOAD_DIR/README.txt" ]]; then
  cp "$DOWNLOAD_DIR/README.txt" "$OUTPUT_DIR/"
fi

echo "[done] Deep360 unzip"

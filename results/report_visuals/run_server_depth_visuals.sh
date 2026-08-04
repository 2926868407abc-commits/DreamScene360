#!/usr/bin/env bash
set -euo pipefail

cd /mnt/data/wangqq/DreamScene360

mkdir -p results/report_visuals

PYTHON_BIN="${PYTHON_BIN:-/mnt/data/wangqq/conda_envs/dreamscene360/bin/python}"

"${PYTHON_BIN}" scripts/visualize_report_depth_results.py metric-bars \
  --summary-csv results/panorama_depth_three_dataset_summary.csv \
  --output results/report_visuals/metric_bars_three_datasets.png

"${PYTHON_BIN}" scripts/visualize_report_depth_results.py depth-comparison \
  --manifest panorama_depth_manifest_mp_s2d3d_balanced143.csv \
  --direct-metrics panorama_depth_eval_da3_direct_mp_s2d3d_fixedscale_calib10/metrics_per_image.csv \
  --ours-metrics panorama_depth_eval_blend_direct_p60_i300_mp_s2d3d_alpha04/metrics_per_image.csv \
  --dataset Matterport3D \
  --output results/report_visuals/matterport3d_depth_comparison.png

"${PYTHON_BIN}" scripts/visualize_report_depth_results.py depth-comparison \
  --manifest panorama_depth_manifest_mp_s2d3d_balanced143.csv \
  --direct-metrics panorama_depth_eval_da3_direct_mp_s2d3d_fixedscale_calib10/metrics_per_image.csv \
  --ours-metrics panorama_depth_eval_blend_direct_p60_i300_mp_s2d3d_alpha04/metrics_per_image.csv \
  --dataset Stanford2D3D \
  --output results/report_visuals/stanford2d3d_depth_comparison.png

"${PYTHON_BIN}" scripts/visualize_report_depth_results.py depth-comparison \
  --manifest panorama_depth_manifest_deep360_balanced143.csv \
  --direct-metrics panorama_depth_eval_da3_direct_deep360_fixedscale_calib10/metrics_per_image.csv \
  --ours-metrics panorama_depth_eval_dreamscene360_pano_geo_da3_p24_i100_deep360_fixedscale_calib10/metrics_per_image.csv \
  --dataset Deep360 \
  --output results/report_visuals/deep360_depth_comparison.png

ls -lh results/report_visuals/*.png

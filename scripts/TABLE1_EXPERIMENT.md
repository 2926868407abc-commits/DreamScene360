# Table 1 Quantitative Comparison

This experiment reproduces the table format:

`CLIP Distance↓ / Q-Align↑ / NIQE↓ / BRISQUE↓ / Runtime`

## Inputs

Prepare one manifest CSV row per method and scene. Each row can point to either one image with `image_path` or a directory with `image_dir` plus `image_glob`.

Required columns:

- `method`: for example `Ours` or `LucidDreamer`
- `scene`: scene id used for grouping/debugging
- `prompt`: text prompt for CLIP Distance
- `image_path` or `image_dir`
- `runtime_sec`: generation runtime for that method/scene

Example:

```csv
method,scene,prompt,image_path,image_dir,image_glob,runtime_sec
Ours,alley,"a photorealistic narrow alley with warm lights",,../output/ours/alley/renders,*.png,440
LucidDreamer,alley,"a photorealistic narrow alley with warm lights",,../output/luciddreamer/alley/renders,*.png,375
```

## Run

From `DreamScene360/`:

Render the six evaluation views for our trained model:

```powershell
python scripts/render_table1_views.py `
  -s data/YOUR_SCENE `
  -m output/OUTPUT_NAME `
  --iteration -1 `
  --output-dir table1_inputs/ours/YOUR_SCENE `
  --save-depth
```

Put the corresponding LucidDreamer rendered views under a comparable folder,
for example `table1_inputs/luciddreamer/YOUR_SCENE/renders`.

Then run the metrics:

```powershell
python scripts/evaluate_table1.py `
  --manifest scripts/table1_manifest.example.csv `
  --output-dir table1_eval
```

The script writes:

- `table1_eval/metrics_per_image.csv`
- `table1_eval/summary.csv`
- `table1_eval/table1.md`
- `table1_eval/table1.tex`
- `table1_eval/metric_errors.json`

## Dependencies

The script loads metric dependencies lazily.

- CLIP Distance: `transformers` or `open_clip_torch`
- NIQE / BRISQUE: `pyiqa`
- Q-Align: Q-Align package, or the vendored copy under `../VideoScore2/eval/eval_methods/utils_q_align`

Use `--fail-on-missing-metric` if you want the script to stop instead of writing partial results.

## Notes

- `render_table1_views.py` uses `model_path/generated_data` automatically when it exists, so rendering does not rerun panorama depth initialization.
- Q-Align in the vendored implementation returns a 0-1 weighted score. The evaluator multiplies it by `--qalign-scale 5.0` by default to match the 0-5 scale used in the paper table.

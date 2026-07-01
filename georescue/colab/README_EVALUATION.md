# Evaluation README

This guide explains how to run `evaluate.py` to annotate flooded areas in satellite images using the base Qwen2-VL model plus the LoRA adapter in `adapter/`.

## What It Does

The evaluation script:

- Loads `Qwen/Qwen2-VL-7B-Instruct`
- Applies the LoRA adapter from `adapter/`
- Runs flood-area annotation on images in `satellite_images_data/`
- Saves annotated output images
- Saves prediction JSON files and predicted masks
- Generates summary reports and metrics

## Folder Structure

Input folders:

- `satellite_images_data/`
- `adapter/`

Generated output folder:

- `output/`

Inside `output/`, the script creates:

- `annotated_images/`  
  Annotated image previews with predicted flood overlays
- `predictions/`  
  Per-image raw/model prediction JSON files
- `predicted_masks/`  
  Predicted flood masks, and GT masks if available
- `reports/`  
  Evaluation summary files

## Requirements

Install the ML dependencies before running evaluation:

```bash
pip install -r requirements.txt
```

Important packages include:

- `torch`
- `transformers`
- `peft`
- `bitsandbytes`
- `qwen-vl-utils`
- `pillow`
- `numpy`
- `pandas`

## Run Evaluation

Basic run:

```bash
python evaluate.py --load-in-4bit
```

Run only a few images for a quick test:

```bash
python evaluate.py --load-in-4bit --max-images 5
```

Reuse existing prediction files when re-running report generation:

```bash
python evaluate.py --load-in-4bit --skip-existing
```

Example with custom paths:

```bash
python evaluate.py \
  --load-in-4bit \
  --images-dir satellite_images_data \
  --metadata satellite_images_data/metadata.csv \
  --adapter-dir adapter \
  --output-dir output
```

## Command Line Options

Available arguments:

- `--images-dir`  
  Directory containing input images
- `--metadata`  
  Metadata CSV containing image filename and geographic bounds
- `--adapter-dir`  
  Path to the LoRA adapter folder
- `--output-dir`  
  Output directory for images, masks, predictions, and reports
- `--model-name`  
  Base model name, default: `Qwen/Qwen2-VL-7B-Instruct`
- `--max-images`  
  Limit how many images are processed
- `--max-new-tokens`  
  Max generated output tokens
- `--device-map`  
  Device placement option for Transformers
- `--dtype`  
  Model dtype: `auto`, `float16`, `bfloat16`, or `float32`
- `--skip-existing`  
  Skip images with existing prediction outputs
- `--load-in-4bit`  
  Load the base model in 4-bit mode to reduce GPU memory usage in Colab

## Input Format

The script expects:

- image tiles in `satellite_images_data/`
- a metadata file at `satellite_images_data/metadata.csv`

Current metadata columns used:

- `idx`
- `lon_min`
- `lat_min`
- `lon_max`
- `lat_max`
- `file`

These geographic bounds are used to convert predicted polygons into pixel masks and visualization overlays.

## Ground Truth Support

The script supports automatic metric computation when ground truth exists.

Supported GT discovery methods:

1. Via metadata columns such as:
- `mask`
- `mask_path`
- `label`
- `label_path`
- `geojson`
- `geojson_path`
- `flood_polygon_path`
- `ground_truth`
- `gt_path`

2. Via sidecar files beside each image, for example:
- `<image_stem>.geojson`
- `<image_stem>.json`
- `<image_stem>_flood.geojson`
- `<image_stem>_flood.json`
- `<image_stem>_mask.png`
- `<image_stem>_mask.jpg`
- `<image_stem>_mask.jpeg`

If GT is found, the script rasterizes polygons or reads mask images and computes label-based metrics automatically.

## Metrics Generated

The reports include:

- Accuracy
- Precision
- Recall
- Specificity
- F1 Score
- IoU
- Dice
- Inference time
- Average inference time
- Median inference time
- Min inference time
- Max inference time
- Throughput (images/sec)
- Parse success rate
- Predicted flood coverage ratio
- Report generation time

## Current Dataset Note

At the time this evaluator was added, the folder `satellite_images_data/` contained:

- image tiles
- `metadata.csv`

It did not contain matching flood ground-truth masks or polygons.

That means:

- prediction outputs and annotated images work now
- timing metrics work now
- label-based metrics such as Accuracy, F1 Score, and IoU will show `N/A` until GT files are added

## Output Files

Main files generated in `output/reports/`:

- `metrics_summary.json`  
  Overall dataset-level metrics
- `per_image_metrics.csv`  
  One row per image with timings and per-image metrics
- `report.md`  
  Markdown summary report
- `report.html`  
  HTML summary report

Per-image artifacts:

- `output/predictions/<image_stem>.json`
- `output/predicted_masks/<image_stem>_pred_mask.png`
- `output/predicted_masks/<image_stem>_gt_mask.png` if GT exists
- `output/annotated_images/<image_stem>_annotated.png`

## Example Workflow

1. Put satellite images in `satellite_images_data/`
2. Keep the adapter files in `adapter/`
3. Run:

```bash
python evaluate.py --load-in-4bit
```

4. Open:

- `output/annotated_images/`
- `output/reports/report.html`

## Troubleshooting

If the model fails to load:

- verify internet/model access for `Qwen/Qwen2-VL-7B-Instruct`
- verify the adapter weights exist in `adapter/`
- verify required Python packages are installed

If metrics show `N/A`:

- add ground-truth polygon or mask files
- confirm GT paths are discoverable through metadata or sidecar naming

If inference is slow:

- run on GPU
- prefer `--load-in-4bit` in Colab
- lower `--max-images` for quick tests
- lower `--max-new-tokens`

## Verified Script

The evaluator was syntax-checked with:

```bash
python -m py_compile evaluate.py
```

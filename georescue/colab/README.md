# Colab Evaluation Pack

This folder is self-contained for Google Colab upload.

## Contents

- `evaluate.py`  
  Runs flood annotation evaluation with Qwen2-VL + LoRA adapter
- `requirements.txt`  
  Python dependencies for Colab setup
- `adapter/`  
  LoRA adapter files
- `satellite_images_data/`  
  Input satellite images and `metadata.csv`
- `README_EVALUATION.md`  
  Full evaluation guide

## Quick Start

Upload this `colab` folder to Google Colab, then run:

```bash
pip install -r requirements.txt
python evaluate.py --load-in-4bit
```

Quick test:

```bash
python evaluate.py --load-in-4bit --max-images 5
```

Outputs will be written to:

- `output/annotated_images/`
- `output/predictions/`
- `output/predicted_masks/`
- `output/reports/`

See `README_EVALUATION.md` for full details.

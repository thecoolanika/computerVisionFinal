#Optical Illusion Final Project

This repo analyzes ambiguous optical illusions, generates size illusions, and analyzes them as well.

## What you need

- PyTorch, torchvision, transformers, pillow, matplotlib, pandas

Install dependencies:

```bash
pip install torch torchvision transformers pillow matplotlib pandas
```

## Folder layout

### Ambiguous Images


### Size Illusions
- `data/generate_illusions.py` — writes PNGs and `annotations.jsonl`
- `datasets/synthetic_loader.py` — reads JSONL, gives tensors and label ids
- `models/cnn_model.py` — ResNet50, EfficientNet-B0
- `models/vit_model.py` — ViT-B/16 
- `models/clip_eval.py` — CLIP and its prompts
- `notebooks/evaluate_models.ipynb` — end-to-end run from data to tables and sample plots 

## How to run for size illusions

Open `notebooks/evaluate_models.ipynb` and run all cells.

The data cell calls `generate_dataset` for train and test. Training images omit contextual fins or rails; test images include them. That split is intentional. Model cells download ImageNet weights on first use. Feature extration is the majority of the runtime with short probe training and short CLIP running for better speed on both CPU and GPU. 

The results table reports train accuracy without illusions and test accuracy with illusions. 

Labels for size illusions:

Horizontal illusions use `left_bigger`, `right_bigger`, `same_size`. Ponzo uses `top_bigger`, `bottom_bigger`, `same_size`. The loader maps old `left` / `right` / `equal` strings when `meta.illusion_type` is present.


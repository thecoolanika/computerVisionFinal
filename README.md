# Optical Illusion Final Project

This repo analyzes ambiguous optical illusions, generates size illusions, and analyzes them as well.

## What you need

- PyTorch, torchvision, transformers, pillow, matplotlib, pandas, kagglehub

Install dependencies:

```bash
pip install torch torchvision transformers pillow matplotlib pandas kagglehub
```

## Folder layout

### Ambiguous images (Ambivision)

- `notebooks/ambivision_demo.ipynb` — one Ambivision image plus its label file; the demo we showed in class
- `notebooks/ambiguous_images_test.ipynb` — Ambivision dataset run across many image and txt pairs

### Size illusions

- `data/generate_illusions.py` — writes PNGs and `annotations.jsonl`
- `datasets/synthetic_loader.py` — reads JSONL, gives tensors and label ids
- `models/cnn_model.py` — ResNet50, EfficientNet-B0
- `models/vit_model.py` — ViT-B/16
- `models/clip_eval.py` — CLIP and its prompts
- `notebooks/evaluate_models.ipynb` — end-to-end run from data to tables and sample plots

## How to run for Ambivision optical illusions

Open `notebooks/ambivision_demo.ipynb` and run all cells for the demo we showed in the presentation.

The notebook downloads the `anonymac12i3/ambivision` bundle through `kagglehub` on first use, then looks under `Neurips_collection` / `No_Direction` for the sample pair `badgerturtle` (image plus `badgerturtle.txt`). The txt file lists two animals in `boundingbox1` and `boundingbox2`; those strings are the labels for our model with `boundingbox1` being the bigger animal.

For the whole dataset, use `notebooks/ambiguous_images_test.ipynb` the same way.

## How to run for size illusions

Open `notebooks/evaluate_models.ipynb` and run all cells.

The data cell calls `generate_dataset` for train and test. Training images are two objects, different sizes without illusions; test images include the illusions. The results table reports train accuracy without illusions and test accuracy with illusions.

Labels for size illusions:

Horizontal illusions use `left_bigger`, `right_bigger`, `same_size`. Ponzo uses `top_bigger`, `bottom_bigger`, `same_size`. 
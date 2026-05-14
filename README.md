# Size illusion evaluation

This repo builds synthetic Ebbinghaus, Müller–Lyer, and Ponzo images, then scores them with frozen backbones plus linear probes and with CLIP zero-shot prompts.

## What you need

- Python 3.10+ recommended
- PyTorch, torchvision, transformers, pillow, matplotlib, pandas

Install dependencies (adjust if you use conda):

```bash
pip install torch torchvision transformers pillow matplotlib pandas
```

## Folder layout

- `data/generate_illusions.py` — writes PNGs and `annotations.jsonl`
- `datasets/synthetic_loader.py` — reads JSONL, yields tensors and label ids
- `models/cnn_model.py` — ResNet50, EfficientNet-B0, probes
- `models/vit_model.py` — ViT-B/16 backbone
- `models/clip_eval.py` — CLIP prompts for synthetic labels
- `notebooks/evaluate_models.ipynb` — end-to-end run from data to tables and sample plots

## How to run a full check

1. Open the repo root in a terminal. If you run the notebook from `notebooks/`, the first code cell adds the parent directory to `sys.path` so imports resolve.

2. Open `notebooks/evaluate_models.ipynb` and run cells top to bottom.

3. The notebook picks a data directory (environment variable `CS153_DATA_ROOT` if set, else a preferred path under `mnt/cs/cs153/data/...`, else `../data` from the notebook). Synthetic files land in `<data_root>/synthetic_illusions/`.

4. The data cell calls `generate_dataset` for train and test. Training images omit contextual fins or rails; test images include them. That split is intentional.

5. Model cells download ImageNet weights on first use (network required once).

6. Feature extraction dominates runtime on CPU (often tens of minutes for the default counts). Probe training is short. CLIP runs on a capped subset for speed.

7. The results table reports train accuracy without test context and test accuracy with context, plus a simple mismatch rate labeled as bias in the notebook.

8. The last cell shows a few correct and incorrect ResNet predictions on the test split.

## Labels

Horizontal illusions use `left_bigger`, `right_bigger`, `same_size`. Ponzo uses `top_bigger`, `bottom_bigger`, `same_size`. The loader maps old `left` / `right` / `equal` strings when `meta.illusion_type` is present.

## Command-line data only

From the repo root:

```bash
python -m data.generate_illusions --output_dir ./synthetic_data --train_n 3000 --test_n 1200 --seed 42
```

## Other notebooks

`ambivision_demo.ipynb` and `illusion_model_compare.ipynb` are separate demos and are not required for the synthetic pipeline above.

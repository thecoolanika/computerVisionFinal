"""Render PNG size illusions and write JSONL annotations.

Ebbinghaus and Müller–Lyer use horizontal left versus right targets. Ponzo uses
stacked horizontal bars; the lower bar follows ``left_size`` and the upper
``right_size``. Ground-truth strings are axis-aware (see ``_ground_truth_label``).
"""

import argparse
import json
import math
import random
from pathlib import Path

from PIL import Image, ImageDraw

_INTERNAL = ["left", "right", "equal"]


def _ground_truth_label(illusion_type: str, internal: str) -> str:
    """Map sampled internal labels to stored ``ground_truth`` strings."""
    if illusion_type == "ponzo":
        if internal == "left":
            return "bottom_bigger"
        if internal == "right":
            return "top_bigger"
        return "same_size"
    if internal == "left":
        return "left_bigger"
    if internal == "right":
        return "right_bigger"
    return "same_size"


def _task_for(illusion_type: str) -> str:
    return "vertical_bar_size" if illusion_type == "ponzo" else "horizontal_bar_size"


def _draw_ebbinghaus(draw, left_size, right_size, with_context, rng, w=384, h=256):
    cx_l, cx_r, cy = w // 4, 3 * w // 4, h // 2
    if with_context:
        n = 8
        for side, cx, r in [("left", cx_l, left_size), ("right", cx_r, right_size)]:
            if side == "left":
                inducer_r, dist = int(r * 0.35), int(r * 2.6)
            else:
                inducer_r, dist = int(r * 0.8), int(r * 2.2)
            for k in range(n):
                t = 2 * math.pi * k / n
                ix = int(cx + dist * math.cos(t))
                iy = int(cy + dist * math.sin(t))
                draw.ellipse([ix - inducer_r, iy - inducer_r, ix + inducer_r, iy + inducer_r], outline=(90, 90, 90), width=2)
    draw.ellipse([cx_l - left_size, cy - left_size, cx_l + left_size, cy + left_size], outline=(0, 0, 0), width=3)
    draw.ellipse([cx_r - right_size, cy - right_size, cx_r + right_size, cy + right_size], outline=(0, 0, 0), width=3)
    return {"left_center": [cx_l, cy], "right_center": [cx_r, cy]}


def _draw_muller_lyer(draw, left_size, right_size, with_context, rng, w=384, h=256):
    cx_l, cx_r, cy = w // 4, 3 * w // 4, h // 2
    draw.line([cx_l - left_size, cy, cx_l + left_size, cy], fill=(0, 0, 0), width=4)
    draw.line([cx_r - right_size, cy, cx_r + right_size, cy], fill=(0, 0, 0), width=4)
    if with_context:
        wing = 16
        ang = math.radians(30)
        for cx, sgn in [(cx_l, -1), (cx_r, 1)]:
            size = left_size if cx == cx_l else right_size
            for x in [cx - size, cx + size]:
                dx = wing * math.cos(ang)
                dy = wing * math.sin(ang)
                draw.line([x, cy, x + sgn * dx, cy - dy], fill=(60, 60, 60), width=3)
                draw.line([x, cy, x + sgn * dx, cy + dy], fill=(60, 60, 60), width=3)
    return {"left_center": [cx_l, cy], "right_center": [cx_r, cy]}


def _draw_ponzo(draw, left_size, right_size, with_context, rng, w=384, h=256):
    x_mid = w // 2
    if with_context:
        draw.line([x_mid - 120, h - 20, x_mid - 20, 20], fill=(80, 80, 80), width=3)
        draw.line([x_mid + 120, h - 20, x_mid + 20, 20], fill=(80, 80, 80), width=3)
    y_low, y_high = int(h * 0.72), int(h * 0.34)
    draw.rectangle([x_mid - left_size, y_low - 6, x_mid + left_size, y_low + 6], outline=(0, 0, 0), width=3)
    draw.rectangle([x_mid - right_size, y_high - 6, x_mid + right_size, y_high + 6], outline=(0, 0, 0), width=3)
    return {"left_center": [x_mid, y_low], "right_center": [x_mid, y_high]}


def _sample_sizes(rng):
    base = rng.uniform(22, 42)
    diff = rng.uniform(6, 16)
    label = rng.choice(_INTERNAL)
    if label == "left":
        return base + diff, base, label
    if label == "right":
        return base, base + diff, label
    return base, base, label


def generate_dataset(output_dir, split, n_samples, with_context, seed=0):
    """Write ``split/images`` PNGs and ``split/annotations.jsonl`` with one JSON object per line."""
    rng = random.Random(seed)
    out = Path(output_dir)
    img_dir = out / split / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    ann_path = out / split / "annotations.jsonl"
    illusion_types = ["ebbinghaus", "muller_lyer", "ponzo"]
    with ann_path.open("w", encoding="utf-8") as f:
        for i in range(n_samples):
            illusion_type = illusion_types[i % len(illusion_types)]
            left_size, right_size, internal_gt = _sample_sizes(rng)
            image = Image.new("RGB", (384, 256), (245, 245, 245))
            draw = ImageDraw.Draw(image)
            if illusion_type == "ebbinghaus":
                extra = _draw_ebbinghaus(draw, int(left_size), int(right_size), with_context, rng)
            elif illusion_type == "muller_lyer":
                extra = _draw_muller_lyer(draw, int(left_size), int(right_size), with_context, rng)
            else:
                extra = _draw_ponzo(draw, int(left_size * 1.6), int(right_size * 1.6), with_context, rng)
            img_name = f"{illusion_type}_{i:05d}.png"
            image.save(img_dir / img_name)
            gt = _ground_truth_label(illusion_type, internal_gt)
            row = {
                "image_path": str((Path(split) / "images" / img_name).as_posix()),
                "task": _task_for(illusion_type),
                "ground_truth": gt,
                "meta": {
                    "left_size": float(left_size),
                    "right_size": float(right_size),
                    "illusion_type": illusion_type,
                    "internal_label": internal_gt,
                    "params": {"with_context": with_context, **extra},
                },
            }
            f.write(json.dumps(row) + "\n")
    return ann_path


def main():
    """CLI entry: train split without context bars, test split with context."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default="synthetic_data")
    parser.add_argument("--train_n", type=int, default=3000)
    parser.add_argument("--test_n", type=int, default=1200)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    generate_dataset(args.output_dir, "train", args.train_n, with_context=False, seed=args.seed)
    generate_dataset(args.output_dir, "test", args.test_n, with_context=True, seed=args.seed + 1)
    print("done", args.output_dir)


if __name__ == "__main__":
    main()

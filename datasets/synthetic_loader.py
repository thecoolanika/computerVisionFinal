import json
from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset


LABEL_TO_ID = {
    "left_bigger": 0,
    "right_bigger": 1,
    "top_bigger": 2,
    "bottom_bigger": 3,
    "same_size": 4,
}
ID_TO_LABEL = {v: k for k, v in LABEL_TO_ID.items()}
NUM_CLASSES = len(LABEL_TO_ID)

# Each illusion only has three physically possible answers; 5-way softmax dilutes scores.
CANDIDATE_LABELS_HORIZONTAL = ("left_bigger", "right_bigger", "same_size")
CANDIDATE_LABELS_VERTICAL = ("top_bigger", "bottom_bigger", "same_size")


def candidate_labels_for_row(row: dict) -> list[str]:
    itype = (row.get("meta") or {}).get("illusion_type", "")
    if itype == "ponzo":
        return list(CANDIDATE_LABELS_VERTICAL)
    return list(CANDIDATE_LABELS_HORIZONTAL)


def valid_class_indices_for_row(row: dict) -> list[int]:
    return [LABEL_TO_ID[l] for l in candidate_labels_for_row(row)]


def _normalize_ground_truth(row: dict) -> str:
    """Accept new axis-aware labels; upgrade old left/right/equal using illusion_type."""
    raw = row["ground_truth"]
    if raw in LABEL_TO_ID:
        return raw
    itype = (row.get("meta") or {}).get("illusion_type")
    if raw == "equal":
        return "same_size"
    if raw == "left":
        if itype == "ponzo":
            return "bottom_bigger"
        if itype in ("ebbinghaus", "muller_lyer"):
            return "left_bigger"
    if raw == "right":
        if itype == "ponzo":
            return "top_bigger"
        if itype in ("ebbinghaus", "muller_lyer"):
            return "right_bigger"
    raise KeyError(f"Unknown ground_truth {raw!r} for illusion_type={itype!r}")


class SyntheticIllusionDataset(Dataset):
    def __init__(self, root_dir, split, transform=None):
        self.root = Path(root_dir)
        self.split = split
        self.transform = transform
        ann = self.root / split / "annotations.jsonl"
        self.rows = []
        for line in ann.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            row["ground_truth"] = _normalize_ground_truth(row)
            self.rows.append(row)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]
        img_path = self.root / row["image_path"]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        y = LABEL_TO_ID[row["ground_truth"]]
        return image, y, row

import json
from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset


LABEL_TO_ID = {"left": 0, "right": 1, "equal": 2}
ID_TO_LABEL = {0: "left", 1: "right", 2: "equal"}


class SyntheticIllusionDataset(Dataset):
    def __init__(self, root_dir, split, transform=None):
        self.root = Path(root_dir)
        self.split = split
        self.transform = transform
        ann = self.root / split / "annotations.jsonl"
        self.rows = [json.loads(line) for line in ann.read_text(encoding="utf-8").splitlines() if line.strip()]

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

import importlib
import pathlib
import sys


SIZE_PATTERNS = [
    "which is larger",
    "which is bigger",
    "which object is larger",
    "which object is bigger",
    "which circle is larger",
    "which circle is bigger",
]


def _is_size_question(text):
    t = (text or "").lower()
    return any(p in t for p in SIZE_PATTERNS)


def load_illusionvqa_size(split="test"):
    repo_root = str(pathlib.Path(__file__).resolve().parents[1])
    removed = False
    if repo_root in sys.path:
        sys.path.remove(repo_root)
        removed = True
    try:
        load_dataset = importlib.import_module("datasets").load_dataset
    finally:
        if removed:
            sys.path.insert(0, repo_root)
    ds = load_dataset("IllusionVQA/IllusionVQA", split=split)
    ds = ds.filter(lambda x: _is_size_question(x.get("question", "")))
    return ds


def to_eval_examples(ds):
    out = []
    for ex in ds:
        out.append(
            {
                "image": ex.get("image"),
                "prompt": ex.get("question", "Which object is larger, left or right?"),
                "answer": ex.get("answer", ex.get("label")),
                "raw": ex,
            }
        )
    return out

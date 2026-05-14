import importlib.util
import importlib.machinery
import site
import sys
from pathlib import Path


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
    load_dataset = _resolve_hf_load_dataset()
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


def _resolve_hf_load_dataset():
    candidates = []
    try:
        candidates.extend(site.getsitepackages())
    except Exception:
        pass
    try:
        user_site = site.getusersitepackages()
        if isinstance(user_site, str):
            candidates.append(user_site)
    except Exception:
        pass
    for base in candidates:
        spec = importlib.machinery.PathFinder.find_spec("datasets", [base])
        if spec is None or spec.loader is None:
            continue
        mod = importlib.util.module_from_spec(spec)
        backup = sys.modules.get("datasets")
        sys.modules["datasets"] = mod
        spec.loader.exec_module(mod)
        if hasattr(mod, "load_dataset"):
            return mod.load_dataset
        if backup is not None:
            sys.modules["datasets"] = backup
    raise ImportError("Could not resolve Hugging Face datasets package from site-packages")

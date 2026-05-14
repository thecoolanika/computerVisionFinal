import torch
import torch.nn.functional as F
from transformers import CLIPModel, CLIPProcessor


IVQA_LABELS = ["left", "right", "equal"]
IVQA_PROMPTS = {
    "left": "the left object is larger",
    "right": "the right object is larger",
    "equal": "the two objects are equal in size",
}

SYNTHETIC_LABELS = [
    "left_bigger",
    "right_bigger",
    "top_bigger",
    "bottom_bigger",
    "same_size",
]
SYNTHETIC_PROMPTS = {
    "left_bigger": "the object on the left is larger",
    "right_bigger": "the object on the right is larger",
    "top_bigger": "the object on the top is larger",
    "bottom_bigger": "the object on the bottom is larger",
    "same_size": "the two objects are the same size",
}

PROMPTS = {**SYNTHETIC_PROMPTS, **IVQA_PROMPTS}

# Backwards compatibility for imports expecting LABELS / PROMPTS dict shape
LABELS = SYNTHETIC_LABELS


def build_clip(device):
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device).eval()
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    return model, processor


@torch.no_grad()
def clip_predict(model, processor, image, labels=None, row=None, device="cpu"):
    if row is not None:
        from datasets.synthetic_loader import candidate_labels_for_row

        labels = candidate_labels_for_row(row)
    elif labels is None:
        labels = list(SYNTHETIC_LABELS)
    prompts = [PROMPTS[l] for l in labels]
    inp = processor(text=prompts, images=image, return_tensors="pt", padding=True).to(device)
    logits = model(**inp).logits_per_image.squeeze(0)
    probs = F.softmax(logits, dim=0)
    idx = probs.argmax().item()
    return labels[idx], probs[idx].item(), {labels[i]: probs[i].item() for i in range(len(labels))}

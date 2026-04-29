import torch
import torch.nn.functional as F
from transformers import CLIPModel, CLIPProcessor


LABELS = ["left", "right", "equal"]
PROMPTS = {
    "left": "the left object is larger",
    "right": "the right object is larger",
    "equal": "the two objects are equal in size",
}


def build_clip(device):
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device).eval()
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    return model, processor


@torch.no_grad()
def clip_predict(model, processor, image, labels=None, device="cpu"):
    labels = labels or LABELS
    prompts = [PROMPTS.get(l, l) for l in labels]
    inp = processor(text=prompts, images=image, return_tensors="pt", padding=True).to(device)
    logits = model(**inp).logits_per_image.squeeze(0)
    probs = F.softmax(logits, dim=0)
    idx = probs.argmax().item()
    return labels[idx], probs[idx].item(), {labels[i]: probs[i].item() for i in range(len(labels))}

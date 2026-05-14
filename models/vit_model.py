"""ViT-B/16 backbone with the classification head replaced by identity."""

import torch.nn as nn
from torchvision.models import ViT_B_16_Weights, vit_b_16


def build_vit_b16(device):
    """ImageNet weights, identity last linear, default transform, feature dim 768."""
    w = ViT_B_16_Weights.DEFAULT
    m = vit_b_16(weights=w).to(device).eval()
    m.heads[-1] = nn.Identity()
    return m, w.transforms(), 768

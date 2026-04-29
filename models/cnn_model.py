import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.models import ResNet50_Weights, EfficientNet_B0_Weights
from torchvision.models import efficientnet_b0, resnet50


def build_resnet50(device):
    w = ResNet50_Weights.DEFAULT
    m = resnet50(weights=w).to(device).eval()
    m.fc = nn.Identity()
    return m, w.transforms(), 2048


def build_efficientnet_b0(device):
    w = EfficientNet_B0_Weights.DEFAULT
    m = efficientnet_b0(weights=w).to(device).eval()
    m.classifier = nn.Identity()
    return m, w.transforms(), 1280


class ProbeHead(nn.Module):
    def __init__(self, in_dim, out_dim=3):
        super().__init__()
        self.fc = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        return self.fc(x)


def extract_features(backbone, dataset, device, batch_size=64):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    xs, ys, rows = [], [], []
    with torch.no_grad():
        for xb, yb, rb in loader:
            xb = xb.to(device)
            fb = backbone(xb).cpu()
            xs.append(fb)
            ys.append(yb)
            rows.extend([{k: rb[k][i] if hasattr(rb[k], "__len__") else rb[k] for k in rb} for i in range(len(yb))])
    x = torch.cat(xs, dim=0)
    y = torch.cat(ys, dim=0)
    return x, y, rows


def train_probe(x_train, y_train, in_dim, epochs=10, lr=1e-2, device="cpu"):
    head = ProbeHead(in_dim).to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    x_train, y_train = x_train.to(device), y_train.to(device)
    for _ in range(epochs):
        logits = head(x_train)
        loss = loss_fn(logits, y_train)
        opt.zero_grad()
        loss.backward()
        opt.step()
    return head


@torch.no_grad()
def predict_probe(head, x):
    logits = head(x)
    probs = torch.softmax(logits, dim=1)
    pred = probs.argmax(dim=1)
    return pred.cpu(), probs.cpu()

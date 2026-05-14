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
    def __init__(self, in_dim, out_dim=5):
        super().__init__()
        self.fc = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        return self.fc(x)


def extract_features(backbone, dataset, device, batch_size=64):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    xs, ys, rows = [], [], []

    def _take_i(v, i):
        if isinstance(v, dict):
            return {k: _take_i(v[k], i) for k in v}
        if isinstance(v, (list, tuple)):
            if len(v) == 0:
                return v
            if i < len(v):
                return _take_i(v[i], 0) if isinstance(v[i], dict) else v[i]
            return v
        try:
            return v[i]
        except Exception:
            return v

    with torch.no_grad():
        for xb, yb, rb in loader:
            xb = xb.to(device)
            fb = backbone(xb).cpu()
            xs.append(fb)
            ys.append(yb)
            if isinstance(rb, dict):
                rows.extend([{k: _take_i(rb[k], i) for k in rb} for i in range(len(yb))])
            else:
                rows.extend([{} for _ in range(len(yb))])
    x = torch.cat(xs, dim=0)
    y = torch.cat(ys, dim=0)
    return x, y, rows


def train_probe(x_train, y_train, in_dim, epochs=10, lr=1e-2, device="cpu", num_classes=5):
    head = ProbeHead(in_dim, out_dim=num_classes).to(device)
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


@torch.no_grad()
def predict_probe_task_masked(head, x, rows, device="cpu"):
    """Argmax over only the three labels valid for each row's illusion type."""
    from datasets.synthetic_loader import valid_class_indices_for_row

    head = head.to(device)
    x = x.to(device)
    logits = head(x)
    for i, row in enumerate(rows):
        keep = set(valid_class_indices_for_row(row if isinstance(row, dict) else {}))
        for j in range(logits.shape[1]):
            if j not in keep:
                logits[i, j] = float("-inf")
    probs = torch.softmax(logits, dim=1)
    pred = probs.argmax(dim=1)
    return pred.cpu(), probs.cpu()

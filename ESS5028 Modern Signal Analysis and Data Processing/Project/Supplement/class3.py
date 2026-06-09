"""M3 — 2D sliding cross-correlation (PyTorch conv2d)."""

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

MAX_SHIFT = 8
BATCH_SIZE = 64
NPZ = Path(__file__).resolve().parent / "mnist_selected.npz"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

data = np.load(NPZ)
ref_images = data["ref_images"].astype(np.float64)
ref_labels = data["ref_labels"]
test_images = data["test_images"].astype(np.float64)
test_labels = data["test_labels"]

ref_t = torch.from_numpy(ref_images).float().to(device)
test_t = torch.from_numpy(test_images).float().to(device)
ref_labels_t = torch.from_numpy(ref_labels).long()
ref_xx = (ref_t * ref_t).sum(dim=(1, 2))
r_yy_all = (test_t * test_t).sum(dim=(1, 2))

predictions = np.zeros(len(test_images), dtype=np.int64)
n_test = len(test_images)

for start in range(0, n_test, BATCH_SIZE):
    end = min(start + BATCH_SIZE, n_test)
    samples = test_t[start:end]
    r_yy = r_yy_all[start:end]
    padded = F.pad(samples, (MAX_SHIFT, MAX_SHIFT, MAX_SHIFT, MAX_SHIFT))
    r_xy = F.conv2d(padded.unsqueeze(1), ref_t.unsqueeze(1))
    denom = torch.sqrt(ref_xx.view(1, -1, 1, 1) * r_yy.view(-1, 1, 1, 1))
    best_coeff = (r_xy / denom).flatten(2).amax(dim=2)
    predictions[start:end] = ref_labels_t[best_coeff.argmax(dim=1).cpu()].numpy()
    if end % 500 < BATCH_SIZE or end == n_test:
        print(f"  progress {end}/{n_test}")

overall = (test_labels == predictions).mean()
print("M3 — 2D sliding correlation")
print(f"  overall accuracy: {overall * 100:.2f}%")

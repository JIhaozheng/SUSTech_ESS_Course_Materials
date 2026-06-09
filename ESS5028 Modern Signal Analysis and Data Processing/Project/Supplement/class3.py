"""M4 — 4x4 grid of 7x7 blocks, sum of peak block scores."""

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

GRID, CELL, BLOCK_SHIFT = 4, 7, 3
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
r_yy_all = (test_t * test_t).sum(dim=(1, 2))

# split ref into blocks: (n_ref, grid, grid, cell, cell)
n_ref = ref_t.shape[0]
ref_grids = ref_t.reshape(n_ref, GRID, CELL, GRID, CELL).permute(0, 1, 3, 2, 4)
ref_xx_grids = (ref_grids * ref_grids).sum(dim=(-2, -1))

predictions = np.zeros(len(test_images), dtype=np.int64)
n_test = len(test_images)

for start in range(0, n_test, BATCH_SIZE):
    end = min(start + BATCH_SIZE, n_test)
    samples = test_t[start:end]
    r_yy = r_yy_all[start:end]
    scores = torch.zeros(samples.shape[0], n_ref, device=device)

    for bi in range(GRID):
        for bj in range(GRID):
            y0, x0 = bi * CELL, bj * CELL
            ref_blocks = ref_grids[:, bi, bj]
            sample_blocks = samples[:, y0 : y0 + CELL, x0 : x0 + CELL]
            ref_xx = ref_xx_grids[:, bi, bj]
            padded = F.pad(sample_blocks, (BLOCK_SHIFT, BLOCK_SHIFT, BLOCK_SHIFT, BLOCK_SHIFT))
            r_xy = F.conv2d(padded.unsqueeze(1), ref_blocks.unsqueeze(1))
            denom = torch.sqrt(ref_xx.view(1, -1, 1, 1) * r_yy.view(-1, 1, 1, 1))
            scores += (r_xy / denom).flatten(2).amax(dim=2)

    predictions[start:end] = ref_labels_t[scores.argmax(dim=1).cpu()].numpy()
    if end % 500 < BATCH_SIZE or end == n_test:
        print(f"  progress {end}/{n_test}")

overall = (test_labels == predictions).mean()
print("M4 — 7x7 block correlation")
print(f"  overall accuracy: {overall * 100:.2f}%")

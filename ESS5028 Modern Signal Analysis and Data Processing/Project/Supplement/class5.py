"""M5 — multi-scale blocks + global shift; saves output/confusion_matrix.png."""

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

GLOBAL_SHIFT = 4
BATCH_SIZE = 64
NUM_CLASSES = 10
NPZ = Path(__file__).resolve().parent / "mnist_selected.npz"
OUT_DIR = Path(__file__).resolve().parent / "output"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

data = np.load(NPZ)
ref_images = data["ref_images"].astype(np.float64)
ref_labels_np = data["ref_labels"]
test_images = data["test_images"].astype(np.float64)
test_labels = data["test_labels"]

ref_t = torch.from_numpy(ref_images).float().to(device)
test_t = torch.from_numpy(test_images).float().to(device)
ref_labels = torch.from_numpy(ref_labels_np).long().to(device)
r_yy_all = (test_t * test_t).sum(dim=(1, 2))

# precompute two scales: (4,7) and (2,14)
scales = [(4, 7, 3), (2, 14, 3)]
ref_scale_data = []
for grid, cell, _ in scales:
    n_ref = ref_t.shape[0]
    grids = ref_t.reshape(n_ref, grid, cell, grid, cell).permute(0, 1, 3, 2, 4)
    xx = (grids * grids).sum(dim=(-2, -1))
    ref_scale_data.append((grids, xx))

predictions = np.zeros(len(test_images), dtype=np.int64)
n_test = len(test_images)
_, h, w = test_t.shape

for start in range(0, n_test, BATCH_SIZE):
    end = min(start + BATCH_SIZE, n_test)
    samples = test_t[start:end]
    r_yy = r_yy_all[start:end]
    padded = F.pad(samples, (GLOBAL_SHIFT, GLOBAL_SHIFT, GLOBAL_SHIFT, GLOBAL_SHIFT))
    best_cls = torch.full((samples.shape[0], NUM_CLASSES), -torch.inf, device=device)

    for dy in range(2 * GLOBAL_SHIFT + 1):
        for dx in range(2 * GLOBAL_SHIFT + 1):
            aligned = padded[:, dy : dy + h, dx : dx + w]
            n_ref = ref_scale_data[0][0].shape[0]
            ref_scores = torch.zeros(aligned.shape[0], n_ref, device=device)

            for (grid, cell, block_shift), (ref_grids, ref_xx_grids) in zip(scales, ref_scale_data):
                scale_score = torch.zeros_like(ref_scores)
                for bi in range(grid):
                    for bj in range(grid):
                        y0, x0 = bi * cell, bj * cell
                        ref_blocks = ref_grids[:, bi, bj]
                        sample_blocks = aligned[:, y0 : y0 + cell, x0 : x0 + cell]
                        ref_xx = ref_xx_grids[:, bi, bj]
                        pad = F.pad(sample_blocks, (block_shift, block_shift, block_shift, block_shift))
                        r_xy = F.conv2d(pad.unsqueeze(1), ref_blocks.unsqueeze(1))
                        denom = torch.sqrt(ref_xx.view(1, -1, 1, 1) * r_yy.view(-1, 1, 1, 1))
                        scale_score += (r_xy / denom).flatten(2).amax(dim=2)
                ref_scores += scale_score

            cls_scores = torch.full((ref_scores.shape[0], NUM_CLASSES), -torch.inf, device=device)
            idx = ref_labels.unsqueeze(0).expand(ref_scores.shape[0], -1)
            cls_scores.scatter_reduce_(1, idx, ref_scores, reduce="amax", include_self=True)
            best_cls = torch.maximum(best_cls, cls_scores)

    predictions[start:end] = best_cls.argmax(dim=1).cpu().numpy()
    if end % 1000 < BATCH_SIZE or end == n_test:
        print(f"  progress {end}/{n_test}")

overall = (test_labels == predictions).mean()
print("M5 — multi-scale block correlation")
print(f"  overall accuracy: {overall * 100:.2f}%")

# confusion matrix
cm = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
for t, p in zip(test_labels, predictions):
    cm[int(t), int(p)] += 1

fig, ax = plt.subplots(figsize=(8, 7))
ax.imshow(cm, cmap="Blues")
ax.set_xticks(range(NUM_CLASSES))
ax.set_yticks(range(NUM_CLASSES))
ax.set_xlabel("Predicted label")
ax.set_ylabel("True label")
ax.set_title(f"M5 confusion matrix\nOverall accuracy: {overall * 100:.2f}%")
OUT_DIR.mkdir(parents=True, exist_ok=True)
out_path = OUT_DIR / "confusion_matrix.png"
fig.tight_layout()
fig.savefig(out_path, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Confusion matrix saved -> {out_path.resolve()}")

"""M1 — zero-lag 2D cross-correlation on the full image."""

from pathlib import Path

import numpy as np

NPZ = Path(__file__).resolve().parent / "mnist_selected.npz"
data = np.load(NPZ)
ref_images = data["ref_images"].astype(np.float64)
ref_labels = data["ref_labels"]
test_images = data["test_images"].astype(np.float64)
test_labels = data["test_labels"]

ref_rr = np.sum(ref_images * ref_images, axis=(1, 2))
predictions = np.zeros(len(test_images), dtype=np.int64)

for i, sample in enumerate(test_images):
    rs = np.sum(ref_images * sample, axis=(1, 2))
    ss = np.sum(sample * sample)
    coeff = rs / np.sqrt(ref_rr * ss)
    predictions[i] = ref_labels[coeff.argmax()]

overall = (test_labels == predictions).mean()
print("M1 — zero-lag 2D correlation")
print(f"  overall accuracy: {overall * 100:.2f}%")

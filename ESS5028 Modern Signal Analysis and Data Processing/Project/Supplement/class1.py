"""M2 — 1D sliding cross-correlation (flattened raster)."""

from pathlib import Path

import numpy as np

MAX_SHIFT = 4
NPZ = Path(__file__).resolve().parent / "mnist_selected.npz"
data = np.load(NPZ)
ref_images = data["ref_images"].astype(np.float64)
ref_labels = data["ref_labels"]
test_images = data["test_images"].astype(np.float64)
test_labels = data["test_labels"]

n_ref = len(ref_images)
ref_flat = ref_images.reshape(n_ref, -1)
signal_len = ref_flat.shape[1]
ref_rr = np.sum(ref_flat * ref_flat, axis=1)
predictions = np.zeros(len(test_images), dtype=np.int64)

for i, img in enumerate(test_images):
    sample_1d = img.ravel()
    ss = np.sum(sample_1d * sample_1d)
    padded = np.pad(sample_1d, MAX_SHIFT, mode="constant")
    best_coeff = np.full(n_ref, -1.0)
    for shift in range(2 * MAX_SHIFT + 1):
        window = padded[shift : shift + signal_len]
        rs = ref_flat @ window
        coeff = rs / np.sqrt(ref_rr * ss)
        best_coeff = np.maximum(best_coeff, coeff)
    predictions[i] = ref_labels[best_coeff.argmax()]
    if (i + 1) % 500 == 0:
        print(f"  progress {i + 1}/{len(test_images)}")

overall = (test_labels == predictions).mean()
print("M2 — 1D sliding correlation")
print(f"  overall accuracy: {overall * 100:.2f}%")

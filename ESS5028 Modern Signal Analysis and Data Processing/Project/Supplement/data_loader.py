"""Build mnist_selected.npz: split MNIST into reference / test, zero-mean + L2 norm."""

import struct
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
OUT_NPZ = ROOT / "mnist_selected.npz"

REF_PER_CLASS = 100
TEST_PER_CLASS = 3000
NUM_CLASSES = 10
SEED = 40

# --- pick idx files in data/ (train preferred) ---
img_path = DATA_DIR / "train-images.idx3-ubyte"
lbl_path = DATA_DIR / "train-labels.idx1-ubyte"
if not img_path.is_file():
    img_path = DATA_DIR / "t10k-images.idx3-ubyte"
    lbl_path = DATA_DIR / "t10k-labels.idx1-ubyte"

with open(img_path, "rb") as f:
    _, n_img, rows, cols = struct.unpack(">IIII", f.read(16))
    all_images = np.frombuffer(f.read(), dtype=np.uint8).reshape(n_img, rows, cols)
with open(lbl_path, "rb") as f:
    _, n_img = struct.unpack(">II", f.read(8))
    all_labels = np.frombuffer(f.read(), dtype=np.uint8)

per_class = n_img // NUM_CLASSES
if REF_PER_CLASS + TEST_PER_CLASS > per_class:
  # t10k: only 1000 per digit
    TEST_PER_CLASS = per_class - REF_PER_CLASS
    print(f"Using test_per_class={TEST_PER_CLASS} (max for this file)")

rng = np.random.default_rng(SEED)
ref_images, ref_labels, test_images, test_labels = [], [], [], []

for digit in range(NUM_CLASSES):
    idx = np.where(all_labels == digit)[0]
    idx = rng.permutation(idx)
    ref_images.append(all_images[idx[:REF_PER_CLASS]])
    test_images.append(all_images[idx[REF_PER_CLASS : REF_PER_CLASS + TEST_PER_CLASS]])
    ref_labels.append(np.full(REF_PER_CLASS, digit, dtype=np.uint8))
    test_labels.append(np.full(TEST_PER_CLASS, digit, dtype=np.uint8))

ref_images = np.concatenate(ref_images).astype(np.float64)
ref_labels = np.concatenate(ref_labels)
test_images = np.concatenate(test_images).astype(np.float64)
test_labels = np.concatenate(test_labels)

# zero-mean + L2 per image
flat = ref_images.reshape(len(ref_images), -1)
flat -= flat.mean(axis=1, keepdims=True)
norm = np.linalg.norm(flat, axis=1, keepdims=True)
norm = np.where(norm > 1e-8, norm, 1.0)
ref_images = flat / norm
ref_images = ref_images.reshape(ref_images.shape[0], rows, cols)

flat = test_images.reshape(len(test_images), -1)
flat -= flat.mean(axis=1, keepdims=True)
norm = np.linalg.norm(flat, axis=1, keepdims=True)
norm = np.where(norm > 1e-8, norm, 1.0)
test_images = flat / norm
test_images = test_images.reshape(test_images.shape[0], rows, cols)

np.savez(
    OUT_NPZ,
    ref_images=ref_images,
    ref_labels=ref_labels,
    test_images=test_images,
    test_labels=test_labels,
)

print(f"Saved {OUT_NPZ}")
print(f"  ref: {NUM_CLASSES} x {REF_PER_CLASS} = {len(ref_labels)}")
print(f"  test: {NUM_CLASSES} x {TEST_PER_CLASS} = {len(test_labels)}")
print(f"  seed: {SEED}")

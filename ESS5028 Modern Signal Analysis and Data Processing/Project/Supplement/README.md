# MNIST normalized cross-correlation (M1–M5)

Simple scripts: no shared library — each file runs top to bottom.

## Files

```
ncc_mnist/
  data/                      MNIST idx files
  data_loader.py             run once → mnist_selected.npz
  class0.py                  M1
  class1.py                  M2
  class2.py                  M3 (PyTorch)
  class3.py                  M4 (PyTorch)
  class4.py                  M5 (PyTorch) + output/confusion_matrix.png
  mnist_selected.npz         created by data_loader.py
```

## Setup

```bash
pip install -r requirements.txt
```

Packages: `numpy`, `matplotlib` (class4 only), `torch` (class2–class4).

Put MNIST in `data/`:

- `train-images.idx3-ubyte` + `train-labels.idx1-ubyte` (preferred), or
- `t10k-images.idx3-ubyte` + `t10k-labels.idx1-ubyte` (auto caps test size)

## Run

```bash
python data_loader.py
python class0.py
python class1.py
python class2.py
python class3.py
python class4.py
```

Default split: 100 references and 3000 test images per digit, seed 40, zero-mean + L2 normalization.

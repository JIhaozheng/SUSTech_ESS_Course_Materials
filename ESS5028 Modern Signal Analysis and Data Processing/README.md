# ESS5028 Modern Signal Analysis and Data Processing
&gt; **Note:** Other homework assignments are not provided here since I used the code provided by the lecturer rather than writing it myself. The notebooks provided by me are more detailed in numerical implementation than those provided by the lecturer.

---
## Homework 1
A notebook to compile music and perform Short-Time Fourier Transform (STFT) analysis for basic musical scales and pieces such as *Canon* and *Yun Gong Xun Yin*.

<img width="360" alt="DoReMi" src="https://github.com/user-attachments/assets/a1062333-f210-4d11-902c-115cbed0e658" />
<img width="360" alt="YunGongXunYin" src="https://github.com/user-attachments/assets/1d843f30-cb55-4c2a-86bf-0a589d01396a" />

## Homework 2
A notebook demonstrating the **Nyquist Sampling Theorem** and the **leakage effect**.  
Shows the derivative and integral of a function with their Fourier Series relationships.

<img height="300" alt="Aliasing" src="https://github.com/user-attachments/assets/475d82e5-6c80-41ab-93c0-0e869ff51a36" />
<img height="300" alt="tri_10_integral_Fourier" src="https://github.com/user-attachments/assets/d1059e91-d228-4c7b-b6f7-b030727b7f6a" />

## Homework 3
A notebook showing how to use **Green's function** and **convolution** to solve a 2nd-order ordinary differential equation under **over-critically** and **under-damped** conditions, using 4th-order Runge-Kutta as benchmark.

<img width="600" alt="CompareGreenFunction_over" src="https://github.com/user-attachments/assets/e06262bf-50e2-4e7c-b128-102c6b61f653" />
<img width="600" alt="CompareGreenFunction_critically" src="https://github.com/user-attachments/assets/d91df266-8b9f-491e-806b-60b11dfd1657" />
<img width="600" alt="CompareGreenFunction_under" src="https://github.com/user-attachments/assets/c4f4d497-5402-4c88-98e5-10fa55bc7f81" />

## Homework 5 & 9
Demonstrates that the **Fourier transform of $f * g$** equals the **product of their transforms** in both continuous and discrete forms.
Demonstrates that the **Laplace transform of $f * g$** equals the **product of their transforms** in continuous form.

---
# From Global to Local: Training-Free Handwritten Digit Classification by Normalized Cross-Correlation

Handwritten digits are classified by **normalized cross-correlation (NCC)** against a small library of reference templates. No network weights are trained. Five methods (**M1–M5**) compare global whole-image matching, shift-tolerant search, and local multi-scale block matching in 1D and 2D.

## Highlights

| Method | Idea | Overall accuracy (100 refs / class) |
|--------|------|-------------------------------------|
| M1 | Fixed 2D alignment | 89.5% |
| M2 | 1D raster + shift | 90.2% |
| M3 | 2D translation | 91.8% |
| M4 | Local 7×7 blocks | 92.5% |
| M5 | Multi-scale blocks + global shift | **94.8%** |

With only **10 references per class**, M5 reaches **86.3%** (+11.4 pp over M1).

## Preview

<p align="center">
  <img src="https://github.com/user-attachments/assets/cf1fdf36-579a-471a-8177-bddb669555a3" alt="Confusion matrix for Method 4" width="285"/>
  &nbsp;&nbsp;
  <img src="https://github.com/user-attachments/assets/186a843f-b8ec-440c-8180-39a0d2252a9c" alt="Method 5 misclassified examples" width="350"/>
</p>

<p align="center">
  <em>Left: confusion matrix for Method 4 (100 refs / class).</em> &nbsp; Right: Method 5 misclassified samples (true → predicted).
</p>

## Repository layout

```
Supplement/
  data/                 MNIST idx files
  data_loader.py        build mnist_selected.npz
  class1.py             M1 — fixed 2D NCC
  class2.py             M2 — 1D raster + shift
  class3.py             M3 — 2D translation (PyTorch)
  class4.py             M4 — local 7×7 blocks (PyTorch)
  class5.py             M5 — multi-scale blocks (PyTorch)
  requirements.txt
  README.md
```

## Setup

```bash
pip install -r requirements.txt
```

Dependencies: `numpy`, `matplotlib` (figures), `torch` (M3–M5).

Place MNIST under `data/`:

- `train-images.idx3-ubyte` + `train-labels.idx1-ubyte`
- `t10k-images.idx3-ubyte` + `t10k-labels.idx1-ubyte`

## Run

```bash
python data_loader.py
python class1.py
python class2.py
python class3.py
python class4.py
python class5.py
```

Default split: **100 references** and **3000 test images** per digit (30,000 tests total), seed 40. Images are zero-meaned and L2-normalized before matching.

## Methods

- **M1–M3:** match the full 28×28 pattern (fixed alignment, 1D shift, or 2D shift).
- **M4:** score independent 7×7 blocks — local stroke features dominate.
- **M5:** combine fine (7×7) and coarse (14×14) blocks with a short whole-image shift; best overall accuracy without training.

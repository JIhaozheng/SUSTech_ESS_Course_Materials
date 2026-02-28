This project aims to implement classical machine learning models using only basic NumPy.

For more detailed information, see `report.pdf`.

**Dataset:** Download and decompress the MNIST training and testing sets from [mnist-datasets](https://pypi.org/project/mnist-datasets/):

- `train-images.idx3-ubyte`
- `train-labels.idx1-ubyte`
- `t10k-images.idx3-ubyte`
- `t10k-labels.idx1-ubyte`

**Usage:**

1. Run `data_loader.py` to process the raw data and generate `.npy` files. This script applies sparse matrix representation and spectral normalization techniques to reduce feature dimensionality.

2. Run the following experiment scripts to train and evaluate models:

| Script | Model |
|--------|-------|
| `experiment_dt.py` | Decision Tree |
| `experiment_knn.py` | K-Nearest Neighbors (KNN) |
| `experiment_mlp.py` | Multi-Layer Perceptron (MLP) |
| `experiment_mlr.py` | Multiclass Logistic Regression (MLR) |

**Core Modules:**

| File | Description |
|------|-------------|
| `kd_tree.py` | KD-Tree data structure for efficient nearest neighbor search |
| `knn_classifier.py` | KNN classifier implementation |
| `logistic_regression.py` | Multiclass Logistic regression model |
| `mlp_classifier.py` | Multi-layer perceptron classifier |
| `metrics.py` | Evaluation metrics for model performance |
| `recover_image_data.py` | Image reconstruction from sparse encoded data |

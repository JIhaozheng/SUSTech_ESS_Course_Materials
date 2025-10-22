import numpy as np
import matplotlib.pyplot as plt
import os

os.makedirs("images", exist_ok=True)


class LinearRegression:
    def __int__(self):
        self.w = None
        self.norm_mode = None
        self.nsample, self.ndim = None, None
        # if norm_mode=minmax
        self.x_min, self.x_max = None, None
        # if norm_mode=mean
        self.x_mean, self.x_std = None, None

    def mean_squared_error(self, true, pred):
        squared_error = np.square(true - pred)
        sum_squared_error = np.sum(squared_error)
        mse_loss = sum_squared_error / true.size
        return mse_loss

    def normalize(self, X, norm_mode=None):
        self.norm_mode = norm_mode
        X_norm = X.copy()
        if norm_mode == "minmax":
            self.x_min = X.min(axis=0)
            self.x_max = X.max(axis=0)
            X_norm = (X - self.x_min) / (self.x_max - self.x_min)
        elif norm_mode == "mean":
            self.x_mean = X.mean(axis=0)
            self.x_std = X.std(axis=0)
            X_norm = (X - self.x_mean) / self.x_std
        elif norm_mode == "none":
            pass
        else:
            raise ValueError("norm_mode must be 'minmax', 'mean' or None")

        return X_norm

    def _normalize(self, X, norm_mode=None):
        if self.norm_mode == "minmax":
            return (X - self.x_min) / (self.x_max - self.x_min)
        elif self.norm_mode == "mean":
            return (X - self.x_mean) / self.x_std
        else:
            return X

    def fit(
        self,
        X_train,
        y_train,
        method="bgd",
        epochs=100,
        lr=0.01,
        norm_mode=None,
        batch_size=32,
        grad_norm=None,
    ):
        self.nsample, self.ndim = X_train.shape
        X_norm = self.normalize(X_train, norm_mode)
        b = np.ones((X_norm.shape[0], 1))
        X_norm = np.hstack([b, X_norm])
        self.w = np.zeros((self.ndim + 1))
        losses = []
        for epoch in range(epochs):
            if method == "bgd":
                y_pred = X_norm @ self.w
                dw = -X_norm.T @ (y_train - y_pred) / self.nsample
                if grad_norm == 1:
                    dw = dw / np.sqrt(np.sum(np.square(dw)))
                self.w = self.w - lr * dw
                loss = self.mean_squared_error(y_train, y_pred)
                losses.append(loss)
            elif method == "sgd":
                idx = np.arange(self.nsample)
                np.random.shuffle(idx)
                for ii in idx:
                    xi = X_norm[ii]
                    yi = y_train[ii]
                    y_pred = xi @ self.w
                    dw = -(yi - y_pred) * xi
                    if grad_norm == 1:
                        dw = dw / np.sqrt(np.sum(np.square(dw)))
                    self.w = self.w - lr * dw
                y_pred = X_norm @ self.w
                loss = self.mean_squared_error(y_train, y_pred)
                losses.append(loss)
            elif method == "mbgd":
                idx = np.arange(self.nsample)
                np.random.shuffle(idx)
                for s_idx in range(0, self.nsample, batch_size):
                    e_idx = s_idx + batch_size
                    batch_idx = idx[s_idx:e_idx]
                    X_batch = X_norm[batch_idx]
                    y_batch = y_train[batch_idx]
                    y_pred = X_batch @ self.w
                    dw = -X_batch.T @ (y_batch - y_pred) / X_batch.shape[0]
                    if grad_norm == 1:
                        dw = dw / np.sqrt(np.sum(np.square(dw)))
                    self.w = self.w - lr * dw
                y_pred = X_norm @ self.w
                loss = self.mean_squared_error(y_train, y_pred)
                losses.append(loss)
            else:
                raise ValueError("Gradient descent must be sgd, bgd, mbgd")

        return losses

    def predict(self, X_pred):
        X_norm = self._normalize(X_pred, norm_mode=None)
        b = np.ones((X_norm.shape[0], 1))
        X_norm = np.hstack((b, X_norm))
        return X_norm @ self.w


#########################################################################################
np.random.seed(42)
X_train = np.arange(100).reshape(100, 1)
a, b = 1, 10
y_train = a * X_train + b + np.random.normal(0, 5, size=X_train.shape)
y_train = y_train.reshape(-1)
N_epochs = 1200
Lr = 1e-2
method = "sgd"
norm_mode = "mean"
batch_size = 32
grad_norm = 1

# Model training
# method: only supports "sgd", "bgd", and "mbgd"
# norm_mode: only supports "min-max", "mean", and "none"
# grad_norm: 1 for gradient normalization, 0 (or other values) for no normalization
model = LinearRegression()
losses = model.fit(
    X_train,
    y_train,
    method=method,
    epochs=N_epochs,
    lr=Lr,
    norm_mode=norm_mode,
    batch_size=batch_size,
    grad_norm=grad_norm,
)

# Predictions
X_test = np.arange(100).reshape(100, 1)
y_pred = a * X_test + b  # True values
y_test = model.predict(X_test)  # Model predictions

param_str = f"Method:{method.upper()}, Epochs:{N_epochs}, LR:{Lr}, Norm:{norm_mode}, GradNorm:{bool(grad_norm)}"
filename_str = f"{method}_{N_epochs}ep_{Lr}lr_{norm_mode}_gradnorm{grad_norm}"

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

# Subplot 1: Regression results
ax1.scatter(X_train, y_train, alpha=0.5, color="b", label="Training data", s=20)
ax1.plot(X_test, y_pred.flatten(), color="r", label="True line", linewidth=2)
ax1.plot(X_test, y_test, color="k", label="Predicted line", linewidth=2, linestyle="--")
ax1.legend()
ax1.set_xlabel("X")
ax1.set_ylabel("y")
ax1.set_title(f"Linear Regression Results")
ax1.grid(True, alpha=0.3)

# Subplot 2: Training loss with proper label
ax2.plot(
    np.arange(N_epochs),
    losses,
    color="green",
    linewidth=2,
    label=f"Training Loss (Final: {losses[-1]:.4f})",
)
ax2.legend()
ax2.set_xlabel("Epochs")
ax2.set_ylabel("Loss")
ax2.set_title(f"Training Loss Over Time")
ax2.grid(True, alpha=0.3)

fig.suptitle(f"Linear Regression Analysis - {param_str}", fontsize=14, y=0.98)

plt.tight_layout()
plt.subplots_adjust(top=0.90)
plt.savefig(f"images/{filename_str}.png", dpi=300, bbox_inches="tight")
plt.show()

print(f"Results saved as: images/{filename_str}.png")
print(f"Final MSE: {losses[-1]:.4f}")
print(f"Learned weights: {model.w}")

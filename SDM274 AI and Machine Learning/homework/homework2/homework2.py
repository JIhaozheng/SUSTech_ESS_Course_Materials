import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

os.makedirs("images", exist_ok=True)


class LogisticRegression:
    def __init__(self):
        self.w = None
        self.nsample, self.ndim = None, None
        self.X, self.Y = None, None
        self.norm_mode = None
        self.x_min, self.x_max = None, None
        self.x_mean, self.x_std = None, None

    def _f(self, x):
        return 1 / (1 + np.exp(-x))

    def normalize(self, X):
        if self.norm_mode == "min-max":
            self.x_min = X.min(axis=0)
            self.x_max = X.max(axis=0)
            X_norm = (X - self.x_min) / (self.x_max - self.x_min)
        elif self.norm_mode == "mean":
            self.x_mean = X.mean(axis=0)
            self.x_std = X.std(axis=0)
            X_norm = (X - self.x_mean) / self.x_std
        elif self.norm_mode == None:
            pass
        else:
            raise ValueError("norm_mode must be 'mean', 'min-max' or None")
        return X_norm

    def _normalize(self, X_pred):
        if self.norm_mode == "min-max":
            return (X_pred - self.x_min) / (self.x_max - self.x_min)
        elif self.norm_mode == "mean":
            return (X_pred - self.x_mean) / self.x_std
        elif self.norm_mode == None:
            return X_pred

    def _pred(self, x):
        return self._f(self.w @ x)

    def fit(
        self, X, Y, method=None, norm_mode=None, epochs=100, lr=0.01, batch_size=32
    ):
        self.X = X
        self.norm_mode = norm_mode
        self.Y = Y
        self.nsample, self.ndim = X.shape
        self.w = np.ones((self.ndim + 1))
        X_norm = self.normalize(X)
        b = np.ones((self.nsample, 1))
        X_norm = np.hstack((b, X_norm))
        losses = []

        for epoch in range(epochs):
            if method == "sgd":
                idx = np.arange(self.nsample)
                np.random.shuffle(idx)
                for ii in idx:
                    xi = X_norm[ii, :]
                    yi = Y[ii]
                    y_pred = self._pred(xi)
                    dw = -xi * (yi - y_pred)
                    self.w = self.w - lr * dw
                loss = -np.dot(self.Y, np.log(self._f(X_norm @ self.w))) - np.dot(
                    (1 - self.Y), np.log(1 - self._f(X_norm @ self.w))
                )
                losses.append(loss)
            if method == "mbgd":
                idx = np.arange(self.nsample)
                np.random.shuffle(idx)
                for s_idx in range(0, self.nsample, batch_size):
                    e_idx = s_idx + batch_size
                    batch_idx = idx[s_idx:e_idx]
                    X_batch = X_norm[batch_idx]
                    y_batch = Y[batch_idx]
                    y_pred = self._f(X_batch @ self.w)
                    dw = -X_batch.T @ (y_batch - y_pred)
                    self.w = self.w - lr * dw
                epsilon = 1e-8
                loss = -np.dot(
                    self.Y, np.log(self._f(X_norm @ self.w) + epsilon)
                ) - np.dot((1 - self.Y), np.log(1 - self._f(X_norm @ self.w) + epsilon))
                losses.append(loss)
        return losses, self.w

    def predict(self, X_pred):
        X_norm = self._normalize(X_pred)
        b = np.ones((X_norm.shape[0], 1))
        X_norm = np.hstack((b, X_norm))
        y_pred = self._f(X_norm @ self.w)
        return y_pred

    def prob2class(self, y, threshold):
        return (y > threshold).astype(int)

    def evaluate(self, ypred, ytest, mode="accuracy"):
        tp, fp, tn, fn = 0, 0, 0, 0
        for ii in range(ytest.shape[0]):
            if ypred[ii] == 1:
                if ytest[ii] == 1:
                    tp += 1
                if ytest[ii] == 0:
                    fp += 1
            elif ypred[ii] == 0:
                if ytest[ii] == 1:
                    fn += 1
                if ytest[ii] == 0:
                    tn += 1
        if mode == "accuracy":
            return (tp + tn) / (len(ypred))
        if mode == "recall":
            return tp / (tp + fn)
        if mode == "precision":
            return tp / (tp + fp)
        if mode == "F1 score":
            R = tp / (tp + fn)
            P = tp / (tp + fp)
            return 2 * P * R / (P + R)


# Problem1 read the table, devide the data set into training set and test set, gen_new 01label for different wines
np.random.seed(42)
df = pd.read_csv("wine.data", header=None)
df_new = df[df[0] != 3].reset_index(drop=True)
X = df_new.iloc[:, 1:].values
Y = df_new.iloc[:, 0].values

y_mean = Y.mean(axis=0)
Y = Y - y_mean
for ii in range(X.shape[0]):
    if Y[ii] < 0:
        Y[ii] = 0
    else:
        Y[ii] = 1

(nsample, ndim) = X.shape
idx = np.arange(nsample)
nper70 = round(nsample * 0.7)
np.random.shuffle(idx)
Xtraining = X[idx[:nper70]]
Ytraining = Y[idx[:nper70]]
Xtest = X[idx[nper70:]]
Ytest = Y[idx[nper70:]]

# Problem2 use the minibatch and stochastic to train the logistic regression mode
method1 = "mbgd"
method2 = "sgd"
norm_mode = "min-max"
N_epoch = 100
Lr = 0.01
batch_size = 16

Model = LogisticRegression()
losses1, weight1 = Model.fit(
    Xtraining,
    Ytraining,
    method=method1,
    norm_mode=norm_mode,
    epochs=N_epoch,
    lr=Lr,
    batch_size=batch_size,
)
# problem3 evaluate the model
y_pred = Model.predict(Xtest)
y_class = Model.prob2class(y_pred, 0.5)

accuracy = Model.evaluate(y_class, Ytest, "accuracy")
recall = Model.evaluate(y_class, Ytest, "recall")
precision = Model.evaluate(y_class, Ytest, "precision")
F1score = Model.evaluate(y_class, Ytest, "F1 score")
print("mini-batch")
print(f"Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
print(f"Recall:    {recall:.4f} ({recall*100:.2f}%)")
print(f"Precision: {precision:.4f} ({precision*100:.2f}%)")
print(f"F1 Score:  {F1score:.4f} ({F1score*100:.2f}%)")

losses2, weight2 = Model.fit(
    Xtraining,
    Ytraining,
    method=method2,
    norm_mode=norm_mode,
    epochs=N_epoch,
    lr=Lr,
    batch_size=batch_size,
)

y_pred = Model.predict(Xtest)
y_class = Model.prob2class(y_pred, 0.5)

# problem3 evaluate the model
accuracy = Model.evaluate(y_class, Ytest, "accuracy")
recall = Model.evaluate(y_class, Ytest, "recall")
precision = Model.evaluate(y_class, Ytest, "precision")
F1score = Model.evaluate(y_class, Ytest, "F1 score")
print("stochastic ")
print(f"Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
print(f"Recall:    {recall:.4f} ({recall*100:.2f}%)")
print(f"Precision: {precision:.4f} ({precision*100:.2f}%)")
print(f"F1 Score:  {F1score:.4f} ({F1score*100:.2f}%)")

plt.plot(np.arange(len(losses1)), losses1, "r", label="minibatch")
plt.plot(np.arange(len(losses2)), losses2, "b--", label="stochastic")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title(f"Training Loss Over Time")
plt.grid(True, alpha=0.3)
plt.legend()
plt.savefig(f"images/Training_Loss_over_time.png", dpi=300, bbox_inches="tight")
plt.show()

f1, f2 = 5, 9  # features
X_plot = X[:, [f1, f2]]
Y_plot = Y

x_min = Xtraining.min(axis=0)
x_max = Xtraining.max(axis=0)
X_plot_norm = (X_plot - x_min[[f1, f2]]) / (x_max[[f1, f2]] - x_min[[f1, f2]])

w0 = weight1[0]
w1 = weight1[1 + f1]
w2 = weight1[1 + f2]

plt.figure(figsize=(8, 6))
colors = ["red", "blue"]
labels = ["Class 0", "Class 1"]
for ii, label in enumerate(np.unique(Y_plot)):
    plt.scatter(
        X_plot_norm[Y_plot == label, 0],
        X_plot_norm[Y_plot == label, 1],
        color=colors[ii],
        label=labels[ii],
        alpha=0.7,
        edgecolor="k",
    )

x1_min, x1_max = X_plot_norm[:, 0].min(), X_plot_norm[:, 0].max()
x1s = np.linspace(x1_min, x1_max, 200)
x2s = -(w0 + w1 * x1s) / w2

plt.plot(x1s, x2s, "k--", linewidth=2, label="Decision Boundary")

plt.xlabel(f"Feature {f1} (normalization)")
plt.ylabel(f"Feature {f2} (normalization)")
plt.title(f"Logistic Regression Projection Plane (Features {f1} vs {f2})")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(
    f"images/Logistic_Regression_Projection_plane_f{f1}_f{f2}.png",
    dpi=300,
    bbox_inches="tight",
)
plt.show()
plt.close()
print(weight1)

import numpy as np
import matplotlib.pyplot as plt
import os

os.makedirs("images", exist_ok=True)


def f(x):
    return np.exp(x)


x0 = 1
x2 = 5
n = 2000

x1 = np.linspace(x0, x2, n)
x = np.linspace(x0, x2, n)
X1, X = np.meshgrid(x1, x)

lin_In = (f(x0) - f(X1)) / (x0 - X1) * (X - X1) + f(x0)

quad_In = (
    f(x0)
    + (X - x0) / (X1 - x0) * (f(X1) - f(x0))
    + ((f(x2) - f(x0)) / (x2 - x0) - (f(x0) - f(X1)) / (x0 - X1))
    * (x - x0)
    * (x - X1)
    / (x2 - X1)
)

lin_err = abs(lin_In - f(X))
quad_err = abs(quad_In - f(X))

fig = plt.figure(figsize=(12, 7))
ax = fig.add_subplot(111, projection="3d")
ax.plot_surface(X, X1, lin_err, cmap="Blues")
ax.plot_surface(X, X1, quad_err, cmap="Oranges")
ax.set_xlabel("x")
ax.set_ylabel("x1")
ax.set_zlabel("Error")
ax.set_title(
    "Linear and Quadratic interpolation error surface at interval [%d,%d]" % (x0, x2)
)
from matplotlib.lines import Line2D

custom_lines = [
    Line2D([0], [0], color="navy", lw=4),
    Line2D([0], [0], color="darkorange", lw=4),
]
ax.legend(
    custom_lines, ["Linear Interpolation", "Quadratic Interpolation"], loc="upper left"
)
ax.view_init(elev=23, azim=55)
plt.draw()
plt.savefig(
    "images/Linear and Quadratic interpolation error surface at interval [%d,%d].png"
    % (x0, x2),
    dpi=300,
    bbox_inches="tight",
    pad_inches=0.2,
)
plt.show()

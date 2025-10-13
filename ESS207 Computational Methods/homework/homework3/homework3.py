# compare Chebyshev
import numpy as np
import matplotlib.pyplot as plt
from homework3_cubic import cubic
import os

os.makedirs("images", exist_ok=True)

x8 = 3
x0 = -3
N = 9
n = N - 1


def f(x):
    return np.exp(-9 * x**2)


xi = np.linspace(x0, x8, N)
yi = np.array([f(xi[k]) for k in range(N)])


def cr2x(cr, x0, x8):
    return (cr + 1) * (x8 - x0) / 2 + x0


def CR(m):
    return np.cos((2 * m + 1) * np.pi / (2 * N))


def diff_table(xi, yi):
    n = len(xi)
    diff = np.zeros((n, n))
    diff[:, 0] = yi
    for ii in range(1, n):
        for jj in range(n - ii):
            diff[jj, ii] = (diff[jj + 1, ii - 1] - diff[jj, ii - 1]) / (
                xi[jj + ii] - xi[jj]
            )
    return diff


def make_newton_inter(diff, xi):
    n = len(xi)

    def newton_inter(x):
        result = diff[0, 0]
        term = 1
        for ii in range(1, n):
            term *= x - xi[ii - 1]
            result = result + diff[0, ii] * term
        return result

    return newton_inter


x_line = np.linspace(x0, x8, 1000)
y_line = f(x_line)
## make equal spaced interpolation
diff_esp = diff_table(xi, yi)
esp = make_newton_inter(diff_esp, xi)
y_esp = esp(x_line)

## make chebyshev interpolation
cheby_point = np.array([CR(m) for m in range(N)])
x_cheby = cr2x(cheby_point, x0, x8)
y_c = f(x_cheby)
diff_cheby = diff_table(x_cheby, f(x_cheby))
cheby = make_newton_inter(diff_cheby, x_cheby)
y_cheby = cheby(x_line)

## make cubic spline
cubic_spline = cubic(f, x0, x8, N)
y_cubic = cubic_spline(x_line)

plt.figure(figsize=(9, 5))
plt.plot(x_line, y_line, color="k", linewidth=2, label="original function")

plt.plot(
    x_line,
    y_cheby,
    label="Chebyshev Interpolation",
    linestyle="-",
    color="b",
    linewidth=2,
)

plt.plot(
    x_line,
    y_esp,
    label="Equally-spaced Interpolation",
    linestyle="--",
    color="g",
    linewidth=2,
)

plt.plot(x_line, y_cubic, label="Cubic Spline", linestyle="-.", color="r", linewidth=2)
plt.scatter(
    x_cheby,
    y_c,
    s=60,
    marker="o",
    edgecolor="k",
    facecolor="white",
    label="Chebyshev nodes",
    zorder=3,
)
plt.scatter(
    xi,
    yi,
    s=60,
    marker="^",
    edgecolor="k",
    facecolor="k",
    label="Equally-spaced nodes",
    zorder=3,
)
plt.xlabel("$x$", fontsize=13)
plt.ylabel("$y$", fontsize=13)
plt.legend()
plt.title("Interpolation Comparison", fontsize=16, fontweight="bold", pad=12)
plt.grid(True, linestyle="--", alpha=0.4)
plt.tight_layout()
plt.savefig("images/InterpolationComparison.png", dpi=300)
plt.show()

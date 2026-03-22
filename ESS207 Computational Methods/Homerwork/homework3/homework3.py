import numpy as np
import matplotlib.pyplot as plt
import os

os.makedirs("images", exist_ok=True)

x0 = -3
x8 = 3
n = 9


def f(x):
    return np.exp(-9 * x * x)


def cr2x(cr):
    return (cr + 1) / 2 * (x8 - x0) + x0


def CR(m):
    return np.cos((2 * m - 1) * np.pi / (2 * n))


def diff_table(nodes, values):
    n = len(nodes)
    diff = np.zeros((n, n))
    diff[:, 0] = values
    for ii in range(1, n):
        for jj in range(n - ii):
            diff[jj, ii] = (diff[jj + 1, ii - 1] - diff[jj, ii - 1]) / (
                nodes[ii + jj] - nodes[jj]
            )
    return diff


def make_newton_inter(nodes, diff):
    n = len(nodes)

    def newton_inter(x):
        result = diff[0, 0]
        prod = 1
        for ii in range(1, n):
            prod *= x - nodes[ii - 1]
            result += diff[0, ii] * prod
        return result

    return newton_inter


xi = np.array([cr2x(CR(m)) for m in range(1, n + 1)])
fi = f(xi)
diff_cheb = diff_table(xi, fi)
cheb_interp = make_newton_inter(xi, diff_cheb)

b = np.linspace(x0, x8, n)
fb = f(b)
diff_eq = diff_table(b, fb)
b_interp = make_newton_inter(b, diff_eq)

a = np.linspace(x0, x8, 1000)
fa = f(a)
pa = np.array([cheb_interp(xx) for xx in a])
pb = np.array([b_interp(xx) for xx in a])


plt.figure(figsize=(8, 5))
plt.plot(a, fa, label="f(x)", linewidth=2)
plt.plot(a, pa, label="Chebyshev Interpolation", linestyle="--")
plt.plot(a, pb, label="Equal-spaced Interpolation", color="green", linestyle="--")
plt.scatter(xi, fi, color="red", label="Chebyshev point")
plt.scatter(b, f(b), color="green", label="Equal-spaced point")
plt.legend()
plt.savefig("images/Chebyshev Interpolation.png", dpi=300)
plt.show()

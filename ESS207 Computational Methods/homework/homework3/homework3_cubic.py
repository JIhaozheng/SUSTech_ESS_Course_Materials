import numpy as np


def cubic(f, x0, x8, N):
    n = N - 1
    xi = np.linspace(x0, x8, N)
    yi = f(xi)

    a = np.zeros(N)
    b = np.zeros(N)
    A = np.zeros(N)
    B = np.zeros(N)
    M = np.zeros(N)

    def h(ii):
        return xi[ii + 1] - xi[ii]

    for ii in range(1, n):
        a[ii] = h(ii) / (h(ii) + h(ii - 1))
        b[ii] = (
            6
            / (h(ii) + h(ii - 1))
            * ((yi[ii + 1] - yi[ii]) / h(ii) - (yi[ii] - yi[ii - 1]) / h(ii - 1))
        )

    A[1] = -a[1] / 2
    B[1] = b[1] / 2
    for ii in range(2, n - 2):
        A[ii] = -a[ii] / (2 + (1 - a[ii]) * A[ii - 1])
        B[ii] = (b[ii] - (1 - a[ii]) * B[ii - 1]) / (2 + (1 - a[ii]) * A[ii - 1])

    M[n - 1] = (b[n - 1] - (1 - a[n - 1]) * B[n - 2]) / (2 + (1 - a[n - 1]) * A[n - 2])

    for ii in range(n - 2, 0, -1):
        M[ii] = A[ii] * M[ii + 1] + B[ii]

    def cubic_spline(x):
        ii = np.searchsorted(xi, x) - 1
        y_out = (
            M[ii] * (xi[ii + 1] - x) ** 3 / (6 * h(ii))
            + M[ii + 1] * (x - xi[ii]) ** 3 / (6 * h(ii))
            + (yi[ii] - M[ii] * h(ii) ** 2 / 6) * (xi[ii + 1] - x) / h(ii)
            + (yi[ii + 1] - M[ii + 1] * h(ii) ** 2 / 6) * (x - xi[ii]) / h(ii)
        )
        return y_out

    return cubic_spline

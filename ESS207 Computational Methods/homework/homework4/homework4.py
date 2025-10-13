import numpy as np
import matplotlib.pyplot as plt
import test
import os

os.makedirs("images", exist_ok=True)


def f(x, y):
    return 20 * np.exp(-9 * (x**2 + y**2))


class Bilinear_Interpolation:
    def __init__(self, xi, yi, zi):
        self.xi = xi
        self.yi = yi
        self.zi = zi
        self.Nx = len(xi)
        self.Ny = len(yi)

    def bi_inter(self, x, y):
        x_idx = np.clip(np.searchsorted(self.xi, x) - 1, 0, self.Nx - 2)
        y_idx = np.clip(np.searchsorted(self.yi, y) - 1, 0, self.Ny - 2)
        n = (x - self.xi[x_idx]) / (self.xi[x_idx + 1] - self.xi[x_idx])
        u = (y - self.yi[y_idx]) / (self.yi[y_idx + 1] - self.yi[y_idx])
        z_pred = (1 - n) * (
            (1 - u) * self.zi[y_idx, x_idx] + u * self.zi[y_idx + 1, x_idx]
        ) + n * (
            (1 - u) * self.zi[y_idx, x_idx + 1] + u * self.zi[y_idx + 1, x_idx + 1]
        )
        return z_pred


class Cubic_Interpolation:
    def __init__(self, xi, yi, zi):
        self.xi = xi
        self.yi = yi
        self.zi = zi
        self.Nx = len(xi)
        self.Ny = len(yi)

    def _make_single_cubic(self, xi, yi, x):
        N = len(yi)
        a = np.zeros(N)
        b = np.zeros(N)
        A = np.zeros(N)
        B = np.zeros(N)
        M = np.zeros(N)
        h = np.zeros(N)
        for ii in range(N - 1):
            h[ii] = xi[ii + 1] - xi[ii]

        for ii in range(1, N - 1):
            a[ii] = h[ii] / (h[ii] + h[ii - 1])
            b[ii] = (
                6
                / (h[ii] + h[ii - 1])
                * ((yi[ii + 1] - yi[ii]) / h[ii] - (yi[ii] - yi[ii - 1]) / h[ii - 1])
            )
            A[ii] = -a[ii] / (2 + (1 - a[ii]) * A[ii - 1])
            B[ii] = (b[ii] - (1 - a[ii]) * B[ii - 1]) / (2 + (1 - a[ii]) * A[ii - 1])

        M[N - 2] = (b[N - 2] - (1 - a[N - 2]) * B[N - 3]) / (
            2 + (1 - a[N - 2]) * A[N - 3]
        )

        for ii in range(N - 3, 0, -1):
            M[ii] = A[ii] * M[ii + 1] + B[ii]

        ii = np.clip(np.searchsorted(xi, x) - 1, 0, len(xi) - 2)
        y_out = (
            M[ii] * (xi[ii + 1] - x) ** 3 / (6 * h[ii])
            + M[ii + 1] * (x - xi[ii]) ** 3 / (6 * h[ii])
            + (yi[ii] - M[ii] * h[ii] ** 2 / 6) * (xi[ii + 1] - x) / h[ii]
            + (yi[ii + 1] - M[ii + 1] * h[ii] ** 2 / 6) * (x - xi[ii]) / h[ii]
        )
        return y_out

    def cubic_inter(self, x, y):
        y_line = np.zeros(self.Nx)
        for ii in range(self.Nx):
            y_line[ii] = self._make_single_cubic(self.yi, self.zi[:, ii], y)
        pred = self._make_single_cubic(self.xi, y_line, x)
        return pred


def draw(f, Cubic_Interpolation, Bilinear_Interpolation):
    N_cubic = 7
    xi_cubic = np.linspace(-1, 1, N_cubic)
    yi_cubic = np.linspace(-1, 1, N_cubic)
    X_cubic, Y_cubic = np.meshgrid(xi_cubic, yi_cubic)
    zi_cubic = f(X_cubic, Y_cubic)
    Cubic = Cubic_Interpolation(xi_cubic, yi_cubic, zi_cubic)

    N_bilinear = 11
    xi_bilinear = np.linspace(-1, 1, N_bilinear)
    yi_bilinear = np.linspace(-1, 1, N_bilinear)
    X_bilinear, Y_bilinear = np.meshgrid(xi_bilinear, yi_bilinear)
    zi_bilinear = f(X_bilinear, Y_bilinear)
    Bilinear = Bilinear_Interpolation(xi_bilinear, yi_bilinear, zi_bilinear)

    x0, xn = -1, 1
    y0, yn = -1, 1
    x_test = np.linspace(x0, xn, 100)
    y_test = np.linspace(y0, yn, 100)
    x_t, y_t = np.meshgrid(x_test, y_test)
    Z_true = f(x_t, y_t)
    Z_pred_cubic = np.zeros_like(x_t)
    for ii in range(len(x_test)):
        for jj in range(len(y_test)):
            Z_pred_cubic[ii, jj] = Cubic.cubic_inter(x_test[ii], y_test[jj])
    abs_err_cubic = np.abs(Z_true - Z_pred_cubic)
    rel_err_cubic = abs_err_cubic / (np.abs(Z_true) + 1e-8)

    Z_pred_bilinear = np.zeros_like(x_t)
    for ii in range(len(x_test)):
        for jj in range(len(y_test)):
            Z_pred_bilinear[ii, jj] = Bilinear.bi_inter(x_test[ii], y_test[jj])
    abs_err_bilinear = np.abs(Z_true - Z_pred_bilinear)
    rel_err_bilinear = abs_err_bilinear / (np.abs(Z_true) + 1e-8)

    fig1, axs1 = plt.subplots(1, 4, figsize=(25, 5))
    im0 = axs1[0].contourf(x_t, y_t, Z_true, levels=60, cmap="viridis")
    axs1[0].scatter(
        X_cubic, Y_cubic, color="red", marker="o", s=50, label="Sample Points (7×7)"
    )
    axs1[0].set_title("True Surface (Cubic) with 7×7 Points", fontsize=15)
    axs1[0].set_aspect(1)
    axs1[0].legend(loc="upper right")
    plt.colorbar(im0, ax=axs1[0], shrink=0.85)

    im1 = axs1[1].contourf(x_t, y_t, Z_pred_cubic, levels=60, cmap="viridis")
    axs1[1].set_title("Cubic Spline Interpolation", fontsize=15)
    axs1[1].set_aspect(1)
    plt.colorbar(im1, ax=axs1[1], shrink=0.85)

    im2 = axs1[2].contourf(x_t, y_t, abs_err_cubic, levels=60, cmap="hot")
    axs1[2].set_title("Cubic Absolute Error", fontsize=15)
    axs1[2].set_aspect(1)
    plt.colorbar(im2, ax=axs1[2], shrink=0.85)

    im3 = axs1[3].contourf(x_t, y_t, rel_err_cubic, levels=60, cmap="coolwarm")
    axs1[3].set_title("Cubic Relative Error", fontsize=15)
    axs1[3].set_aspect(1)
    plt.colorbar(im3, ax=axs1[3], shrink=0.85)

    for ax in axs1:
        ax.set_xlabel("x", fontsize=12)
        ax.set_ylabel("y", fontsize=12)
    plt.tight_layout()
    fig1.savefig("images/cubic_surface.png", dpi=300, bbox_inches="tight")
    plt.show()

    fig2, axs2 = plt.subplots(1, 4, figsize=(25, 5))
    im0b = axs2[0].contourf(x_t, y_t, Z_true, levels=60, cmap="viridis")
    axs2[0].scatter(
        X_bilinear,
        Y_bilinear,
        color="red",
        marker="o",
        s=50,
        label="Sample Points (11×11)",
    )
    axs2[0].set_title("True Surface (Bilinear) with 11×11 Points", fontsize=15)
    axs2[0].set_aspect(1)
    axs2[0].legend(loc="upper right")
    plt.colorbar(im0b, ax=axs2[0], shrink=0.85)

    im1b = axs2[1].contourf(x_t, y_t, Z_pred_bilinear, levels=60, cmap="viridis")
    axs2[1].set_title("Bilinear Interpolation", fontsize=15)
    axs2[1].set_aspect(1)
    plt.colorbar(im1b, ax=axs2[1], shrink=0.85)

    im2b = axs2[2].contourf(x_t, y_t, abs_err_bilinear, levels=60, cmap="hot")
    axs2[2].set_title("Bilinear Absolute Error", fontsize=15)
    axs2[2].set_aspect(1)
    plt.colorbar(im2b, ax=axs2[2], shrink=0.85)

    im3b = axs2[3].contourf(x_t, y_t, rel_err_bilinear, levels=60, cmap="coolwarm")
    axs2[3].set_title("Bilinear Relative Error", fontsize=15)
    axs2[3].set_aspect(1)
    plt.colorbar(im3b, ax=axs2[3], shrink=0.85)

    for ax in axs2:
        ax.set_xlabel("x", fontsize=12)
        ax.set_ylabel("y", fontsize=12)
    plt.tight_layout()
    fig2.savefig("images/bilinear_surface.png", dpi=300, bbox_inches="tight")
    plt.show()

    fig3 = plt.figure(figsize=(9, 8))
    ax = fig3.add_subplot(111, projection="3d")
    surf_c = ax.plot_surface(x_t, y_t, abs_err_cubic, cmap="Reds", alpha=0.65)
    surf_b = ax.plot_surface(x_t, y_t, abs_err_bilinear, cmap="Blues", alpha=0.45)
    from matplotlib.patches import Patch

    legend_elements = [
        Patch(facecolor="red", alpha=0.5, label="Cubic"),
        Patch(facecolor="blue", alpha=0.5, label="Bilinear"),
    ]
    ax.legend(handles=legend_elements, loc="upper right")
    ax.set_xlabel("x", fontsize=13)
    ax.set_ylabel("y", fontsize=13)
    ax.set_zlabel("Absolute Error", fontsize=13)
    ax.set_title("3D Error Surface Comparison", fontsize=16)
    plt.tight_layout()
    fig3.savefig("images/3d_error_surface.png", dpi=300, bbox_inches="tight")
    plt.show()

    fig4 = plt.figure(figsize=(20, 4.5))
    plt.subplot(1, 4, 1)
    im = plt.contourf(x_t, y_t, Z_true, levels=60, cmap="viridis")
    plt.colorbar(im, fraction=0.046)
    plt.scatter(
        X_cubic,
        Y_cubic,
        color="white",
        s=28,
        edgecolors="k",
        label="Cubic 7×7",
        zorder=4,
    )
    plt.scatter(
        X_bilinear,
        Y_bilinear,
        color="yellow",
        s=10,
        edgecolors="k",
        label="Bilinear 11×11",
        zorder=4,
    )
    t = np.linspace(x0, xn, 100)
    plt.plot(t, t, "r--", lw=2, label="y=x")
    plt.plot(t, 0 * t, "g-.", lw=2, label="y=0")
    plt.plot(t, 0.5 * t, "b:", lw=2, label="y=0.5x")
    plt.legend()
    plt.title("Function & Lines", fontsize=14)
    plt.xlim(x0, xn)
    plt.ylim(y0, yn)
    plt.gca().set_aspect("equal", adjustable="box")

    plt.subplot(1, 4, 2)
    z_true1 = f(t, t)
    z_c1 = np.array([Cubic.cubic_inter(x, y) for x, y in zip(t, t)])
    z_b1 = np.array([Bilinear.bi_inter(x, y) for x, y in zip(t, t)])
    plt.plot(t, z_true1, "k-", lw=2, label="True")
    plt.plot(t, z_c1, "r--", lw=2, label="Cubic")
    plt.plot(t, z_b1, "b:", lw=2, label="Bilinear")
    plt.title("y=x", fontsize=14)
    plt.xlabel("x")
    plt.ylabel("f(x, x)")
    plt.xlim(x0, xn)
    plt.grid(True)
    plt.legend()

    plt.subplot(1, 4, 3)
    z_true2 = f(t, 0 * t)
    z_c2 = np.array([Cubic.cubic_inter(x, 0) for x in t])
    z_b2 = np.array([Bilinear.bi_inter(x, 0) for x in t])
    plt.plot(t, z_true2, "k-", lw=2, label="True")
    plt.plot(t, z_c2, "r--", lw=2, label="Cubic")
    plt.plot(t, z_b2, "b:", lw=2, label="Bilinear")
    plt.title("y=0", fontsize=14)
    plt.xlabel("x")
    plt.ylabel("f(x, 0)")
    plt.xlim(x0, xn)
    plt.grid(True)
    plt.legend()

    plt.subplot(1, 4, 4)
    z_true3 = f(t, 0.5 * t)
    z_c3 = np.array([Cubic.cubic_inter(x, 0.5 * x) for x in t])
    z_b3 = np.array([Bilinear.bi_inter(x, 0.5 * x) for x in t])
    plt.plot(t, z_true3, "k-", lw=2, label="True")
    plt.plot(t, z_c3, "r--", lw=2, label="Cubic")
    plt.plot(t, z_b3, "b:", lw=2, label="Bilinear")
    plt.title("y=0.5x", fontsize=14)
    plt.xlabel("x")
    plt.ylabel("f(x, 0.5x)")
    plt.xlim(x0, xn)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    fig4.savefig("images/line_compare.png", dpi=300, bbox_inches="tight")
    plt.show()


draw(f, Cubic_Interpolation, Bilinear_Interpolation)

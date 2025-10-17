import numpy as np
import matplotlib.pyplot as plt
import os

os.makedirs("images", exist_ok=True)


def f(x):
    return x**3


# fitted points
xn = np.linspace(0, 2, 10)
yn_clean = f(xn)
noise = np.random.normal(0, 0.2, size=yn_clean.shape)
yn_noise = yn_clean + noise

# dimension of basis
N = 10


# def basis functions
def polynomial(n):
    return lambda x: x**n


basis = [polynomial(k) for k in range(N)]


class Orthogonal_basis_fit:
    def __init__(self):
        self.N = None
        self.xn = None
        self.yn = None
        self.S = []
        self.lam = None
        self.miu = None
        self.b = None

    def get_fit_parameters(self, xn, yn, basis):
        self.N = len(basis)
        self.xn = xn
        self.yn = yn
        self.S = [s for s in basis]
        self.lam = np.zeros(self.N)
        self.miu = np.zeros(self.N)

    def cal_lam(self, ii):
        numerator = np.dot(self.xn * self.S[ii - 1](self.xn), self.S[ii - 1](self.xn))
        denominator = np.dot(self.S[ii - 1](self.xn), self.S[ii - 1](self.xn))
        return numerator / denominator

    def cal_miu(self, ii):
        if ii == 1:
            return 0.0
        else:
            numerator = np.dot(self.S[ii - 1](self.xn), self.S[ii - 1](self.xn))
            denominator = np.dot(self.S[ii - 2](self.xn), self.S[ii - 2](self.xn))
        return numerator / denominator

    def cal_b(self, ii):
        numerator = np.dot(self.S[ii](self.xn), self.yn)
        denominator = np.dot(self.S[ii](self.xn), self.S[ii](self.xn))
        return numerator / denominator

    def cal_S(self, ii):
        lam, miu, S1, S2 = self.lam[ii], self.miu[ii], self.S[ii - 1], self.S[ii - 2]

        def S_ii(x):
            return (x - lam) * S1(x) - miu * S2(x)

        return S_ii

    def cal_ortho_basis(self):

        for ii in range(1, self.N):
            self.lam[ii] = self.cal_lam(ii)
            self.miu[ii] = self.cal_miu(ii)
            self.S[ii] = self.cal_S(ii)
        self.b = np.array([self.cal_b(ii) for ii in range(self.N)])
        return self.S

    def predict(self, x):
        y = np.zeros_like(x)
        for ii in range(self.N):
            y += self.b[ii] * self.S[ii](x)
        return y


# draw the diagrams
def drawN():
    x_lin = np.linspace(0, 2, 1000)
    for n in range(1, N + 1):
        Least_square = Orthogonal_basis_fit()
        Least_square.get_fit_parameters(xn, yn_noise, basis[:n])
        S = Least_square.cal_ortho_basis()
        y_pred = Least_square.predict(x_lin)

        plt.figure(figsize=(6, 4))
        plt.plot(x_lin, f(x_lin), "k--", label="$f(x)=x^3$ (true)")
        plt.scatter(xn, yn_noise, c="r", marker="o", label="Noisy data")
        plt.plot(x_lin, y_pred, label=f"Fit (mode={n})", linewidth=2)
        plt.xlabel("$x$")
        plt.ylabel("$y$")
        plt.title(f"Orthogonal Polynomial Fit: Degree {n}")
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"images/fit_{n}.png")
        plt.close()


def draw():
    x_lin = np.linspace(0, 2, 1000)
    deg_list = [1, 3, 5, 10]  # degrees to show, can adjust as needed
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    axes = axes.flatten()
    for idx, n in enumerate(deg_list):
        Least_square = Orthogonal_basis_fit()
        Least_square.get_fit_parameters(xn, yn_noise, basis[:n])
        S = Least_square.cal_ortho_basis()
        y_pred = Least_square.predict(x_lin)

        ax = axes[idx]
        ax.plot(x_lin, f(x_lin), "k--", label="$f(x)=x^3$ (true)")
        ax.scatter(xn, yn_noise, c="r", marker="o", label="Noisy data")
        ax.plot(x_lin, y_pred, label=f"Fit (degree={n})", linewidth=2)
        ax.set_xlabel("$x$")
        ax.set_ylabel("$y$")
        ax.set_title(f"Degree {n}")
        ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig("images/fit_grid.png")
    plt.show()
    plt.close()


# drawN()
draw()

import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.special import roots_legendre
from scipy.special import ellipk

os.makedirs("images", exist_ok=True)

def f(x):
    return np.exp(3*x)*np.cos(x)
def g(x):
    return 1/np.sqrt(16-x**2)
def wg(x):
    return 1/(np.sqrt(1-x**2)*np.sqrt(16-x**2))

class GaussQuadrature:
    def __init__(self,f,n,a,b,weight="1"):
        self.f=f
        self.n=n
        self.a=a
        self.b=b
        self.weight=weight
    
    def cal_quad(self):
        if self.weight=="1":
            nodes,weights=roots_legendre(self.n)
            xn=(self.b-self.a)/2*nodes+(self.b+self.a)/2
            return (self.b-self.a)/2*sum(self.f(xn)*weights)
        if self.weight=="cheby":
            weights=np.pi/self.n
            nodes=np.array([np.cos((2*jj+1)/(2*self.n)*np.pi) for jj in range(self.n)])
            xn=(self.b-self.a)/2*nodes+(self.b+self.a)/2
            return (self.b-self.a)/2*sum(self.f(xn)*weights)
        

def If(x):
    return 1/10*(np.sin(x)+3*np.cos(x))*np.exp(3*x)

GQ = GaussQuadrature(f, 2, 0, np.pi)
Q_f = GQ.cal_quad()
I_f = If(np.pi) - If(0)
abs_err_f = abs(Q_f - I_f)
rel_err_f = abs_err_f / abs(I_f)
print("【Gauss-Legendre on [0, pi]】 n_nodes=2")
print("True value          :", I_f)
print("Numerical value     :", Q_f)
print("Absolute error      :", abs_err_f)
print("Relative error      :", rel_err_f)
print()
GQ = GaussQuadrature(f, 3, 0, np.pi)
Q_f = GQ.cal_quad()
I_f = If(np.pi) - If(0)
abs_err_f = abs(Q_f - I_f)
rel_err_f = abs_err_f / abs(I_f)
print("【Gauss-Legendre on [0, pi]】n_nodes=3")
print("True value          :", I_f)
print("Numerical value     :", Q_f)
print("Absolute error      :", abs_err_f)
print("Relative error      :", rel_err_f)
print()
GQ = GaussQuadrature(f, 4, 0, np.pi)
Q_f = GQ.cal_quad()
I_f = If(np.pi) - If(0)
abs_err_f = abs(Q_f - I_f)
rel_err_f = abs_err_f / abs(I_f)
print("【Gauss-Legendre on [0, pi]】n_nodes=4")
print("True value          :", I_f)
print("Numerical value     :", Q_f)
print("Absolute error      :", abs_err_f)
print("Relative error      :", rel_err_f)
print()


k = 1/4
m = k ** 2
K_k = ellipk(m)
I_wg_true = 0.5 * K_k

GQ = GaussQuadrature(g, 2, -1, 1, "cheby")
Q_g = GQ.cal_quad()
abs_err_g = abs(Q_g - I_wg_true)
rel_err_g = abs_err_g / abs(I_wg_true)
print("【Gauss-Chebyshev with weight on [-1, 1]】")
print("True value          :", I_wg_true)
print("Numerical value     :", Q_g)
print("Absolute error      :", abs_err_g)
print("Relative error      :", rel_err_g)
print()

GQ = GaussQuadrature(wg, 2, -1, 1)
Q_wg = GQ.cal_quad()
abs_err_wg = abs(Q_wg - I_wg_true)
rel_err_wg = abs_err_wg / abs(I_wg_true)
print("【Gauss-Legendre on [-1, 1] for wg(x)】")
print("True value          :", I_wg_true)
print("Numerical value     :", Q_wg)
print("Absolute error      :", abs_err_wg)
print("Relative error      :", rel_err_wg)


import numpy as np
import matplotlib.pyplot as plt
import os
os.makedirs("images", exist_ok=True)

def f(x):
    return np.exp(x)

class Diff:
    def __init__(self,f,x0,h,e=1e-3):
        self.f=f
        self.x0=x0
        self.h=h
        xn=np.array([self.x0-n*self.h for n in range(2,-3,-1)])
        yn=self.f(xn)
        print(f"The true function values at sample points xn:\n{yn}")
        self.yn=np.round(yn/e)*e
        print(f"The function values with finite precision (rounded to nearest {e}):\n{self.yn}")
    def cal_diff(self,mode="forward"):
        diff=(self.yn[3]-self.yn[2])/(self.h)
        if mode=="backward":
            diff=(self.yn[2]-self.yn[1])/(self.h)
        elif mode=="centered":
            diff=(self.yn[3]-self.yn[1])/(2*self.h)
        elif mode=="five-point":
            diff=(self.yn[0]-8*self.yn[1]+8*self.yn[3]-self.yn[4])/(12*self.h)
        return diff
def comparsion_result(x0,h):
    D=Diff(f,x0,h)
    d=np.zeros(4)
    t=f(x0)
    d[0]=D.cal_diff("forward")
    d[1]=D.cal_diff("backward")
    d[2]=D.cal_diff("centered")
    d[3]=D.cal_diff("five-point")
    print(f"The finite-difference results at x0={x0} and difference interval h={h}:")
    print(f"Forward difference:  {d[0]:.5f} with relative error {np.abs(d[0]-t)/t}")
    print(f"Backward difference  {d[1]:.5f} with relative error {np.abs(d[1]-t)/t}")
    print(f"Centered difference:  {d[2]:.5f} with relative error {np.abs(d[2]-t)/t}")
    print(f"Five-point difference  {d[3]:.5f} with relative error {np.abs(d[3]-t)/t}")
    print("\n\n")

x0=1.7
h=0.5
comparsion_result(x0,h)
h=0.2
comparsion_result(x0,h)
h=0.1
comparsion_result(x0,h)
h=0.005
comparsion_result(x0,h)
h=0.001
comparsion_result(x0,h)

def draw_all_methods():
    x0 = 1.7
    t = f(x0)
    h = np.logspace(-3, -0.5, 200)  # h from 0.001 to ~0.316 on log scale
    methods = ["forward", "backward", "centered", "five-point"]
    labels = [
        "Forward Difference",
        "Backward Difference",
        "Centered Difference",
        "Five-point Difference"
    ]
    os.makedirs("images", exist_ok=True)

    plt.figure(figsize=(12, 5))
    
    # Left: Relative Error
    ax1 = plt.subplot(1,2,1)
    for method, label in zip(methods, labels):
        D = Diff(f, x0, h)
        d = D.cal_diff(method)
        rel_err = np.abs(d - t) / np.abs(t)
        plt.plot(h, rel_err, label=label)
    plt.xscale('log')
    plt.xlabel("Step Size $h$", fontsize=12)
    plt.ylabel("Relative Error", fontsize=12)
    plt.title("Relative Error of Finite Difference Methods", fontsize=13)
    plt.legend(fontsize=9)
    plt.grid(True, linestyle='--', alpha=0.7)

    # Right: Absolute Error
    ax2 = plt.subplot(1,2,2)
    for method, label in zip(methods, labels):
        D = Diff(f, x0, h)
        d = D.cal_diff(method)
        abs_err = np.abs(d - t)
        plt.plot(h, abs_err, label=label)
    plt.xscale('log')
    plt.xlabel("Step Size $h$", fontsize=12)
    plt.ylabel("Absolute Error", fontsize=12)
    plt.title("Absolute Error of Finite Difference Methods", fontsize=13)
    plt.legend(fontsize=9)
    plt.grid(True, linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.savefig(os.path.join("images", "diff_all_methods_subplots_logspace.png"))
    plt.show()

draw_all_methods()
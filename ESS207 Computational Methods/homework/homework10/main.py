import numpy as np
import matplotlib.pyplot as plt
import os
os.makedirs("images", exist_ok=True)
from bisection import BisectionMethod
from fixed_point import FixedPointMethod
from Newton_Secant import Newton_method, Secant_method

def bisection_method():
    def f(x):
        return x**3-np.exp(-x)
    S=BisectionMethod(f,0.02)
    r,n =S.bisection(-1,1)
    print("############# Bisection Method #############")
    for count in range(len(n)):
        print(f"Iteration {count}: x = {n[count]:.6f}, f(x)={f(n[count]):.6e}")
    print(f"Final fixed point:{r[0]:.6f} with value {f(r[0]):.6e}")
    x_lin=np.linspace(-1,1,100)
    plt.plot(x_lin,f(x_lin),label="f(x)")
    plt.scatter(n,f(n),label="Test points",color="orange")
    plt.scatter(r,f(r),label="Root point",color="red")
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.legend()
    plt.grid()
    plt.title("Bisection Method")
    plt.savefig(os.path.join("images","Bisection_method.png"),dpi=300)
    plt.show()


def fixed_point_method():
    def f(x):
        return np.exp(-x/3)-x
    def f1(x):
        return -1/3*np.exp(-x/3)-1
    x0,tol=100,1e-2
    print("\n############# Fixed-point Method #############")
    S=FixedPointMethod(f,f1,x0,tol)
    r,n=S.fixed_point()
    print(f"Final fixed point:{r:.6f} with value {f(r):.6e}")

    x_lin=np.linspace(-1,1,100)
    plt.plot(x_lin,f(x_lin),label="f(x)")
    plt.scatter(n,f(n),label="Test points",color="orange")
    plt.scatter(r,f(r),label="Root point",color="red")
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.legend()
    plt.grid()
    plt.title("Fixed-point Method")
    plt.savefig(os.path.join("images","Fixed_point_method.png"),dpi=300)
    plt.show()

def newton_method():
    def f(x):
        return x**3-np.exp(-x)
    def f1(x):
        return 3*x**2+np.exp(-x)
    
    x0,tol=20,1e-2
    print("\n############# Newton Method #############")
    r,n=Newton_method(f,f1,x0,tol)
    x_lin=np.linspace(-1,x0,100)
    plt.plot(x_lin,f(x_lin),label="f(x)")
    plt.scatter(n,f(n),label="Test points",color="orange")
    plt.scatter(r,f(r),label="Root point",color="red")
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.legend()
    plt.grid()
    plt.title("Newton Method")
    plt.savefig(os.path.join("images","Newton_method.png"),dpi=300)
    plt.show()

def secant_method():
    def f(x):
        return x**3-np.exp(-x)
    x0,x1,tol=10,20,1e-2
    print("\n############# Secant Method #############")
    r,n=Secant_method(f,x0,x1,tol)
    x_lin=np.linspace(-1,20,100)
    plt.plot(x_lin,f(x_lin),label="f(x)")
    plt.scatter(n,f(n),label="Test points",color="orange")
    plt.scatter(r,f(r),label="Root point",color="red")
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.legend()
    plt.grid()
    plt.title("Secant Method")
    plt.savefig(os.path.join("images","Secant_method.png"),dpi=300)
    plt.show()

bisection_method()
fixed_point_method()
newton_method()
secant_method()
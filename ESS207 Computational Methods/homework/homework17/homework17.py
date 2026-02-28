import numpy  as np
import matplotlib.pyplot as plt
import time
import os
os.makedirs("images", exist_ok=True)
def f(x,y):
    return y-2*x/y

def y_true(x):
    return np.sqrt(2*x + 1)

class SloveODE():
    def __init__(self,f,I,y0):
        self.f=f
        self.x0=I[0]
        self.xe=I[1]
        self.y0=y0
        self.y_values=None
    
    def RungeKutta4(self,h):
        N=int(np.ceil((self.xe-self.x0)/h))
        y_values=np.zeros(N+1)
        y_values[0]=self.y0
        x_values=np.zeros(N+1)
        x_values[0]=self.x0
        x,y=self.x0,self.y0
        t_s=time.time()
        for ii in range(1,N+1):
            y=y_values[ii-1]
            if x+h>self.xe:
                h=self.xe-x
            k1=self.f(x,y)
            k2=self.f(x+h/2,y+h*k1/2)
            k3=self.f(x+h/2,y+h*k2/2)
            k4=self.f(x+h,y+h*k3)
            y_values[ii]=y+h/6*(k1+2*k2+2*k3+k4)
            x_values[ii]=x_values[ii-1]+h
            x+=h
        t_e=time.time()
        return y_values,x_values,t_e-t_s
    
    def ImprovedEuler(self,h):
        N=int(np.ceil((self.xe-self.x0)/h))
        y_values=np.zeros(N+1)
        y_values[0]=self.y0
        x_values=np.zeros(N+1)
        x_values[0]=self.x0
        x,y=self.x0,self.y0
        t_s=time.time()
        for ii in range(1,N+1):
            y=y_values[ii-1]
            if x+h>self.xe:
                h=self.xe-x
            k1=self.f(x,y)
            k2=self.f(x+h,y+h*k1)
            y_values[ii]=y+h/2*(k1+k2)
            x_values[ii]=x_values[ii-1]+h
            x+=h
        t_e=time.time()
        return y_values,x_values,t_e-t_s

I = np.array([0, 1.0])
y0 = 1
S = SloveODE(f, I, y0)

y_rk, x_rk, t_rk = S.RungeKutta4(0.2)
y_ie, x_ie, t_ie = S.ImprovedEuler(0.1)

err_rk = np.abs(y_true(x_rk) - y_rk)
err_ie = np.abs(y_true(x_ie) - y_ie)

print("====== Method Accuracy Comparison ======")
print(f"Runge-Kutta 4: time = {t_rk:.12e} s")
print(f"   Max error = {np.max(err_rk):.6e}")
print(f"   Mean error = {np.mean(err_rk):.6e}\n")

print(f"Improved Euler: time = {t_ie:.12e} s")
print(f"   Max error = {np.max(err_ie):.6e}")
print(f"   Mean error = {np.mean(err_ie):.6e}\n")

plt.figure(figsize=(8,5))
plt.plot(x_rk, err_rk, 'o-', label='RK4 error')
plt.plot(x_ie, err_ie, 'x-', label='Improved Euler error')

plt.xlabel("x")
plt.ylabel("Absolute Error |y - y_true|")
plt.title("Error Comparison of RK4 and Improved Euler")
plt.grid(True)
plt.legend()
plt.savefig('images/Error Comparison of RK4 and Improved Euler.png',dpi=300)
plt.show()


x_lin = np.linspace(I[0], I[1], 300)
plt.plot(x_lin, y_true(x_lin), 'k--', label="True solution", linewidth=2)

plt.scatter(x_ie, y_ie, color='b', s=25, label="Improved Euler")
plt.scatter(x_rk, y_rk, color='r', s=25, label="Runge–Kutta 4")


plt.xlabel("x")
plt.ylabel("y")
plt.title("Comparison: True Solution vs RK4 vs Improved Euler")
plt.grid(True)
plt.legend()
plt.savefig('images/Comparison True Solution vs RK4 vs Improved Euler.png',dpi=300)
plt.show()




import numpy as np
import matplotlib.pyplot as plt
import os

os.makedirs("images",exist_ok=True)


def f(x):
    return np.exp(3*x)*np.cos(x)
def I_f(x):
    return 1/10*(np.sin(x)+3*np.cos(x))*np.exp(3*x)


class Quadrature:
    def __init__(self,f,x0,xn,tol=1e-5):
        self.f=f
        self.x0=x0
        self.xn=xn
        self.tol=1e-5
        self.points=[]

    def CompositeSimpsonRule(self,n):
        x=np.linspace(self.x0,self.xn,n+1)
        h=(self.xn-self.x0)/n
        fn=self.f(x)
        for ii in range(1,n):
            if ii%2==0:
                fn[ii]=2*fn[ii]
            else:
                fn[ii]=4*fn[ii]
        I=h/3*np.sum(fn)
        return I
    
    def _SimpsonRule(self,a,b):
        m = (a + b) / 2
        self.points.extend([a, m, b])
        return (b-a)/6*(self.f(a)+4*self.f((a+b)/2)+self.f(b))
    
    def _AdaptiveQuad(self,x0,xn):
        S=self._SimpsonRule(x0,xn)
        m=(x0+xn)/2
        L=self._SimpsonRule(x0,m)
        R=self._SimpsonRule(m,xn)
        self.points.append(m)
        if abs(L+R-S)<self.tol:
            return L+R
        else:
            L=self._AdaptiveQuad(x0,m)
            R=self._AdaptiveQuad(m,xn)
            return L+R
        
    def AdaptiveQuad(self,tol):
        self.tol=tol
        self.points.append(self.x0)
        self.points.append(self.xn)
        return self._AdaptiveQuad(self.x0,self.xn)
    
x0=0
n=4
xn=np.pi
Interval=Quadrature(f,x0,xn)

SimpsonRule4 = Interval.CompositeSimpsonRule(n)
Adaptive_value = Interval.AdaptiveQuad(1e-5)

True_value = I_f(np.pi) - I_f(0)
abs_err = [SimpsonRule4 - True_value, Adaptive_value - True_value]
rel_err = [(SimpsonRule4 - True_value) / True_value, (Adaptive_value - True_value) / True_value]


points = np.array(sorted(set(Interval.points)))
points_density=np.array([1/(points[ii+1]-points[ii]) for ii in range(points.shape[0]-1)])

print(f"Simpson(4 intervals) integral value: {SimpsonRule4}")
print(f"Adaptive Simpson integral value: {Adaptive_value}")
print(f"True value: {True_value}")
print(f"Absolute errors: Simpson: {abs_err[0]}  Adaptive: {abs_err[1]}")
print(f"Relative errors: Simpson: {rel_err[0]:.2e}  Adaptive: {rel_err[1]:.2e}")


print(f"Number of adaptive sampling points: {points.shape[0]}")
plt.figure(figsize=(8,4))
plt.scatter(points, f(points), color='lightcoral', s=20, alpha=0.6, label='Sampling Points')
plt.plot(points, f(points), 'k',linewidth=2.5, label='$f(x)$')
plt.xlabel("x")
plt.ylabel("f(x)")
plt.title("Distribution of Adaptive Simpson Sampling Points")
plt.legend()
plt.tight_layout()
plt.savefig('images/adaptive_simpson_sampling_points.png', dpi=300)
plt.show()

plt.plot(points[1:],points_density)
plt.xlabel("x")
plt.ylabel("local point density (1/Δx)")
plt.title("Sampling Point Density Curve")
plt.savefig('images/Sampling_Point_Density_Curve.png',dpi=300)
plt.show()
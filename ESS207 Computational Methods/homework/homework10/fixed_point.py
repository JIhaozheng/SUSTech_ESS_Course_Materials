import numpy as np
import matplotlib.pyplot as plt

class FixedPointMethod():
    def __init__(self,f,f1,x0,tol):
        self.f=f
        self.f1=f1
        self.x0=x0
        self.tol=tol   
        self.points=[] 

    def _fixed_point(self,x0,count=0):
        print(f"Interation {count}:x = {x0:.6f}, f(x0)={self.f(x0):.6e}")
        if abs((self.f1(x0)+1))>1:
            raise ValueError("enter the correct condition to satisfy the lipschitz condition")
        x_new=x0+self.f(x0)
        self.points.append(x_new)
        if abs(x_new-x0)<self.tol:
            return x_new
        else:
            return self._fixed_point(x_new,count+1)
    
    def fixed_point(self):
        x0=self.x0
        self.points=[]
        return self._fixed_point(x0),np.array(sorted(self.points))

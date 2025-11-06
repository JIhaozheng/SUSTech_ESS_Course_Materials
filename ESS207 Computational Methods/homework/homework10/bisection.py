import numpy as np
import matplotlib.pyplot as plt

class BisectionMethod:
    def __init__(self,f,tol):
        self.f=f
        self.tol=tol
        self.points=[]
        self.roots=[]

    def _bi(self,a,b):
        m=(a+b)/2
        self.points.append(m)
        if (b-a)<2*self.tol:
            if self.f(a)*self.f(b)<0:
                self.roots.append(m)
                return m
            else:
                raise ValueError("please change the interval")
        else:
            if abs(self.f(m))==0:
                self.roots.append(m)
                return m
            elif self.f(a)*self.f(m)<0:
                return self._bi(a,m)
            elif self.f(m)*self.f(b)<0:
                return self._bi(m,b)
            else:
                raise ValueError("please change the interval")
            
    def bisection(self,a,b):
        self.points=[]
        self.roots=[]
        if self.f(a)==0:
            self.roots.append(a)
        if self.f(b)==0:
            self.roots.append(b)
        if self.f(a)*self.f(b)<0:
            self._bi(a,b)
            return np.array(sorted(self.roots)), np.array(self.points)
        else:
            raise ValueError("please change the interval")
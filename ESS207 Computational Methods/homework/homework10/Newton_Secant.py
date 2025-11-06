import numpy as np
import matplotlib.pyplot as plt

def Newton_method(f,f1,x0,tol,points=None,count=0):
    if points is None:
        points = []
    points.append(x0)
    print(f"Interation {count}:x = {x0:.6f}, f(x)={f(x0):.6e}")
    
    x_new=x0-f(x0)/f1(x0)
    if abs(x_new-x0)<tol:
        points.append(x_new)
        print(f"Final fixed point: {x_new:.6f}, f(x)={f(x_new):.6e}")
        return x_new,np.array(points)
    else:
        return Newton_method(f,f1,x_new,tol,points,count+1)
    
def Secant_method(f,x0,x1,tol,points=None,count=0):
    if points is None:
        points=[]
    points.append(x1)
    print(f"Interation {count}:x = {x0:.6f}, f(x)={f(x0):.6e}")

    x_new=x1-(x1-x0)*f(x1)/(f(x1)-f(x0))
    if abs(x_new-x1)<tol:
        points.append(x_new)
        print(f"Final fixed point: {x_new:.6f}, f(x)={f(x_new):.6e}")
        return x_new,np.array(points)
    else:
        return Secant_method(f,x1,x_new,tol,points,count+1)
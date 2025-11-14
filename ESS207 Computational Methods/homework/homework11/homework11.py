import numpy as np
import matplotlib.pyplot as plt
import os
os.makedirs("images", exist_ok=True)

def f(x,y):
    return x+1-x**2-y**2/4

def new_f(x,y):
    return np.sqrt(1-y**2/4)

def g(x,y):
    return np.sin(x-y)-x

def f0(x,y):
    return 4*x**2+y**2-4
def g0(x,y):
    return x+y-np.sin(x-y)

def fp(f,g,x,y,mode=0,count=0,tol=1e-6):
    if mode==0:
        x_new=f(x,y)
        y_new=g(x,y)
    else:
        x_new=f(x,y)
        y_new=g(x_new,y)
    d=(x-x_new)**2+(y-y_new)**2
    if count>=300:
        print(f"Reached maximum iterations ({count}). Returning current approximation.")
        return x_new, y_new
    elif d<tol**2:
        print(f"Converged in {count + 1} iterations.")
        return x_new, y_new
    else:
        return fp(f,g,x_new,y_new,mode,count+1)
    
def fixed_point(f,g,x0,y0,mode=0):
    return fp(f,g,x0,y0,mode)

guess=np.array([1.0,0.0])

print("###### Original fixed point method ######\n")
print("--- Without appropriate F(x,y) ---")
root=fixed_point(f,g,guess[0],guess[1],0)

print(f"\nFinal approximated root: ({root[0]:.9f}, {root[1]:.9f})")
    
print(f"\n--- Verification using original system (f0(x,y)=0, g0(x,y)=0) ---")
f_root = f0(root[0], root[1])
g_root = g0(root[0], root[1])
print(f"f0: {f_root:.8e}")
print(f"g0: {g_root:.8e}")

print("--- With appropriate F(x,y) ---")
root1=fixed_point(new_f,g,guess[0],guess[1],0)
print(f"\nFinal approximated root: ({root1[0]:.9f}, {root1[1]:.9f})")
    
print(f"\n--- Verification using original system (f0(x,y)=0, g0(x,y)=0) ---")
f_root = f0(root1[0], root1[1])
g_root = g0(root1[0], root1[1])
print(f"f0: {f_root:.8e}")
print(f"g0: {g_root:.8e}")



print("\n\n###### Improved fixed point method ######\n")
print("--- Without appropriate F(x,y) ---")
root=fixed_point(f,g,guess[0],guess[1],1)
print(f"\nFinal approximated root: ({root[0]:.9f}, {root[1]:.9f})")
    
print(f"\n--- Verification using original system (f0(x,y)=0, g0(x,y)=0) ---")
f_root = f0(root[0], root[1])
g_root = g0(root[0], root[1])
print(f"f0: {f_root:.8e}")
print(f"g0: {g_root:.8e}")

print("--- With appropriate F(x,y) ---")
root1=fixed_point(new_f,g,guess[0],guess[1],1)
print(f"\nFinal approximated root: ({root1[0]:.9f}, {root1[1]:.9f})")
    
print(f"\n--- Verification using original system (f0(x,y)=0, g0(x,y)=0) ---")
f_root = f0(root1[0], root1[1])
g_root = g0(root1[0], root1[1])
print(f"f0: {f_root:.8e}")
print(f"g0: {g_root:.8e}")


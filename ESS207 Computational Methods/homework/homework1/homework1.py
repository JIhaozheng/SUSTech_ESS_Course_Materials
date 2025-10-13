import numpy as np
import matplotlib.pyplot as plt
import os

os.makedirs('images',exist_ok = True)

def f(x):
    return np.exp(5*x)

def f1(x):
    return 5*np.exp(5*x)
def f2(x):
    return 25*np.exp(5*x)

x0 = 1
h = np.logspace(-16,-1,200)
f_approx = (f(x0+h)-f(x0)) / h
f_real = f1(x0)

print(np.finfo(float).eps)
E_err = np.finfo(float).eps * f(x0) / h    # rounding error
T_err = 0.5 * f2(x0) * h                   # truncation error
eC = E_err + T_err                         # theoretical error
rel_eC = eC / abs(f_real)                  # relative theoretical error
abs_err = np.abs(f_approx - f_real)        # absolute calculation error
rel_err = abs_err / abs(f_real)            # relative calculation error
idx_min = np.argmin(eC)

# Theoretical error minimum
idx_min = np.argmin(eC)
h_min = h[idx_min]
print(f"The minimum theoretical combined error eC = {eC[idx_min]:.3e} occurs at h = {h_min:.3e}")
# Actual (computed) error minimum
idx_abs = np.argmin(abs_err)
h_min_abs = h[idx_abs]
print(f"The minimum computational absolute error abs_err = {abs_err[idx_abs]:.3e} occurs at h = {h_min_abs:.3e}")

plt.figure()
plt.loglog(h,eC,'k-',linewidth = 1.7, label ='Theoretical Combined Error')
plt.loglog(h,abs_err,'ro',markersize=4, markerfacecolor='r', label ='Actual Absolute Error')
plt.xlabel('h(step stze)')
plt.ylabel('Absolute error')
plt.title('Numerical Derivate Error for f(x)=exp(5x) at x0 = 1')
plt.legend()
plt.grid(True,which='both',ls="--")
plt.savefig('images/Numerical Derivate Error.png', dpi =300 , bbox_inches='tight')
plt.show()


plt.figure()
plt.loglog(h,rel_eC,'k-',linewidth = 1.7, label ='Theoretical Combined Error')
plt.loglog(h,rel_err,'ro',markersize=4, markerfacecolor='r', label ='Actual Relative Error')
plt.xlabel('h(step stze)')
plt.ylabel('Relative error')
plt.title('Relative Numerical Derivate Error for f(x)=exp(5x) at x0 = 1')
plt.legend()
plt.grid(True,which='both',ls="--")
plt.savefig('images/Relative Numerical Derivate Error.png', dpi =300 , bbox_inches='tight')
plt.show()

plt.figure()
plt.loglog(h,E_err,'k-',linewidth = 1.7, label ='Theoretical Rounding Error')
plt.loglog(h,T_err,'r-',linewidth = 1.7,label ='Theoretical Truncation Error')
plt.loglog(h,eC,'b-',linewidth = 1.7, label ='Theoretical Combined Error')
plt.xlabel('h(step stze)')
plt.ylabel('absolute error')
plt.title('Rounding and truncation Error for f(x)=exp(5x) at x0 = 1')
plt.legend()
plt.grid(True,which='both',ls="--")
plt.savefig('images/Rounding and truncation Error.png', dpi =300 , bbox_inches='tight')
plt.show()
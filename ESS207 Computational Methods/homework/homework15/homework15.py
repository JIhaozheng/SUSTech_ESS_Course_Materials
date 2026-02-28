import numpy as np
import time
class EigenvalueAlgorithms():
    def __init__(self,A,tol=1e-3,N=10):
        self.A=A
        self.tol=tol
        self.N=N
        self.shift=None

    def _LUDecomposition(self,A):
        n=A.shape[0]
        L=np.identity(n)
        U=np.zeros_like(A)

        for ii in range(n):
            for jj in range(ii,n):
                U[ii,jj]=A[ii,jj]
                for kk in range(ii):
                    U[ii,jj]-=L[ii,kk]*U[kk,jj]
                
            for jj in range(ii+1,n):
                L[jj,ii]=A[jj,ii]
                for kk in range(ii):
                    L[jj, ii] -= L[jj, kk] * U[kk, ii]
                L[jj, ii] /= U[ii, ii]                    
        return L,U
    
    def _solvex(self,b,L,U):
        n=len(b)
        y=np.zeros(n)
        x=np.zeros(n)
        for ii in range(n):
            y[ii] = b[ii]
            for jj in range(ii):
                y[ii] -= L[ii, jj] * y[jj]

        for ii in range(n-1, -1, -1):
            x[ii] = y[ii]
            for jj in range(ii+1, n):
                x[ii] -= U[ii, jj] * x[jj]
            x[ii] /= U[ii, ii]
        
        return x

    def _InverseShift(self,x0):
        A=self.A.copy()
        for ii in range(A.shape[0]):
            A[ii,ii]-=self.shift
        L,U=self._LUDecomposition(A)
        x=x0/np.linalg.norm(x0)
        start=time.time()
        for ii in range(self.N):
            x_new=self._solvex(x,L,U)
            miu=np.linalg.norm(x_new)/np.linalg.norm(x)
            x_new=x_new/np.linalg.norm(x_new)
            if self.mode==1:
                y_new=self.A.dot(x_new)
                rayleigh=np.dot(x_new,y_new)/np.dot(x_new,x_new)
                x_new=y_new/miu

            if np.linalg.norm(x_new-x)<self.tol:
                end=time.time()
                if self.mode == 0:
                    print(f"Method: Inverse Iteration with Shift (σ = {self.shift})")
                    print(f"Iterations: {ii}")
                    print(f"Runtime: {end-start:.6f} seconds")
                    print(f"Convergence: {'Converged' if ii < self.N else 'Not Converged'}")
                    print(f"Eigenvalue: {self.shift+1/miu:.6f}")
                    print(f"Eigenvector: {x_new}")
                    print("-" * 50)
                else:
                    print(f"Method: Inverse Iteration with Shift Quotient (σ = {self.shift}) + Rayleigh")
                    print(f"Iterations: {ii}")
                    print(f"Runtime: {end-start:.6f} seconds")
                    print(f"Convergence: {'Converged' if ii < self.N else 'Not Converged'}")
                    print(f"Eigenvalue: {rayleigh:.6f}")
                    print(f"Eigenvector: {x_new}")
                    print("-" * 50)
                break
            x=x_new

        eigenvalue=self.shift+1/miu
        return eigenvalue,x
    
    def SloveEigenvalue(self,x0,shift=6,mode=0):
        self.shift=shift
        self.mode=mode
        x=self._InverseShift(x0)
        return x
    

A=np.array([[6.0,2,1],
            [2,3,1],
            [1,1,1]])
x0=np.array([-1,2,3])
EA=EigenvalueAlgorithms(A,tol=1e-8,N=1000)
shifts=np.array([0,1,6-1e-8])

for shift in shifts:
    eval,evec=EA.SloveEigenvalue(x0,shift=shift,mode=0)
    EA.SloveEigenvalue(evec,shift=shift,mode=1)
    print('#'*50)
print('\n\n')

for shift in shifts:
    eval,evec=EA.SloveEigenvalue(x0,shift=shift,mode=1)
    EA.SloveEigenvalue(evec,shift=shift,mode=0)
    print('#'*50)

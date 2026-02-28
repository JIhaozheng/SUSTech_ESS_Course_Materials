import numpy as np

def calH(alpha): 
    alpha[0,0]-=np.linalg.norm(alpha) 
    omega=alpha/np.linalg.norm(alpha) 
    oo=omega@omega.T 
    H=np.eye(oo.shape[0])-2*oo 
    return H

def calQR(A):
    n=A.shape[0]
    subA=A.copy()
    Q=np.eye(n)
    for ii in range(n-1):
        alpha=subA[:,0].reshape(-1,1)
        h=calH(alpha)
        H=np.eye(n)
        H[ii:,ii:]=h
        Q=Q@H
        A=H@A        
        subA=A[ii+1:,ii+1:].copy()
    return Q,A

def calE(A,iter=200,tol=1e-10):
    Ak=A.copy().astype(float)
    for ii in range(iter):
        Q,R=calQR(Ak)
        Ak=R@Q
        if np.linalg.norm(Ak-np.triu(Ak)) < tol:
            break
    return np.diag(Ak)

B=np.array([[0,3,1],[0,4,-2],[2,1,1]])

A=np.array([[10,7,8,7],
            [7,5,6,5],
            [8,6,10,9],
            [7,5,9,10]])

print(calE(A))

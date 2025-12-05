import numpy as np
import time
import matplotlib.pyplot as plt
import os
os.makedirs("images", exist_ok=True)

class SPD:
    def __init__(self):
        self.n=None
        self.tol=1e-6

    def gen_coefficient_matrix(self):
        A=np.zeros((self.n,self.n))
        b=np.zeros((self.n))
        b[0]=1
        if self.n%2==1:
            b[self.n-1]=1
        A[0,0],A[0,1]=2,-1
        A[self.n-1,self.n-2],A[self.n-1,self.n-1]=-1,2
        for ii in range(1,self.n-1):
            A[ii,ii-1],A[ii,ii],A[ii,ii+1]=-1,2,-1
            if ii%2==0:
                b[ii]=1
        return A,b
    
    def TridiagonalMethod(self,b):
        a,B,c=-np.ones(self.n),2*np.ones(self.n),-np.ones(self.n)
        start_time=time.time()
        gamma,delta,y,x=np.zeros(self.n),np.zeros(self.n),np.zeros(self.n),np.zeros((self.n))
        beta=a
        gamma[0]=B[0]
        delta[0]=c[0]/gamma[0]
        y[0]=b[0]/gamma[0]
        for ii in range(1,self.n):
            gamma[ii]=B[ii]-a[ii]*delta[ii-1]
            delta[ii]=c[ii]/gamma[ii]
            y[ii]=(b[ii]-beta[ii]*y[ii-1])/gamma[ii]
        x[self.n-1]=y[self.n-1]
        for ii in range(self.n-2,-1,-1):
            x[ii]=y[ii]-delta[ii]*x[ii+1]
        end_time=time.time()
        res=np.linalg.norm(self.cal_Ax(x)-b)
        spend=end_time-start_time
        return x,spend,res
    
    def cal_Ax(self,x):
        b=np.zeros((self.n))
        b[0]=2*x[0]-x[1]
        b[1:self.n-1]=-x[0:self.n-2]+2*x[1:self.n-1]-x[2:self.n]
        b[self.n-1]=-x[self.n-2]+2*x[self.n-1]
        return b

    def ConjugateGradient(self,b):
        x=np.zeros((self.n))
        start_time=time.time()
        r=b
        p=r.copy()
        dot_r0=np.dot(r,r)
        for ii in range(self.n):
            Ap=self.cal_Ax(p)
            alpha=dot_r0/np.dot(p,Ap)
            x+=alpha*p
            r-=alpha*Ap
            dot_r1=np.dot(r,r)
            beta=dot_r1/dot_r0
            p=r+beta*p
            dot_r0=dot_r1
            if dot_r1<=self.tol**2:
                break
        end_time=time.time()
        spend=end_time-start_time
        return x,spend,np.sqrt(dot_r1)
        
    def solve(self,n,mode=0,tol=1e-3):
        self.n=n
        self.tol=tol
        A,b=self.gen_coefficient_matrix()
        if mode==0:
            x,spend,res=self.TridiagonalMethod(b)
        else:
            x,spend,res=self.ConjugateGradient(b)
        return x,spend,res

solver = SPD()
dim=np.array([3,4,5])
for ii in range(3):
    x1,_,_=solver.solve(dim[ii],mode=0)
    x2,_,_=solver.solve(dim[ii],mode=1)
    print(f'for dim={dim[ii]}, \nuse tridiagonal method we can get :{x1}\nuse conjugate gradient method we can get :{x2}\n\n')

dim_list=np.linspace(3,10000,20).astype(int)
t=np.zeros((len(dim_list),2))
x=t.copy()
res=np.zeros((len(dim_list)))
for ii in range(len(dim_list)):
        x1,t[ii,0],_=solver.solve(dim_list[ii],mode=0)
        x2,t[ii,1],res[ii]=solver.solve(dim_list[ii],mode=1)
        x[ii,0],x[ii,1]=np.linalg.norm(x1),np.linalg.norm(x2)


plt.plot(dim_list, t[:, 1], label='Conjugate Gradient',color='red')
plt.plot(dim_list, t[:, 0], label='Tridiagonal Method',color='blue')
plt.xlabel('Dimension',fontsize=12)
plt.ylabel('Time (s)',fontsize=12)
plt.title('Comparison of Computation Time: Tridiagonal vs. Conjugate Gradient Methods', fontsize=14)
plt.legend()
plt.grid()
plt.savefig('images/ComparisonOfComputationTime.png',dpi=300)
plt.show()

plt.semilogy(dim_list, res[:], label='Conjugate Gradient',color='black')
plt.xlabel('Dimension',fontsize=12)
plt.ylabel('Residual Norm ||Ax-b||)',fontsize=12)
plt.title('Numerical Residuals of Conjugate Gradient Methods', fontsize=14)
plt.legend()
plt.grid()
plt.savefig('images/NumericalResidualofConjugteGradientMethod.png',dpi=300)
plt.show()

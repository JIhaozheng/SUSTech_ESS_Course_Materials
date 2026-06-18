import numpy as np
import matplotlib.pyplot as plt

class Seismogram:
    def __init__(self,Xt,a,b,rho):
        self.Xt=Xt
        self.a=a
        self.b=b
        self.rho=rho

# used to calculate the integral of X(t-tau)*tau
    def int_Xtt(self,Xt,r,t,a,b):
        t_low,t_upper=r/a,r/b
        tau=np.linspace(t_low,t_upper,1000)
        int_Xtt=np.zeros_like(t)
        for ii in range(len(t)):
            int_Xtt[ii]=np.trapz(Xt(t[ii]-tau)*tau,tau)
        return int_Xtt

# used to calculate the radial displacement
    def ur(self,theta,r,t):
        int_value=self.int_Xtt(self.Xt,r,t,self.a,self.b)
        cos_theta=np.cos(theta)
        cos_theta[abs(cos_theta)<1e-10]=0
        first=2*cos_theta/(4*np.pi*self.rho*r**3)*int_value
        second=cos_theta/(4*np.pi*self.rho*self.a**2*r)*self.Xt(t-r/self.a)
        result=first+second
        return result

# used to calculate the tangential displacement
    def utheta(self,theta,r,t):
        int_value=self.int_Xtt(self.Xt,r,t,self.a,self.b)
        sin_theta=np.sin(theta)
        sin_theta[abs(sin_theta)<1e-10]=0
        first=sin_theta/(4*np.pi*self.rho*r**3)*int_value
        second=-sin_theta/(4*np.pi*self.rho*self.b**2*r)*self.Xt(t-r/self.b)
        result=first+second
        return result

if __name__ == "__main__":
    a,b,rho=4,1,2.7*10**3   
    def Ht(t):
        t=np.asarray(t,dtype=float)
        Ht_value=np.zeros_like(t)
        Ht_value[t>=0]=1
        return Ht_value

    SeisH=Seismogram(Ht,a,b,rho)
    theta=np.array([np.pi/6])
    r=30
    t=np.linspace(0,20,3000)
    ur=SeisH.ur(theta,r,t)
    plt.figure(figsize=(10,6))
    plt.plot(t,ur/np.max(ur),'k-',linewidth=2)
    plt.xlabel('Time (s)',fontsize=16)
    plt.ylabel('Displacement (km)',fontsize=16)
    plt.title('Radial Displacement',fontsize=20)
    # plt.xlim(7.5,10)
    plt.show()

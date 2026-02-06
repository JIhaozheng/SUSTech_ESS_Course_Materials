import numpy as np

class Eikonal3d:
    def __init__(self,v,ds=1,tol=1e-10,max_step=10000):
        self.v=v
        self.ds=ds
        self.max_step=max_step
        self.tol=tol
    
    def _1vx(self,x,y,z):
        delta=1e-3
        return (1/self.v(x+delta,y,z)-1/self.v(x,y,z))/delta
    def _1vy(self,x,y,z):
        delta=1e-3
        return (1/self.v(x,y+delta,z)-1/self.v(x,y+delta,z))/delta
    def _1vz(self,x,y,z):
        delta=1e-3
        return (1/self.v(x,y,z+delta)-1/self.v(x,y,z))/delta

    def _k(self,parameter):
        x,y,z,px,py,pz,_=parameter
        u=self.v(x,y,z)
        k11=u*px
        k12=u*py
        k13=u*pz
        k14=self._1vx(x,y,z)
        k15=self._1vy(x,y,z)
        k16=self._1vz(x,y,z)
        k17=1/u
        return np.array([k11,k12,k13,k14,k15,k16,k17])
    
    def _RK4(self,parameter1,ds):
        k1=self._k(parameter1)
        parameter2=parameter1+ds/2*k1
        k2=self._k(parameter2)
        parameter3=parameter1+ds/2*k2
        k3=self._k(parameter3)
        parameter4=parameter1+ds*k3
        k4=self._k(parameter4)
            
        return parameter1+ds/6*(k1+2*k2+2*k3+k4)  
    
    def _updateP(self,parameter):
        px0,py0,pz0=parameter[3],parameter[4],parameter[5]
        v=self.v(parameter[0],parameter[1],parameter[2])
        denom=np.sqrt(px0**2+py0**2+pz0**2)*v
        parameter[3]=px0/denom
        parameter[4]=py0/denom
        parameter[5]=pz0/denom

    def _checkds(self,ds,parameter,count=0):
        if abs(parameter[3]**2+parameter[4]**2+parameter[5]**2-
               1/self.v(parameter[0],parameter[1],
                        parameter[2])**2)>self.tol and count<10:
            ds=ds/2
            return self._checkds(ds,parameter,count+1)
        return ds

    
    def Raypath(self,x0,y0,z0,toa,azi,multi=0):
        px0,py0,pz0=np.sin(toa)*np.cos(azi)/self.v(x0,y0,z0),np.sin(toa)*np.sin(azi)/self.v(x0,y0,z0),np.cos(toa)/self.v(x0,y0,z0)
        ds=self._checkds(self.ds,np.array([x0,y0,z0,px0,py0,pz0]))
        path_x=[x0]
        path_y=[y0]
        path_z=[z0]
        path_t=[0]
        parameter1=self._RK4(np.array([x0,y0,z0,px0,py0,pz0,0]),ds)
        for ii in range(self.max_step):
            path_x.append(parameter1[0])
            path_y.append(parameter1[1])
            path_z.append(parameter1[2])
            path_t.append(parameter1[6])
            if multi==0:
                if path_z[-1]<0:
                    break
            ds=self._checkds(self.ds,parameter1)
            parameter1=self._RK4(parameter1,ds)
            self._updateP(parameter1)
                

        return np.array(path_x),np.array(path_y),np.array(path_z),np.array(path_t)

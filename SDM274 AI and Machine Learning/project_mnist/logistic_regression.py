import numpy as np

class MulticlassLogisticRegression:
    def __init__(self,reg_lambda=0.01):
        self.w=None
        self.nsample,self.ndim=None,None
        self.X,self.Y=None,None
        self.normalization=None
        self.mean,self.std=None,None
        self.reg_lam=reg_lambda
    
    def _f(self,z):
        z=z-np.max(z,axis=1, keepdims=True)
        return np.exp(z)/np.sum(np.exp(z),axis=1, keepdims=True)
    
    def normalize(self,X):
        X_norm=X.copy()
        if self.normalization:
            self.mean=X.mean(axis=0)
            self.std=X.std(axis=0)
            for ii in range(len(self.std)):
                if self.std[ii]==0:
                    X_norm[:,ii]=0
                else:
                    X_norm[:,ii]=(X_norm[:,ii]-self.mean[ii])/self.std[ii]
        return X_norm
    
    def _normalize(self,x):
        x_copy=x.copy()
        if self.normalization:
            for ii in range(len(self.std)):
                if self.std[ii]==0:
                    x_copy[:,ii]=0
                else:
                    x_copy[:,ii]=(x_copy[:,ii]-self.mean[ii])/self.std[ii]
        return x_copy
    
    def fit(
            self,X,Y,epochs=100,lr=0.01,batch_size=32,normalization=True):
        self.X=X.astype(np.float32)
        self.Y=Y
        self.normalization=normalization
        self.nsample,self.ndim=X.shape
        self.nclass=np.max(Y)+1
        X_norm=self.normalize(X)
        onehot=np.zeros((self.nsample,self.nclass))
        onehot[np.arange(self.nsample),Y]=1
        b=np.ones((self.nsample,1))
        X_norm=np.hstack((b,X_norm))

        self.w=np.zeros((self.ndim+1,self.nclass))

        losses=[]

        for epoch in range(epochs):
            idx=np.arange(self.nsample)
            np.random.shuffle(idx)

            for s_idx in range(0,self.nsample,batch_size):
                e_idx=s_idx+batch_size
                batch_idx=idx[s_idx:e_idx]
                X_batch=X_norm[batch_idx]
                Y_batch=onehot[batch_idx]
                y_pred=self._f(X_batch@self.w)
                dw=-X_batch.T@(Y_batch-y_pred)
                self.w=self.w-lr*dw
            epsilon=1e-8
            loss = -np.sum(onehot * np.log(self._f(X_norm @ self.w) + epsilon))
            loss += (self.reg_lam / 2) * np.sum(self.w * self.w)

            losses.append(loss)

        return losses,self.w
    
    def predict(self,X_pred):
        X_norm=self._normalize(X_pred)
        b=np.ones((X_norm.shape[0],1))
        X_norm=np.hstack((b,X_norm))
        y_pred=self._f(X_norm@self.w)
        return np.argmax(y_pred,axis=1)
    
    def accuracy(self,x,real):
        pred=self.predict(x)
        accuracy=np.mean(pred==real)*100
        true_idx=np.where(pred==real)[0]
        wrong_idx=np.where(pred!=real)[0]
        return accuracy,pred,real,true_idx,wrong_idx
    









        


        


        


        


        


        


        


        


        


        


        


        


        


        


        


        


        

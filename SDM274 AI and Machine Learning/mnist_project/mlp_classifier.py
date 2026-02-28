import numpy as np

class Layer:
    def __init__(self,input_dim,output_dim,activation='relu'):
        self.input_dim=input_dim
        self.output_dim=output_dim
        self.activation=activation

        self.W=np.random.randn(input_dim,output_dim)*np.sqrt(2./input_dim)
        self.b=np.zeros((1,output_dim))

        self.X=None
        self.Z=None
        self.A=None

        self.dw=None
        self.db=None
    def forward(self,X):
        self.X=X
        self.Z=X@self.W+self.b
        self.A=self._activate(self.Z)
        return self.A
    def _activate(self,Z):
        if self.activation=='relu':
            return np.maximum(0,Z)
        elif self.activation=='softmax':
            Z_exp=np.exp(Z-np.max(Z,axis=1,keepdims=True))
            return Z_exp/np.sum(Z_exp,axis=1,keepdims=True)
    def backward(self,dA,lr):
        if self.activation=='relu':
            dZ=dA.copy()
            dZ[self.Z<=0]=0
        elif self.activation=='softmax':
            dZ=dA
        self.dW=self.X.T@dZ/self.X.shape[0]
        self.db=np.sum(dZ,axis=0,keepdims=True)/self.X.shape[0]

        dX=dZ@self.W.T

        self.W-=lr*self.dW
        self.b-=lr*self.db

        return dX

class MultiLayerPerceptron:
    def __init__(self,layer_dims,reg_lambda=0.001,normalization=True):
        self.layers=[]
        self.n_layers=len(layer_dims)-1
        self.reg_lam=reg_lambda
        self.normalizaiton=normalization
        self.mean=None
        self.std=None

        for ii in range(len(layer_dims)-1):
            act='softmax' if ii == len(layer_dims)-2 else 'relu'
            self.layers.append(Layer(layer_dims[ii],layer_dims[ii+1],activation=act))

    def normalize(self,X):
        X_norm=X.copy()
        if self.normalizaiton:
            self.mean=X.mean(axis=0)
            self.std=X.std(axis=0)
            for ii in range(len(self.std)):
                if self.std[ii]==0:
                    X_norm[:,ii]=0
                else:
                    X_norm[:,ii]=(X_norm[:,ii]-self.mean[ii])/self.std[ii]
        return X_norm
    
    def forward(self,X):
        A=X
        for layer in self.layers:
            A=layer.forward(A)
        return A
    def cross_entropy_loss(self,Y_pred,onehot):
        epsilon=1e-8
        loss=-np.sum(onehot*np.log(Y_pred+epsilon))/Y_pred.shape[0]
        for layer in self.layers:
            loss+=(self.reg_lam/2)*np.sum(layer.W**2)
        return loss

    def backward(self,Y_pred,onehot,lr):
        dA=(Y_pred-onehot)/Y_pred.shape[0]
        for layer in reversed(self.layers):
            dA=layer.backward(dA,lr)


    def fit(self,X,Y,epochs=100,lr=0.01,batch_size=128):
        X=self.normalize(X)
        nsample=X.shape[0]
        nclass=np.max(Y)+1
        onehot=np.zeros((nsample,nclass))
        onehot[np.arange(nsample),Y]=1
        losses=[]

        for epoch in range(epochs):
            idx=np.arange(nsample)
            np.random.shuffle(idx)
            for s_idx in range(0,nsample,batch_size):
                e_idx=s_idx+batch_size
                X_batch=X[s_idx:e_idx]
                Y_batch=onehot[s_idx:e_idx]

                Y_pred=self.forward(X_batch)
                self.backward(Y_pred,Y_batch,lr)

            Y_pred_full=self.forward(X)
            loss=self.cross_entropy_loss(Y_pred_full,onehot)
            losses.append(loss)
        return losses
    
    def predict(self,X):
        X_norm=self.normalize(X)
        Y_pred=self.forward(X_norm)
        return np.argmax(Y_pred,axis=1)
    
    def accuracy(self,X,real):
        pred=self.predict(X)
        accuracy=np.mean(pred==real)*100
        true_idx=np.where(pred==real)[0]
        wrong_idx=np.where(pred!=real)[0]
        return accuracy,pred,real,true_idx,wrong_idx



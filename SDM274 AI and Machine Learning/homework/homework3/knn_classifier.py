import numpy as np
from collections import Counter

class KnnClassifier:
    def __init__(self,features,labels,normalization=None):
        self.X_train=features
        self.label=labels
        self.num,self.dim=features.shape

        self.normalization=normalization
        if normalization:
            self.mean=self.X_train.mean(axis=0)
            self.std=self.X_train.mean(axis=0)
            for ii in range(self.dim):
                self.X_train[:,ii]=(self.X_train[:,ii]-self.mean[ii])/self.std[ii]
        
    def _Euclidean_distance(self,x):
        diff=self.X_train-x
        diff2=abs(diff**self.mode)
        dist=np.sum(diff2,axis=1)
        return dist
    
    def get_nearest(self,x,k):
        dis_list=self._Euclidean_distance(x)
        return np.argsort(dis_list)[:k]
    
    def pred(self,x,k):
        N=x.shape[0]
        x_copy=x.copy().astype(np.float64)
        if self.normalization:
            for ii in range(self.dim):
                x_copy[:,ii]=(x_copy[:,ii]-self.mean[ii])/self.std[ii]
        pred=[]
        for ii in range(N):
            test=x_copy[ii,:]
            k_nearest_idx=self.get_nearest(test,k)
            k_nearest_label=self.label[k_nearest_idx]
            pred.append(Counter(k_nearest_label).most_common(1)[0][0])
        return np.array(pred)
        
    def accuracy(self,x,label,k,mode):
        self.mode=mode
        pred=self.pred(x,k)
        accuracy=np.mean(pred==label)*100
        true_idx=np.where(pred==label)[0]
        false_idx=np.where(pred!=label)[0]
        return accuracy,pred,true_idx,false_idx
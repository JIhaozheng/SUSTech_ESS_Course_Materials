import numpy as np
from collections import Counter
from Kd_tree import KdTree

class KnnClassifier:
    def __init__(self,train_set,train_label,use_kdtree=False,normalization=False,lead_size=1):
        self.label=train_label
        self.num=len(train_label)
        self.mode=use_kdtree
        self.X_train=train_set.astype(np.float32)
        self.mean=self.X_train.mean(axis=0)
        self.std=self.X_train.std(axis=0)
        self.normalization=normalization
        if normalization:
            for ii in range(len(self.std)):
                if self.std[ii]==0:
                    self.X_train[:,ii]=0 
                else:
                    self.X_train[:,ii]=(self.X_train[:,ii]-self.mean[ii])/self.std[ii]
        self.kd_tree=None

        if self.mode:
            self.kd_tree=KdTree(self.X_train,self.label)

    def _Euclidean_distance(self,x):
        diff=self.X_train-x
        diff2=diff**2
        dist=np.sum(diff2,axis=1)
        return dist
    
    def get_nearest(self,x,k):
        dis_list=self._Euclidean_distance(x)
        k_nearest_idx=np.argsort(dis_list)[:k]
        return k_nearest_idx
    
    def pred(self,x,k):
        N=x.shape[0]
        x_copy = x.copy().astype(np.float32)
        if self.normalization:
            for ii in range(len(self.std)):
                if self.std[ii] == 0:
                    x_copy[:, ii] = 0
                else:
                    x_copy[:, ii] = (x_copy[:, ii] - self.mean[ii]) / self.std[ii]
        pred=[]
        for ii in range(N):
            test=x_copy[ii,:]
            if self.mode:
                k_nearest_label=self.kd_tree.query(test,k)
                pred.append(Counter(k_nearest_label).most_common(1)[0][0])
            else:
                k_nearest_idx=self.get_nearest(test,k)
                k_nearest_label=self.label[k_nearest_idx]
                pred.append(Counter(k_nearest_label).most_common(1)[0][0])
        return np.array(pred)
    
    def accuracy(self,x,real,k):
        pred=self.pred(x,k)
        accuracy=np.mean(pred==real)*100
        true_idx=np.where(pred==real)[0]
        wrong_idx=np.where(pred!=real)[0]
        return accuracy,pred,real,true_idx,wrong_idx
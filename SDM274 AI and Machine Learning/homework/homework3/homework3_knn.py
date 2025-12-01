from knn_classifier import KnnClassifier
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import os
os.makedirs("images", exist_ok=True)

np.random.seed(42)
df=pd.read_csv("wdbc.data",header=None)
data_features=df.iloc[:,2:12].values
data_labels=df.iloc[:,1].values
data_labels = (data_labels == 'M').astype(int)
for mode in range(1,11):
    table=np.zeros(len(data_labels),dtype=bool)
    for ii in range(10):
        idx=np.where(data_labels==ii)[0]
        np.random.shuffle(idx)
        number=round(len(idx)*0.7)
        table[idx[:number]]=True
        print(f"\nFor label {ii} ({'Benign' if ii == 0 else 'Malignant'}):")
        print(f"  Total samples: {len(idx)}")
        print(f"  Train samples: {number}")
        print(f"  Test samples: {len(idx) - number}")
    train_features,train_labels=data_features[table],data_labels[table]
    test_features,test_labels=data_features[~table],data_labels[~table]
    k=np.arange(1,11)
    knn=KnnClassifier(train_features,train_labels,normalization=True)
    accuracy=[]
    for ii in k:
        a,_,_,_=knn.accuracy(test_features,test_labels,ii,mode)
        accuracy.append(a)

    max_idx=np.argmax(accuracy)
    print(accuracy[max_idx])
    plt.figure(figsize=(10, 6))
    plt.plot(k, accuracy, marker='o', linestyle='-', color='b', label='Accuracy')
    plt.plot(k[max_idx], accuracy[max_idx], marker='o', color='red', label='Max Accuracy')

    plt.title(f'KNN Accuracy vs K Value mode ({mode}) with largest accuracy {accuracy[max_idx]:.4f}', fontsize=16)
    plt.xlabel('K Value', fontsize=14)
    plt.ylabel('Accuracy', fontsize=14)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.xticks(k)
    plt.legend(fontsize=12)
    plt.savefig(f"images/KnnAccuracyWithKvalues{mode}")
    plt.show()
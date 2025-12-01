import numpy as np
import decision_tree as DT
from decisiontreeplotter import DecisionTreePlotter
import matplotlib.pyplot as plt
import importlib
import os
importlib.reload(DT)

data = np.loadtxt('../dataset/lenses/lenses.data', dtype=int)


import os
features_dict = {
    0: {'name': 'age', 'value_names': {1:'young', 
                                       2:'pre-presbyopic',
                                       3:'presbyopic'}
        },
    1: {'name':'prescript', 
        'value_names': {1: 'myope',
                        2: 'hypermetrope'}
        },
    2: {'name': 'astigmatic', 
        'value_names': {1: 'no', 
                        2: 'yes'}
    },
    3: {'name': 'tear rate', 
        'value_names': {1:'reduced', 
                        2:'normal'}
        },
}

label_dict = {
    1: 'hard',
    2: 'soft',
    3: 'no_lenses',
}

X = data[:, 1:-1]
y = data[:, -1]
idx=np.arange(len(y))

np.random.seed(42)
np.random.shuffle(idx)
per=np.arange(0.05,0.95,0.05)
accuracy=[]
training_size=[]
depth=[]
for ii in range(len(per)):
    num=round(len(idx)*per[ii])
    X_train=X[idx[0:num],:]
    y_train=y[idx[0:num]]
    X_test=X[idx[num:-1],:]
    y_test=y[idx[num:-1]]
    dt01 = DT.DecisionTree()
    dt01.train(X_train,y_train)
    y_pred=dt01.predict(X_test)
    accuracy.append(np.mean(y_pred==y_test)*100)
    training_size.append(len(X_train))
    #dtp = DecisionTreePlotter(dt01.tree_, feature_names = features_dict, label_names=label_dict)
    #dtp.plot()
    #os.rename("Decision Tree.gv.pdf", f"Decision Tree (training size{len(X_train)}).pdf")
    depth.append( dt01.get_tree_depth())
max_idx=np.argmax(accuracy)
print(accuracy[max_idx])
plt.figure(figsize=(10, 6))
plt.plot(training_size, accuracy, marker='o', linestyle='-', color='b', label='Accuracy')
plt.plot(training_size[max_idx], accuracy[max_idx], marker='o', color='red', label='Max Accuracy')
plt.title('Decision Tree Accuracy vs Training Set Size', fontsize=16)
plt.xlabel('Training Set Size', fontsize=14)
plt.ylabel('Accuracy (%)', fontsize=14)
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.xticks(training_size)
plt.legend(fontsize=12)
plt.savefig("images/AccuracyWithTrainingSetSize.png")
plt.show()

plt.plot(training_size,depth, marker='o', linestyle='-', color='b')
plt.title('Decision Tree Depth vs Training Set Size', fontsize=16)
plt.xlabel('Training Set Size', fontsize=14)
plt.ylabel('Accuracy (%)', fontsize=14)
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.xticks(training_size)
plt.legend(fontsize=12)
plt.savefig("images/DepthWithTrainingSetSize.png")
plt.show()
import numpy as np
import struct
import matplotlib.pyplot as plt
import random
import os

os.makedirs("images/data_load", exist_ok=True)

images_train_dir = "input/train-images.idx3-ubyte"
images_test_dir = "input/t10k-images.idx3-ubyte"

labels_train_dir = "input/train-labels.idx1-ubyte"
labels_test_dir = "input/t10k-labels.idx1-ubyte"

def read_image(dir):
    with open(dir,'rb') as f:
        magic, =struct.unpack('>I',f.read(4))
        n_images,n_rows,n_cols = struct.unpack('>III',f.read(12))
        images=np.fromfile(f,dtype=np.uint8).reshape(n_images,n_rows,n_cols)
    return images

def read_label(dir):
    with open(dir,'rb') as f:
        magic, =struct.unpack('>I',f.read(4))
        n_labels, = struct.unpack('>I',f.read(4))
        labels=np.fromfile(f,dtype=np.uint8)
    return labels


train_images=read_image(images_train_dir)
test_images=read_image(images_test_dir)
images_set = np.concatenate([train_images, test_images], axis=0)


train_labels=read_label(labels_train_dir)
test_labels=read_label(labels_test_dir)
labels_set = np.concatenate([train_labels, test_labels], axis=0)

np.random.seed(42)
table=np.zeros(len(labels_set),dtype=bool)
for lbl in np.unique(labels_set):
    lbl_idx=np.where(labels_set==lbl)[0]
    np.random.shuffle(lbl_idx)
    number=round(len(lbl_idx)*0.8)
    table[lbl_idx[:number]]=True
    print(f"Number of Lable{lbl}: {len(lbl_idx)}, Number of training set: {number}, Number of test set: {len(lbl_idx)-number}")
new_train_images,new_test_images=images_set[table],images_set[~table]
new_train_labels,new_test_labels=labels_set[table],labels_set[~table]


random_idx=random.sample(range(len(new_train_labels)),20)
plt.figure(figsize=(12,7))
for ii in range(20):
    plt.subplot(4,5,ii+1)
    plt.imshow(new_train_images[random_idx[ii]],cmap='gray')
    plt.title(f"ID: {random_idx[ii]}, label: {new_train_labels[random_idx[ii]]}", fontsize=12)
    plt.axis('off')
plt.suptitle("Check the train data set ( before preprocessing )",fontweight='bold')

plt.savefig(
    f"images/data_load/Check_the_train_data_set_before_preprocessing.png",
    dpi=300,
    bbox_inches="tight",
)
plt.show()

new_train_images = new_train_images.astype(np.float32)
train_mean=new_train_images.mean(axis=(1,2),keepdims=True)
new_train_images=(new_train_images>train_mean).astype(np.uint8)

plt.figure(figsize=(12,7))
for ii in range(20):
    plt.subplot(4,5,ii+1)
    plt.imshow(new_train_images[random_idx[ii]],cmap='gray')
    plt.title(f"ID: {random_idx[ii]}, label: {new_train_labels[random_idx[ii]]}", fontsize=12)
    plt.axis('off')
plt.suptitle("Check the train data set ( after preprocessing )",fontweight='bold')

plt.savefig(
    f"images/data_load/Check_the_train_data_set_after_preprocessing.png",
    dpi=300,
    bbox_inches="tight",
)
plt.show()


random_idx=random.sample(range(len(new_test_labels)),20)
plt.figure(figsize=(12,7))
for ii in range(20):
    plt.subplot(4,5,ii+1)
    plt.imshow(new_test_images[random_idx[ii]],cmap='gray')
    plt.title(f"ID: {random_idx[ii]}, label: {new_test_labels[random_idx[ii]]}", fontsize=12)
    plt.axis('off')
plt.suptitle("Check the train data set ( before preprocessing )",fontweight='bold')
plt.savefig(
    f"images/data_load/Check_the_test_data_set_before_preprocessing.png",
    dpi=300,
    bbox_inches="tight",
)
plt.show()

new_test_images  = new_test_images.astype(np.float32)
test_mean=new_test_images.mean(axis=(1,2),keepdims=True)
new_test_images=(new_test_images>test_mean).astype(np.uint8)
plt.figure(figsize=(12,7))
for ii in range(20):
    plt.subplot(4,5,ii+1)
    plt.imshow(new_test_images[random_idx[ii]],cmap='gray')
    plt.title(f"ID: {random_idx[ii]}, label: {new_test_labels[random_idx[ii]]}", fontsize=12)
    plt.axis('off')
plt.suptitle("Check the training data set ( after preprocessing )",fontweight='bold')
plt.savefig(
    f"images/data_load/Check_the_test_data_set_after_preprocessing.png",
    dpi=300,
    bbox_inches="tight",
)
plt.show()

def gen_feature(image,N_gap):
    nrows,ncols=image.shape
    idx=np.full(nrows*N_gap*2, -1, dtype=np.int8)
    for ii in range(nrows):
        count=0
        condition=-1
        for jj in range(ncols-1):
            d = np.int16(image[ii,jj+1]) - np.int16(image[ii,jj]) 
            if d!=0:
                if condition==-1:
                    idx[N_gap*2*ii+count]=jj+1
                else:
                    idx[N_gap*2*ii+count]=jj
                count=count+1
                condition=-1*condition
                if count>=2*N_gap:
                    break        
    return idx

N_gap=2
test_features=[]
for ii in range(len(new_test_labels)):
    feature=gen_feature(new_test_images[ii],N_gap)
    test_features.append(feature)
test_set=np.array(test_features)


train_features=[]
for ii in range(len(new_train_labels)):
    feature=gen_feature(new_train_images[ii],N_gap)
    train_features.append(feature)
train_set=np.array(train_features)

np.save('input/train_set.npy',train_set)
np.save('input/train_label.npy',new_train_labels)
np.save('input/test_set.npy',test_set)

np.save('input/test_label.npy',new_test_labels)

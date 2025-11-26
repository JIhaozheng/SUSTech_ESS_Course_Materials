import numpy as np

def decode(storge,dia):
    d=len(dia)
    M=np.zeros((d,d))
    M[0,0]=storge[0]
    for ii in range(d-1):
        n=dia[ii+1]-dia[ii]
        M[ii+1,ii+2-n:ii+2]=storge[dia[ii]:dia[ii+1]]
        M[ii+2-n:ii+2,ii+1]=storge[dia[ii]:dia[ii+1]]
    return M

def encode(M):
    d=len(M)
    storge=[]
    dia=[]
    for ii in range(d):
        idx=M[ii].nonzero()[0]
        storge.extend(M[ii,idx[0]:ii+1])
        dia.append(len(storge))
    return storge,dia


M1=np.array([
    [2,2,0,0,0,0],
    [2,1,0,3,9,0],
    [0,0,1,0,0,0],
    [0,3,0,5,5,0],
    [0,9,0,5,7,3],
    [0,0,0,0,3,8]])

M2=np.array([
    [2,2,0,3,0,0],
    [2,1,4,0,0,0],
    [0,4,1,0,9,0],
    [3,0,0,5,5,0],
    [0,0,9,5,7,3],
    [0,0,0,0,3,8]])

s1, d1 = encode(M1)
print(f"Encode: Matrix 1\nStorage with 1D={s1}")
print(f"Diagonal index={d1}")
m1 = decode(s1, d1)
print(f"Test: Use storage and diagonal index to decode=\n{m1}\n")

s2, d2 = encode(M2)
print(f"Encode: Matrix 2\nStorage with 1D={s2}")
print(f"Diagonal index={d2}")
m2 = decode(s2, d2)
print(f"Test: Use storage and diagonal index to decode=\n{m2}")

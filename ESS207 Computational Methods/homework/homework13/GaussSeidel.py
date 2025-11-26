import numpy as np

A=np.array([
    [0,0.5,0,0],
    [0,1/4,1/2,0],
    [0,1/8,1/4,1/2],
    [0,1/16,1/8,1/4]
])
b=np.array([
    1/2,1/4,5/8,5/16
])

x=np.array([
    0,0,0,0
])
x_prec = np.array([1.2, 1.4, 1.6, 0.8], dtype=float)

for ii in range(36):
    x = A @ x + b

    l2_norm = np.linalg.norm(x - x_prec)

    print(
        f"Iter {ii:2d}: x1={x[0]:.6f}, x2={x[1]:.6f}, x3={x[2]:.6f}, x4={x[3]:.6f} "
        f"| L2 error={l2_norm:.6f}"
    )
    
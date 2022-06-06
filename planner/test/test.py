import numpy as np

A = [1, 2, 3]
A = np.array(A)
print(A)
print(len(A.shape))
matrixA = np.random.rand(1, 10000)
print(matrixA.shape)
# for row in A:
#     print(row)
#     row = row.reshape(1, len(row))
#     print(row)
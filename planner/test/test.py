import numpy as np
import math

A = np.array([[1, 2, 3], [4, 5, 6]])
mu = np.mean(A, axis = 0)
sigma = np.std(A, axis = 0)

norm_A = (A - mu) / sigma
print(norm_A)
w = np.array([2, 1, 3])
print(norm_A.dot(w))
# for row in A:
#     print(row)
#     row = row.reshape(1, len(row))
#     print(row)
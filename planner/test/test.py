import numpy as np
import math

A = np.random.uniform(-100, 100, (3000000, 20))
mu = np.mean(A, axis = 0)
sigma = np.std(A, axis = 0)

print(abs(A).max())
norm_A = (A - mu) / sigma
print(abs(norm_A).max())
print(norm_A.max(axis = 0))
norm_A = norm_A / norm_A.max(axis = 0)
# w = np.array([2, 1, 3])
# print(norm_A.dot(w))
# for row in A:
#     print(row)
#     row = row.reshape(1, len(row))
#     print(row)
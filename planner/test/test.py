import numpy as np

A = [1, 2, 3]
A = np.array(A)
B = np.zeros(3)
C = np.hstack((A,B))
D = C.reshape((2, 3))
print(D[0])

# for row in A:
#     print(row)
#     row = row.reshape(1, len(row))
#     print(row)
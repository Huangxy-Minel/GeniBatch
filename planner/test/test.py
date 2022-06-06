import numpy as np

A = [1, 2, 3]
A = np.array(A)
B = np.zeros(3)
print(np.hstack((A,B)))
# for row in A:
#     print(row)
#     row = row.reshape(1, len(row))
#     print(row)
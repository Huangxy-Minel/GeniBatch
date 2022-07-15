import numpy as np
import math
import time
import multiprocessing
import pickle, sys

A = np.random.uniform(-100, 100, (10000000, 3))

time1 = time.time()
test = pickle.dumps(A)
time2 = time.time()
print(sys.getsizeof(test))
print(time2 - time1)

# print(A)
# print("-----")
# print(A[0:])

# print(multiprocessing.cpu_count())
# w = np.array([2, 1, 3])
# print(norm_A.dot(w))
# for row in A:
#     print(row)
#     row = row.reshape(1, len(row))
#     print(row)
import numpy as np
class DataStorage(object):
    def __init__(self):
        self.matrix_idx = 0
        self.data_store = {}                        # store all input matrices. key (matrix ID) : value (matrix)

    def addMatrix(self, matrix:np.ndarray):
        self.data_store[self.matrix_idx] = matrix
        self.matrix_idx += 1
        return self.matrix_idx - 1

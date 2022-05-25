import numpy as np
class DataStorage(object):
    def __init__(self):
        self.matrix_idx = 0
        self.data_store = {}                        # store all input matrices. key (matrix ID) : value (matrix)

    def addMatrix(self, matrix:np.ndarray):
        self.data_store[self.matrix_idx] = matrix
        self.matrix_idx += 1
        return self.matrix_idx - 1

    def getMatrixNum(self, data_idx):
        return self.matrix_idx - 1

    def getDataFromIdx(self, data_idx):
        '''
            Get data based on data_idx
            Input:
                data_idx: 4 tuples, (matrix_id, row_id, slot_start_idx, vector_len)
        '''
        matrix_id, row_id, slot_start_idx, vector_len = data_idx
        return self.data_store[matrix_id][row_id][slot_start_idx:slot_start_idx+vector_len]

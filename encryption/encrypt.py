import numpy as np
from federatedml.FATE_Engine.python.bigintengine.gpu.gpu_store import FPN_store
class BatchEncryption(object):
    @staticmethod
    def batchEncrypt(data:np.array, pub_key, batch_schemes):
        res = []
        ''''Encrypt a matrix based on a given batch scheme'''
        col_num = data.shape[1]
        for row_vec, batch_scheme in zip (data, batch_schemes):
            # make up zeros
            max_element_num, split_num = batch_scheme
            row_vec = np.hstack((row_vec, np.zeros(max_element_num * split_num - col_num)))
            # package into batch number
            fpn_store = FPN_store.init_from_arr(row_vec, pub_key.n, pub_key.max_int)
            pen_store = fpn_store.encrypt(pub_key)
            pen_store = pen_store.obfuscation()
            encrypted_row_vector = pen_store.get_PEN_ndarray()
            # reshape
            encrypted_row_vector = np.array(encrypted_row_vector)
            res.append(encrypted_row_vector.reshape((split_num, max_element_num)))
        return res
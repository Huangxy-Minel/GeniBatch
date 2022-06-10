import numpy as np
from federatedml.FATE_Engine.python.bigintengine.gpu.gpu_store import FPN_store
class BatchEncryption(object):
    def __init__(self, encoder, pub_key=None, private_key=None):
        self.encoder = encoder
        self.pub_key = pub_key
        self.private_key = private_key

    def batchEncrypt(self, data:np.array, batch_scheme, pub_key=None):
        ''''
            Encrypt a row vector based on a given batch scheme; row_vector should be 2-D array
            Encrypting process: row vector -> several BatchEncodeNumber -> FixPointNumber -> PaillierEncryptedNumber
        '''
        if pub_key:
            pub_key_used_in_encrypt = pub_key
        else:
            pub_key_used_in_encrypt = self.pub_key
        col_num = data.shape[1]
        row_vec = data[0]
        # make up zeros
        max_element_num, split_num = batch_scheme
        row_vec = np.hstack((row_vec, np.zeros(max_element_num * split_num - col_num)))
        # package into batch number
        fpn_store = FPN_store.init_from_arr(row_vec, pub_key_used_in_encrypt.n, pub_key_used_in_encrypt.max_int)
        pen_store = fpn_store.encrypt(pub_key_used_in_encrypt)
        pen_store = pen_store.obfuscation()
        encrypted_row_vector = pen_store.get_PEN_ndarray()
        # reshape
        encrypted_row_vector = np.array(encrypted_row_vector)
        return encrypted_row_vector.reshape((split_num, max_element_num))
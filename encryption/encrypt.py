import numpy as np
from federatedml.FATE_Engine.python.bigintengine.gpu.gpu_store import FPN_store
from federatedml.FATE_Engine.python.bigintengine.gpu.gpu_engine import fp_p2c_big_integer

class BatchEncryptedNumber(object):
    def __init__(self, value, scaling, size):
        self.value = value
        self.scaling = scaling
        self.size = size

class BatchEncryption(object):
    def __init__(self, encoder, pub_key=None, private_key=None):
        self.encoder = encoder
        self.pub_key = pub_key
        self.private_key = private_key

    def gpuBatchEncrypt(self, data:np.array, pub_key=None):
        ''''
            Encrypt several row_vec
            Encrypting process: row vector -> several BatchEncodeNumber -> FixPointNumber -> PaillierEncryptedNumber
            Input:
                data: 2-D array. Each element of data includes several slot number. The function will transform these slot numbers to a BatchEncodeNumber. then encrypt.
        '''
        if pub_key:
            pub_key_used_in_encrypt = pub_key
        else:
            pub_key_used_in_encrypt = self.pub_key
        # package into batch number
        plaintext_list = [self.encoder.batchEncode(row_vec) for row_vec in data]    # a list of BatchEncodeNumber
        values_of_BENs = [v.value for v in plaintext_list]
        fpn_store = FPN_store.fromBigIntegerList(values_of_BENs, pub_key_used_in_encrypt)
        pen_store = fpn_store.encrypt(pub_key_used_in_encrypt)
        print(pen_store)
        pen_store = pen_store.obfuscation()
        encrypted_value_list = pen_store.get_PEN_ndarray()  # a list of PEN
        encrypted_row_vector = [BatchEncryptedNumber(value, batchEncodeNumber.scaling, batchEncodeNumber.size) for value, batchEncodeNumber in zip(encrypted_value_list, plaintext_list)]
        return encrypted_row_vector
import numpy as np
import time

from threading import Thread
from federatedml.FATE_Engine.python.bigintengine.gpu.gpu_store import FPN_store, PEN_store
from federatedml.FATE_Engine.python.bigintengine.gpu.gpu_engine import fp_p2c_big_integer

class BatchEncryptedNumber(object):
    def __init__(self, value, scaling, size):
        self.value = value          # store the value of all encrypted BatchEncodeNumber: PEN_store
        self.scaling = scaling
        self.size = size

class BatchEncryption(object):
    def __init__(self, pub_key=None, private_key=None):
        self.pub_key = pub_key
        self.private_key = private_key

    def gpuBatchEncrypt(self, fpn_store:FPN_store, scaling, size, pub_key):
        ''''
            Encrypt several row_vec
            Encrypting process: row vector -> several BatchEncodeNumber -> FixPointNumber -> PaillierEncryptedNumber
            Input:
                data: list. Each element of data includes several slot number. The function will transform these slot numbers to a BatchEncodeNumber. then encrypt.
            Return: a BatchEncryptedNumber
        '''
        if pub_key:
            pub_key_used_in_encrypt = pub_key
        else:
            raise NotImplementedError("Please provide a public key when encrypting!")
        # package into batch number
        pen_store = fpn_store.encrypt(pub_key_used_in_encrypt)
        pen_store = pen_store.obfuscation()
        encrypted_row_vector = BatchEncryptedNumber(pen_store, scaling, size)
        return encrypted_row_vector

    def gpuBatchDecrypt(self, data:BatchEncryptedNumber, private_key):
        pen_store = data.value
        # time1 = time.time()
        batch_encoding_values = pen_store.decrypt_without_decode(private_key)
        # time2 = time.time()
        # print(time2 - time1)
        return batch_encoding_values

    def toList(self):
        '''Transform the PEN_store to a list of PaillierEncryptedNumber'''
        return self.value.get_PEN_ndarray()
        

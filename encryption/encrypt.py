import numpy as np
import time, math, copy

from federatedml.FATE_Engine.python.bigintengine.gpu.gpu_store import FPN_store, PEN_store
from federatedml.FATE_Engine.python.BatchPlan.encoding.encoder import BatchEncoder
from federatedml.secureprotol.fate_paillier import PaillierKeypair, PaillierPublicKey, PaillierPrivateKey, PaillierEncryptedNumber


class BatchEncryptedNumber(object):
    def __init__(self, value, scaling, size):
        self.value = value          # store the value of all encrypted BatchEncodeNumber: PEN_store or a list of PaillierEncryptedNumber
        self.scaling = scaling
        self.size = size
        '''Use for lazy operation'''
        self.batch_idx_map = None
        self.valid_idx = None
    
    def __add__(self, other):
        if not isinstance(self.value, list):
            raise TypeError("In CPU mode, BatchEncryptedNumber.value should be a list of PaillierEncryptedNumber!")
        if isinstance(other, list) and len(self.value) == len(other):
            value = [v1 + v2 for v1, v2 in zip(self.value, other)]
        elif isinstance(other, BatchEncryptedNumber) and len(self.value) == len(other.value):
            value = [v1 + v2 for v1, v2 in zip(self.value, other.value)]
        return BatchEncryptedNumber(value, self.scaling, self.size)

    def __mul__(self, other):
        if not isinstance(self.value, list):
            raise TypeError("In CPU mode, BatchEncryptedNumber.value should be a list of PaillierEncryptedNumber!")
        if len(self.value) != len(other):
            raise NotImplementedError("The shapes of self and other are not equal!")
        value = [v1 * v2 for v1, v2 in zip(self.value, other)]
        return BatchEncryptedNumber(value, self.scaling, self.size)

    def sum(self, sum_idx=None):
        if not sum_idx:
            sum_value = 0
            for v in self.value:
                sum_value = v + sum_value
            return [sum_value]
        else:
            batch_data = [[] for split_idx in range(self.size)]
            for idx in sum_idx:
                split_idx, batch_num = self.get_batch_num_with_idx(idx)
                batch_data[split_idx].append(copy.deepcopy(batch_num))
            res = [0 for split_idx in range(self.size)]
            for split_idx in range(len(batch_data)):
                for v in batch_data[split_idx]:
                    res[split_idx] = v + res[split_idx]
            return res

    def get_batch_num_with_idx(self, idx):
        batch_idx = int(idx / self.size)
        return [idx - batch_idx * self.size, self.value[batch_idx]]

    def split(self, start_idx, end_idx=None):
        if end_idx:
            return BatchEncryptedNumber(self.value[start_idx : end_idx], self.scaling, self.size)
        else:
            return BatchEncryptedNumber(self.value[start_idx : ], self.scaling, self.size)
    def merge(self, other):
        if not isinstance(other, BatchEncryptedNumber):
            raise TypeError("The input of merge function should be BatchENcryptedNumber!")
        self.value.extend(other.value)

class BatchEncryption(object):
    def __init__(self, pub_key=None, private_key=None):
        self.pub_key = pub_key
        self.private_key = private_key

    def cpuBatchEncrypt(self, row_vec, encoder:BatchEncoder, pub_key):
        '''
            Encrypt several row_vec
            Encrypting process: row vector -> several BatchEncodeNumber -> FixPointNumber -> PaillierEncryptedNumber
            Input:
                row_vec: 2-D array, each row should be a list of slot numbers. These slot numbers will be encoded to a BatchEncodeNumber, then encrypt.
                encoder: BatchEncoder
                pub_key: paillier public key
            Return: a BatchEncryptedNumber
        '''
        if pub_key:
            pub_key_used_in_encrypt = pub_key
        else:
            raise NotImplementedError("Please provide a public key when encrypting!")
        # Encode
        batch_encode_number_list = [encoder.batchEncode(slot_number) for slot_number in row_vec]
        pen_with_batch_list = [PaillierEncryptedNumber(pub_key_used_in_encrypt, pub_key_used_in_encrypt.raw_encrypt(v), 0) for v in batch_encode_number_list]
        return BatchEncryptedNumber(pen_with_batch_list, encoder.scaling, encoder.size)
        
    def gpuBatchEncrypt(self, fpn_store:FPN_store, scaling, size, pub_key):
        ''''
            Encrypt several row_vec
            Encrypting process: row vector -> several BatchEncodeNumber -> FixPointNumber -> PaillierEncryptedNumber
            Input:
                fpn_store: FPN_store. Each element of fpn_store is a BatchEncodeNumber. The function will transform BatchEncodeNumbers to BatchEncrytpedNumber.
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
        batch_encoding_values = pen_store.decrypt_without_decode(private_key)
        return batch_encoding_values

    def cpuBatchDecrypt(self, data:BatchEncryptedNumber, encoder:BatchEncoder, private_key):
        batch_decrypt_number_list = [private_key.decrypt_without_decode(v) for v in data.value]
        batch_encoding_values = [encoder.batchDecode(v.encoding, data.scaling, data.size) for v in batch_decrypt_number_list]
        return batch_encoding_values
        

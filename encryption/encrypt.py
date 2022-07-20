import numpy as np
import time, math, copy

from federatedml.FATE_Engine.python.bigintengine.gpu.gpu_store import FPN_store, PEN_store
from federatedml.FATE_Engine.python.BatchPlan.encoding.encoder import BatchEncoder
from federatedml.secureprotol.fate_paillier import PaillierKeypair, PaillierPublicKey, PaillierPrivateKey, PaillierEncryptedNumber


class BatchEncryptedNumber(object):
    def __init__(self, value, scaling, size, lazy_flag=False):
        self.value = None          # store the value of all encrypted BatchEncodeNumber: PEN_store or a list of PaillierEncryptedNumber. Exp: [1 2 3], [4 5 6], [7 8 9]
        self.scaling = scaling
        self.size = size
        '''Use for lazy operation'''
        self.lazy_flag = lazy_flag              # represents if it is in a lazy mode or not
        self.batch_idx_map = None
        self.valid_idx = None
        self.slot_based_value = []          # used in lazy mode. each element is a list, represents a list of PaillierEncryptedNumber. Exp: [1 4 7], [2 5 8], [3 6 9]
        # assign value
        if lazy_flag:
            self.slot_based_value = value
        else:
            self.value = value
    
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

    def batch_add(self, other):
        if not self.lazy_flag:
            if isinstance(other, list) and len(self.value) == len(other):
                value = [v1 + v2 for v1, v2 in zip(self.value, other)]
            elif isinstance(other, BatchEncryptedNumber) and len(self.value) == len(other.value):
                value = [v1 + v2 for v1, v2 in zip(self.value, other.value)]
            else:
                raise TypeError("Please check the type of input, should be list or BatchEncryptedNumber. Please check if the shape of inputs are same or not!")
            return BatchEncryptedNumber(value, self.scaling, self.size)
        # else:
        #     if isinstance(other, list) and len(self.value[0]) == len(other):

    def batch_mul(self, other):
        if not self.lazy_flag:
            self.to_slot_based_value()
        value = []
        for split_idx in range(self.size):
            value.append([self.slot_based_value[split_idx][value_idx] * other[split_idx][value_idx] for value_idx in range(len(other[0]))])
        return BatchEncryptedNumber(value, self.scaling, self.size, lazy_flag=True)

    def batch_sum(self):
        if not self.lazy_flag:
            sum_value = 0
            for v in self.value:
                sum_value = v + sum_value
            return BatchEncryptedNumber([sum_value], self.scaling, self.size)
        else:
            sum_value = []
            for slot_value_list in self.slot_based_value:
                temp = 0
                for v in slot_value_list:
                    temp = v + temp
                sum_value.append([temp])
            return BatchEncryptedNumber(sum_value, self.scaling, self.size, lazy_flag=True)

    def shift_add(self, other, slot_idx, element_idx):
        '''Current version: other should be a BatchEncodeNumber (a big integer)'''
        if not self.lazy_flag:
            self.to_slot_based_value()
        self.slot_based_value[slot_idx][element_idx] = other + self.slot_based_value[slot_idx][element_idx]

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

    def get_batch_value(self, idx):
        '''Return batch number value given element idx'''
        if not self.lazy_flag:
            batch_idx = int(idx / self.size)
            return (self.value[batch_idx], idx - batch_idx * self.size)
        else:
            raise NotImplementedError("Currently only support common mode, instead of lazy operation mode!")


    def get_batch_num_with_idx(self, idx):
        batch_idx = int(idx / self.size)
        return [idx - batch_idx * self.size, self.value[batch_idx]]

    def to_slot_based_value(self):
        '''Transform to slot-based value'''
        if self.lazy_flag:
            # Current BatchEncryptedNumber has been the lazy mode
            return
        if isinstance(self.value, list):
            # cpu model: self.value is a list of PaillierEncryptedNumber
            for _ in range(self.size):
                self.slot_based_value.append([copy.deepcopy(v) for v in self.value])
        elif isinstance(self.value, PEN_store):
            # gpu model: self.value is a PEN_store
            for _ in range(self.size):
                self.slot_based_value.append(self.value.deep_copy())
        else:
            raise NotImplementedError("Current BatchEncryptedNumber.value is not legal!")
        self.lazy_flag = True
        self.value = None

    def split(self, start_idx, end_idx=None):
        if end_idx:
            return BatchEncryptedNumber(self.value[start_idx : end_idx], self.scaling, self.size)
        else:
            return BatchEncryptedNumber(self.value[start_idx : ], self.scaling, self.size)
    def merge(self, other):
        if not isinstance(other, BatchEncryptedNumber):
            raise TypeError("The input of merge function should be BatchENcryptedNumber!")
        if not self.lazy_flag:
            self.value.extend(other.value)
        else:
            for split_idx in range(len(self.slot_based_value)):
                self.slot_based_value[split_idx].extend(other.slot_based_value[split_idx])

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

    def cpuBatchDecrypt(self, data:BatchEncryptedNumber, encoder:BatchEncoder, private_key):
        if not data.lazy_flag:
            batch_decrypt_number_list = [private_key.decrypt_without_decode(v) for v in data.value]
            batch_encoding_values = np.array([encoder.batchDecode(v.encoding, data.scaling, data.size) for v in batch_decrypt_number_list])
            return batch_encoding_values.reshape(batch_encoding_values.size)
        else:
            res = []
            slot_idx = 0
            for slot_value_list in data.slot_based_value:
                batch_decrypt_number_list = [private_key.decrypt_without_decode(v) for v in slot_value_list]
                batch_encoding_values = [encoder.batchDecode(v.encoding, data.scaling, data.size)[slot_idx] for v in batch_decrypt_number_list]
                slot_idx += 1
                res.append(batch_encoding_values)
            return res

    def gpuBatchDecrypt(self, data:BatchEncryptedNumber, encoder:BatchEncoder, private_key):
        if not data.lazy_flag:
            batch_encoding_values = data.value.decrypt_with_batch_decode(private_key, data.scaling, data.size, 
                                                                            encoder.slot_mem_size, encoder.bit_width, encoder.sign_bits)
            return batch_encoding_values
        else:
            res = []
            slot_idx = 0
            split_num = data.slot_based_value[0].store.vec_size
            for slot_value_list in data.slot_based_value:       # each slot_value_list is a pen_store
                batch_decrypt_number_list = slot_value_list.decrypt_with_batch_decode(private_key, data.scaling, data.size, 
                                                                                        encoder.slot_mem_size, encoder.bit_width, encoder.sign_bits)
                batch_encoding_values = [batch_decrypt_number_list[split_idx * encoder.size + slot_idx] for split_idx in range(split_num)]
                slot_idx += 1
                res.append(batch_encoding_values)
            return res

        

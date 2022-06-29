import numpy as np
import copy
from federatedml.FATE_Engine.python.BatchPlan.encoding.encoder import BatchEncoder
from federatedml.FATE_Engine.python.bigintengine.gpu.gpu_store import FPN_store, PEN_store
from federatedml.secureprotol import PaillierEncrypt

def encode():
    '''Init encoder'''
    pub_key_len = 1024
    max_value, element_mem_size, encode_slot_mem, encode_sign_bits = 1, 64, 256, 64
    encoder = BatchEncoder(max_value, element_mem_size, encode_slot_mem, encode_sign_bits, int(pub_key_len / encode_slot_mem))        
    row_vec = np.random.uniform(-1, 1, int(pub_key_len / encode_slot_mem) * 10)     # return 10 batch number after encoding 
    '''Encoder'''
    print("-------------------Original row vector-------------------")
    print(row_vec)
    row_vec = row_vec.reshape(10, int(pub_key_len / encode_slot_mem))
    encode_number_list = [encoder.batchEncode(slot_number) for slot_number in row_vec]
    # print("After batch encoding: " + str(encode_number_list))
    '''Decode'''
    plaintext_list = np.array([encoder.batchDecode(ben, encoder.scaling, int(pub_key_len / encode_slot_mem)) for ben in encode_number_list])
    plaintext_list = plaintext_list.reshape(plaintext_list.size)
    print("-------------------After batch decoding-------------------")
    print(plaintext_list)
    if np.allclose(plaintext_list, row_vec.reshape(row_vec.size)):
        print("\n-------------------Test Pass!-------------------")
    else:
        print("\n-------------------Test Fail-------------------")
        print(plaintext_list == row_vec.reshape(row_vec.size))

# def test_mul():
#     encoder = BatchEncoder(1, 64, 271, 69, 3)        # encode [-1, 1] using 8 bits
#     row_vec_A = np.random.uniform(-1, 1, 3)
#     row_vec_B = np.random.uniform(-1, 1, 3)
#     print("----------------Original vector:----------------")
#     print(row_vec_A)
#     print(row_vec_B)
#     print("----------------Encode:----------------")
#     batch_encode_A = encoder.batchEncode(row_vec_A)
#     scalar_encode_B = encoder.scalarEncode(row_vec_B)
#     print("encode A: ", '0x%x'%batch_encode_A)
#     print("scalar B: ")
#     for scalar in scalar_encode_B:
#         print('0x%x'%scalar)
#     print("----------------Multiplication:----------------")
#     batch_data = [copy.deepcopy(batch_encode_A) for _ in range(len(scalar_encode_B))]
#     for split_idx in range(len(scalar_encode_B)):
#         batch_data[split_idx] = batch_data[split_idx] * scalar_encode_B[split_idx]
#     # shift sum
#     res = batch_data[2]
#     res = res * pow(2, encoder.slot_mem_size)
#     res += batch_data[1]
#     res = res * pow(2, encoder.slot_mem_size)
#     res += batch_data[0]
#     res = res >> 2 * encoder.slot_mem_size
#     print("----------------Decode:----------------")
#     res = encoder.batchDecode(res, encoder.scaling*encoder.scaling, 3)
#     print(res)
#     temp = row_vec_A * row_vec_B
#     temp = temp.sum()
#     print(temp)

def encode_with_gpu():
    '''Init encoder'''
    pub_key_len = 1024
    max_value, element_mem_size, encode_slot_mem, encode_sign_bits = 1, 64, 256, 64
    encoder = BatchEncoder(max_value, element_mem_size, encode_slot_mem, encode_sign_bits, int(pub_key_len / encode_slot_mem))      
    '''Init pub key'''
    encrypter = PaillierEncrypt()
    encrypter.generate_key()
    row_vec = np.random.uniform(-1, 1, int(pub_key_len / encode_slot_mem) * 10)
    print("-------------------Original row vector-------------------")
    print(row_vec)
    '''Encode'''
    fpn_store_with_batch = FPN_store.batch_encode(row_vec, encoder.scaling, encoder.size, encoder.slot_mem_size, encoder.bit_width, encoder.sign_bits, encrypter.public_key)
    '''Decode'''
    plaintext_list = fpn_store_with_batch.batch_decode(encoder.scaling, encoder.size, encoder.slot_mem_size, encoder.bit_width, encoder.sign_bits)
    print("-------------------After batch decoding-------------------")
    print(plaintext_list)
    if np.allclose(plaintext_list, row_vec.reshape(row_vec.size)):
        print("\n-------------------Test Pass!-------------------")
    else:
        print("\n-------------------Test Fail-------------------")
        print(plaintext_list == row_vec.reshape(row_vec.size))

encode_with_gpu()
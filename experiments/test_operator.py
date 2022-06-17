import numpy as np
import time, copy
from federatedml.FATE_Engine.python.BatchPlan.planner.batch_plan import BatchPlan
from federatedml.FATE_Engine.python.BatchPlan.storage.data_store import DataStorage
from federatedml.FATE_Engine.python.BatchPlan.encoding.encoder import BatchEncoder
from federatedml.FATE_Engine.python.BatchPlan.encryption.encrypt import BatchEncryption
from federatedml.FATE_Engine.python.bigintengine.gpu.gpu_store import FPN_store, PEN_store
from federatedml.secureprotol.fate_paillier import PaillierKeypair, PaillierPublicKey, PaillierPrivateKey, PaillierEncryptedNumber

from federatedml.secureprotol import PaillierEncrypt
from federatedml.secureprotol.fixedpoint import FixedPointNumber

import multiprocessing
from joblib import Parallel, delayed

# N_JOBS = multiprocessing.cpu_count()
N_JOBS = 10

def test_encryption():
    row_vec_A = np.random.uniform(-1, 1, 3000000)
    row_vec_A = row_vec_A.astype(np.float64)
    key_generator = PaillierEncrypt()
    key_generator.generate_key()
    print("\n--------------------------------------Encoding Test Report:--------------------------------------")
    # print("\n-------------------CPU:-------------------")
    # # print(N_JOBS)
    # start_time = time.time()
    # # FPN_num = Parallel(n_jobs=40)(delayed(FixedPointNumber.encode)(v) for v in row_vec_A)
    # FPN_num = [FixedPointNumber.encode(v) for v in row_vec_A]
    # stop_time = time.time()
    # print("Duration: ", stop_time - start_time)
    print("\n-------------------CPU with BatchEncode:-------------------")
    start_time = time.time()
    split_num = 1000000
    max_element_num = 3
    encoder = BatchEncoder(1, 64, 322, 86, 3)        # encode [-1, 1] using 8 bits
    row_vec = row_vec_A.reshape(split_num, max_element_num)
    # encode
    encode_number_list = [encoder.batchEncode(slot_number) for slot_number in row_vec]    # a list of BatchEncodeNumber
    stop_time = time.time()
    print("Duration: ", stop_time - start_time)
    print("\n-------------------GPU:-------------------")
    start_time = time.time()
    fpn_store = FPN_store.init_from_arr(row_vec_A, key_generator.public_key.n, key_generator.public_key.max_int)
    stop_time = time.time()
    print("Duration: ", stop_time - start_time)

    print("\n--------------------------------------Encryption Test Report:--------------------------------------")
    # print("\n-------------------CPU:-------------------")
    # start_time = time.time()
    # pen_list = [PaillierEncryptedNumber(key_generator.public_key, key_generator.public_key.raw_encrypt(v.encoding), v.exponent) for v in FPN_num]
    # pen_list = [pen.apply_obfuscator() for pen in pen_list]
    # stop_time = time.time()
    # print("Duration: ", stop_time - start_time)
    # print("\n-------------------CPU with BatchEncode:-------------------")
    # start_time = time.time()
    # pen_list = [PaillierEncryptedNumber(key_generator.public_key, key_generator.public_key.raw_encrypt(v), 0) for v in encode_number_list]
    # pen_list = [pen.apply_obfuscator() for pen in pen_list]
    # stop_time = time.time()
    print("Duration: ", stop_time - start_time)
    print("\n-------------------GPU:-------------------")
    start_time = time.time()
    pen_store = fpn_store.encrypt(key_generator.public_key)
    pen_store = pen_store.obfuscation()
    stop_time = time.time()
    print("Duration: ", stop_time - start_time)
    print("\n-------------------GPU with BatchEncode:-------------------")
    encrypter = BatchEncryption()
    start_time = time.time()
    batch_encrypted_number = encrypter.gpuBatchEncrypt(encode_number_list, encoder.scaling, encoder.size, key_generator.public_key)
    stop_time = time.time()
    print("Duration: ", stop_time - start_time)

def test_matrix_operation():
    row_vec_A = np.random.uniform(-1, 1, 3000000)
    row_vec_B = np.random.uniform(-1, 1, 3000000)
    row_vec_A = row_vec_A.astype(np.float64)
    row_vec_B = row_vec_B.astype(np.float64)
    key_generator = PaillierEncrypt()
    key_generator.generate_key()
    '''CPU'''
    # FPN_num_A = [FixedPointNumber.encode(v) for v in row_vec_A]
    # FPN_num_B = [FixedPointNumber.encode(v) for v in row_vec_B]
    # pen_list_A = [PaillierEncryptedNumber(key_generator.public_key, key_generator.public_key.raw_encrypt(v.encoding), v.exponent) for v in FPN_num_A]
    # pen_list_B = [PaillierEncryptedNumber(key_generator.public_key, key_generator.public_key.raw_encrypt(v.encoding), v.exponent) for v in FPN_num_B]
    '''CPU with BatchEncode'''
    max_element_num = 3
    split_num = int(len(row_vec_A) / max_element_num)
    encoder = BatchEncoder(1, 64, 322, 86, 3)        # encode [-1, 1] using 8 bits
    # encode
    row_vec = row_vec_A.reshape(split_num, max_element_num)
    encode_number_list_A = [encoder.batchEncode(slot_number) for slot_number in row_vec]    # a list of BatchEncodeNumber
    # row_vec = row_vec_B.reshape(split_num, max_element_num)
    # encode_number_list_B = [encoder.batchEncode(slot_number) for slot_number in row_vec]    # a list of BatchEncodeNumber
    # # encrypt
    # pen_list_A = [PaillierEncryptedNumber(key_generator.public_key, key_generator.public_key.raw_encrypt(v), 0) for v in encode_number_list_A]
    # pen_list_B = [PaillierEncryptedNumber(key_generator.public_key, key_generator.public_key.raw_encrypt(v), 0) for v in encode_number_list_B]
    '''GPU'''
    fpn_store_A = FPN_store.init_from_arr(row_vec_A, key_generator.public_key.n, key_generator.public_key.max_int)
    fpn_store_B = FPN_store.init_from_arr(row_vec_B, key_generator.public_key.n, key_generator.public_key.max_int)
    pen_store_A = fpn_store_A.encrypt(key_generator.public_key)
    pen_store_A = pen_store_A.obfuscation()
    # pen_store_B = fpn_store_B.encrypt(key_generator.public_key)
    # pen_store_B = pen_store_B.obfuscation()
    '''GPU with BatchEncode'''
    encrypter = BatchEncryption()
    batch_encrypted_number_A = encrypter.gpuBatchEncrypt(encode_number_list_A, encoder.scaling, encoder.size, key_generator.public_key)
    # batch_encrypted_number_B = encrypter.gpuBatchEncrypt(encode_number_list_B, encoder.scaling, encoder.size, key_generator.public_key)
    print("\n--------------------------------------ADD Test Report:--------------------------------------")
    # print("\n-------------------CPU:-------------------")
    # start_time = time.time()
    # res = [a + b for a, b in zip(pen_list_A, pen_list_B)]
    # stop_time = time.time()
    # print("Duration: ", stop_time - start_time)
    # print("\n-------------------CPU with BatchEncode:-------------------")
    # start_time = time.time()
    # res = [a + b for a, b in zip(pen_list_A, pen_list_B)]
    # stop_time = time.time()
    # print("Duration: ", stop_time - start_time)
    # print("\n-------------------GPU:-------------------")
    # start_time = time.time()
    # res = pen_store_A + pen_store_B
    # stop_time = time.time()
    # print("Duration: ", stop_time - start_time)
    # print("\n-------------------GPU with BatchEncode:-------------------")
    # start_time = time.time()
    # res = batch_encrypted_number_A.value + batch_encrypted_number_B.value
    # stop_time = time.time()
    # print("Duration: ", stop_time - start_time)

    print("\n--------------------------------------MUL Test Report:--------------------------------------")
    print("\n-------------------GPU:-------------------")
    start_time = time.time()
    res = pen_store_A * fpn_store_B
    res = res.sum()
    stop_time = time.time()
    print("Duration: ", stop_time - start_time)
    print("\n-------------------GPU with BatchEncode:-------------------")
    start_time = time.time()
    coefficients = []
    row_vec = row_vec_B.reshape(split_num, max_element_num)
    for split_idx in range(max_element_num):
        coefficient = [v[split_idx] for v in row_vec]
        coefficient = encoder.scalarEncode(coefficient)       # encode
        coefficients.append(coefficient)
    stop_time = time.time()
    print("Encode duration: ", stop_time - start_time)

    start_time = time.time()
    pen_list = batch_encrypted_number_A.value.get_PEN_ndarray()      # a list of PaillierEncryptedNumber
    batch_data = [copy.deepcopy(pen_list) for _ in range(encoder.size)]     # copy
    stop_time = time.time()
    print("Copy duration: ", stop_time - start_time)
    start_time = time.time()
    for split_idx in range(max_element_num):
        batch_data[split_idx] = PEN_store.set_from_PaillierEncryptedNumber(batch_data[split_idx])   # transform to PEN_store
    stop_time = time.time()
    print("p2c duration: ", stop_time - start_time)

    start_time = time.time()
    for split_idx in range(max_element_num):
        batch_data[split_idx] = batch_data[split_idx].mul_with_big_integer(coefficients[split_idx])
        batch_data[split_idx] = batch_data[split_idx].sum()
    stop_time = time.time()
    print("Evaluation duration: ", stop_time - start_time)

test_matrix_operation()
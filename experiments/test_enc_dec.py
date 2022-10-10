import numpy as np
import time, copy, pickle, sys
from federatedml.FATE_Engine.python.BatchPlan.planner.batch_plan import BatchPlan
from federatedml.FATE_Engine.python.BatchPlan.storage.data_store import DataStorage
from federatedml.FATE_Engine.python.BatchPlan.encoding.encoder import BatchEncoder
from federatedml.FATE_Engine.python.BatchPlan.encryption.encrypt import BatchEncryptedNumber, BatchEncryption
from federatedml.FATE_Engine.python.bigintengine.gpu.gpu_store import FPN_store, PEN_store
from federatedml.secureprotol.fate_paillier import PaillierKeypair, PaillierPublicKey, PaillierPrivateKey, PaillierEncryptedNumber

from federatedml.secureprotol import PaillierEncrypt
from federatedml.secureprotol.fixedpoint import FixedPointNumber

def cpu_test():
    '''Init test vector'''
    row_vec_A = np.random.uniform(-1, 1, 1000000)
    row_vec_A = row_vec_A.astype(np.float32)        # use float data type (precision is 23 bits)
    elements_num = row_vec_A.size
    key_generator = PaillierEncrypt()
    key_generator.generate_key()
    '''Init batch encoder'''
    max_value, element_mem_size, encode_slot_mem, encode_sign_bits = 1, 24, 168, 40
    max_element_num = int(1024 / encode_slot_mem)

    split_num = int(np.ceil(len(row_vec_A) / max_element_num))
    encoder = BatchEncoder(max_value, element_mem_size, encode_slot_mem, encode_sign_bits, max_element_num)
    row_vec = np.hstack((row_vec_A, np.zeros(max_element_num * split_num - elements_num)))
    row_vec_used_in_cpu = row_vec.reshape(split_num, max_element_num)
    row_vec_used_in_cpu = row_vec_used_in_cpu.astype(np.float32)

    res = []

    print("\n--------------------------------------Encryption Test Report:--------------------------------------")
    print("\n-------------------CPU:-------------------")
    start_time = time.time()
    # use single process
    CPU_encode_num_list = [FixedPointNumber.encode(v, key_generator.public_key.n, key_generator.public_key.max_int) for v in row_vec_A]
    pen_list = [PaillierEncryptedNumber(key_generator.public_key, key_generator.public_key.raw_encrypt(v.encoding), v.exponent) for v in CPU_encode_num_list]
    stop_time = time.time()
    t1 = stop_time - start_time
    print("Duration: ", stop_time - start_time)
    print("Throughput: ", int(elements_num / (stop_time - start_time)))
    print("\n-------------------CPU with BatchEncode:-------------------")
    start_time = time.time()
    # single process
    CPU_with_batch_encode_number_list = [encoder.batchEncode(slot_number) for slot_number in row_vec_used_in_cpu]    # a list of BatchEncodeNumber
    pen_with_batch_list = [PaillierEncryptedNumber(key_generator.public_key, key_generator.public_key.raw_encrypt(v), 0) for v in CPU_with_batch_encode_number_list]
    stop_time = time.time()
    print("Duration: ", stop_time - start_time)
    t2 = stop_time - start_time
    res.append(t1/t2)
    print("Throughput: ", int(elements_num / (stop_time - start_time)))

    print("\n--------------------------------------Decryption Test Report:--------------------------------------")
    print("\n-------------------CPU:-------------------")
    start_time = time.time()
    # single process
    CPU_decrypt_number_list = [key_generator.privacy_key.decrypt_without_decode(v) for v in pen_list]
    CPU_decode_number_list = [v.decode() for v in CPU_decrypt_number_list]
    stop_time = time.time()
    print("Duration: ", stop_time - start_time)
    t1 = stop_time - start_time
    print("Throughput: ", int(elements_num / (stop_time - start_time)))
    print("\n-------------------CPU with BatchEncode:-------------------")
    start_time = time.time()
    # single process
    CPU_with_batch_decrypt_number_list = [key_generator.privacy_key.decrypt_without_decode(v) for v in pen_with_batch_list]
    CPU_with_batch_decode_number_list = [encoder.batchDecode(v.encoding, encoder.scaling, encoder.size) for v in CPU_with_batch_decrypt_number_list]
    stop_time = time.time()
    print("Duration: ", stop_time - start_time)
    t2 = stop_time - start_time
    res.append(t1/t2)
    print("Throughput: ", int(elements_num / (stop_time - start_time)))

    return res



def gpu_test():
    '''Init test vector'''
    row_vec_A = np.random.uniform(-1, 1, 10000000)
    row_vec_A = row_vec_A.astype(np.float32)        # use float data type (precision is 23 bits)
    elements_num = row_vec_A.size
    key_generator = PaillierEncrypt()
    key_generator.generate_key()
    '''Init batch encoder'''
    max_value, element_mem_size, encode_slot_mem, encode_sign_bits = 1, 24, 168, 40
    max_element_num = int(1024 / encode_slot_mem)

    split_num = int(np.ceil(len(row_vec_A) / max_element_num))
    encoder = BatchEncoder(max_value, element_mem_size, encode_slot_mem, encode_sign_bits, max_element_num)
    row_vec = np.hstack((row_vec_A, np.zeros(max_element_num * split_num - elements_num)))
    row_vec = row_vec.astype(np.float32)

    res = []

    print("\n--------------------------------------Encryption Test Report:--------------------------------------")
    print("\n-------------------GPU:-------------------")
    start_time = time.time()
    fpn_store = FPN_store.init_from_arr(row_vec_A, key_generator.public_key.n, key_generator.public_key.max_int)
    pen_store = fpn_store.encrypt(key_generator.public_key)
    pen_store = pen_store.obfuscation()
    stop_time = time.time()
    t1 = stop_time - start_time
    print("Duration: ", stop_time - start_time)
    print("Throughput: ", int(elements_num / (stop_time - start_time)))
    print("\n-------------------GPU with BatchEncode:-------------------")
    encrypter = BatchEncryption()
    start_time = time.time()
    fpn_store_with_batch = FPN_store.batch_encode(row_vec, encoder.scaling, encoder.size, encoder.slot_mem_size, encoder.bit_width, encoder.sign_bits, key_generator.public_key)
    pen_with_batch_store = encrypter.gpuBatchEncrypt(fpn_store_with_batch, encoder.scaling, encoder.size, key_generator.public_key)
    stop_time = time.time()
    print("Duration: ", stop_time - start_time)
    t2 = stop_time - start_time
    res.append(t1/t2)
    print("Throughput: ", int(elements_num / (stop_time - start_time)))

    print("\n--------------------------------------Decryption Test Report:--------------------------------------")
    print("\n-------------------GPU:-------------------")
    start_time = time.time()
    GPU_decrypt_number_list = pen_store.decrypt_without_decode_to_fp(key_generator.privacy_key)
    stop_time = time.time()
    GPU_decode_number_list = GPU_decrypt_number_list.decode()
    print("Duration: ", stop_time - start_time)
    t1 = stop_time - start_time
    print("Throughput: ", int(elements_num / (stop_time - start_time)))
    print("\n-------------------GPU with BatchEncode:-------------------")
    start_time = time.time()
    GPU_with_batch_decrypt_number_list = pen_with_batch_store.value.decrypt_without_decode_to_fp(key_generator.privacy_key)
    stop_time = time.time()
    GPU_with_batch_decode_number_list = GPU_with_batch_decrypt_number_list.batch_decode(encoder.scaling, encoder.size, encoder.slot_mem_size, encoder.bit_width, encoder.sign_bits)
    print("Duration: ", stop_time - start_time)
    t2 = stop_time - start_time
    res.append(t1/t2)
    print("Throughput: ", int(elements_num / (stop_time - start_time)))

    return res

res = gpu_test()
print("Speedup:", res)


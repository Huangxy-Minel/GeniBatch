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
    row_vec_A = np.random.uniform(-1, 1, 10000)
    elements_num = row_vec_A.size
    row_vec_B = np.random.uniform(-1, 1, 10000)
    row_vec_A = row_vec_A.astype(np.float32)
    row_vec_B = row_vec_B.astype(np.float32)
    key_generator = PaillierEncrypt()
    key_generator.generate_key()
    '''Init encoder'''
    max_value, element_mem_size, encode_slot_mem, encode_sign_bits = 1, 24, 168, 40
    max_element_num = int(1024 / encode_slot_mem)
    split_num = int(np.ceil(len(row_vec_A) / max_element_num))
    encoder = BatchEncoder(max_value, element_mem_size, encode_slot_mem, encode_sign_bits, max_element_num)
    row_vec_used_in_gpu_A = np.hstack((row_vec_A, np.zeros(max_element_num * split_num - elements_num))).astype(np.float32)
    row_vec_used_in_cpu_A = row_vec_used_in_gpu_A.reshape(split_num, max_element_num).astype(np.float32)
    row_vec_used_in_gpu_B = np.hstack((row_vec_B, np.zeros(max_element_num * split_num - elements_num))).astype(np.float32)
    row_vec_used_in_cpu_B = row_vec_used_in_gpu_B.reshape(split_num, max_element_num).astype(np.float32)
    '''GPU Enc'''
    fpn_store_A = FPN_store.init_from_arr(row_vec_A, key_generator.public_key.n, key_generator.public_key.max_int)
    fpn_store_B = FPN_store.init_from_arr(row_vec_B, key_generator.public_key.n, key_generator.public_key.max_int)
    pen_store_A = fpn_store_A.encrypt(key_generator.public_key)
    pen_store_A = pen_store_A.obfuscation()
    pen_store_B = fpn_store_B.encrypt(key_generator.public_key)
    pen_store_B = pen_store_B.obfuscation()
    CPU_pen_list_A = pen_store_A.get_PEN_ndarray()
    CPU_pen_list_B = pen_store_B.get_PEN_ndarray()
    '''GPU Enc with BatchEncode'''
    encrypter = BatchEncryption()
    fpn_with_batch_store_A = FPN_store.batch_encode(row_vec_used_in_gpu_A, encoder.scaling, encoder.size, encoder.slot_mem_size, encoder.bit_width, encoder.sign_bits, key_generator.public_key)
    fpn_with_batch_store_B = FPN_store.batch_encode(row_vec_used_in_gpu_B, encoder.scaling, encoder.size, encoder.slot_mem_size, encoder.bit_width, encoder.sign_bits, key_generator.public_key)
    pen_with_batch_store_A = encrypter.gpuBatchEncrypt(fpn_with_batch_store_A, encoder.scaling, encoder.size, key_generator.public_key)
    pen_with_batch_store_B = encrypter.gpuBatchEncrypt(fpn_with_batch_store_B, encoder.scaling, encoder.size, key_generator.public_key)
    CPU_with_batch_pen_list_A = pen_with_batch_store_A
    CPU_with_batch_pen_list_B = pen_with_batch_store_B
    CPU_with_batch_pen_list_A.value = list(CPU_with_batch_pen_list_A.value.get_PEN_ndarray())
    CPU_with_batch_pen_list_B.value = list(CPU_with_batch_pen_list_B.value.get_PEN_ndarray())

    print("\n--------------------------------------VecAdd Test Report:--------------------------------------")
    print("\n-------------------CPU:-------------------")
    start_time = time.time()
    res = [a + b for a, b in zip(CPU_pen_list_A, CPU_pen_list_B)]
    stop_time = time.time()
    print("Duration: ", stop_time - start_time)
    print("Throughput: ", int(elements_num / (stop_time - start_time)))
    print("\n-------------------CPU with BatchEncode:-------------------")
    start_time = time.time()
    res = CPU_with_batch_pen_list_A.batch_add(CPU_with_batch_pen_list_B)
    stop_time = time.time()
    print("Duration: ", stop_time - start_time)
    print("Throughput: ", int(elements_num / (stop_time - start_time)))

    print("\n--------------------------------------ScalarMul Test Report:--------------------------------------")
    c = np.random.uniform(-1, 1, 1).astype(np.float32)
    print("\n-------------------CPU:-------------------")
    start_time = time.time()
    res = [a*c[0] for a in CPU_pen_list_A]
    stop_time = time.time()
    print("Duration: ", stop_time - start_time)
    print("Throughput: ", int(elements_num / (stop_time - start_time)))
    print("\n-------------------CPU with BatchEncode:-------------------")
    c = encoder.scalarEncode(c)
    start_time = time.time()
    res = [a*c[0] for a in CPU_with_batch_pen_list_A.value]
    stop_time = time.time()
    print("Duration: ", stop_time - start_time)
    print("Throughput: ", int(elements_num / (stop_time - start_time)))

    print("\n--------------------------------------InnerSum Test Report:--------------------------------------")
    print("\n-------------------CPU:-------------------")
    print(len(CPU_pen_list_A), len(CPU_with_batch_pen_list_A.value))
    start_time = time.time()
    res = 0
    for v in CPU_pen_list_A:
        res += v
    stop_time = time.time()
    print("Duration: ", stop_time - start_time)
    print("Throughput: ", int(elements_num / (stop_time - start_time)))
    print("\n-------------------CPU with BatchEncode:-------------------")
    CPU_with_batch_pen_list_A.to_slot_based_value()
    start_time = time.time()
    res = CPU_with_batch_pen_list_A.batch_sum()
    c = 2 ** encode_slot_mem
    temp = res.slot_based_value[0][0]
    for _ in range(CPU_with_batch_pen_list_A.size - 1):
        temp *= c
    stop_time = time.time()
    print("Duration: ", stop_time - start_time)
    print("Throughput: ", int(elements_num / (stop_time - start_time)))

    print("\n--------------------------------------HadmrdProd Test Report:--------------------------------------")
    print("\n-------------------CPU:-------------------")
    CPU_encode_num_list_B = [FixedPointNumber.encode(v) for v in row_vec_B]
    start_time = time.time()
    res = [a * b for a, b in zip(CPU_pen_list_A, CPU_encode_num_list_B)]
    stop_time = time.time()
    print("Duration: ", stop_time - start_time)
    print("Throughput: ", int(elements_num / (stop_time - start_time)))
    print("\n-------------------CPU with BatchEncode:-------------------")
    coefficients_list = []
    for split_idx in range(max_element_num):
        coefficients = [v[split_idx] for v in row_vec_used_in_cpu_B]
        coefficients = encoder.scalarEncode(coefficients)       # encode
        coefficients_list.append(coefficients)
    CPU_with_batch_pen_list_A.to_slot_based_value()
    # begin computation
    start_time = time.time()
    res = CPU_with_batch_pen_list_A.batch_mul(coefficients_list)
    stop_time = time.time()
    print("Duration: ", stop_time - start_time)
    print("Throughput: ", int(elements_num / (stop_time - start_time)))


def gpu_test():
    row_vec_A = np.random.uniform(-1, 1, 10000000)
    elements_num = row_vec_A.size
    row_vec_B = np.random.uniform(-1, 1, 10000000)
    row_vec_A = row_vec_A.astype(np.float32)
    row_vec_B = row_vec_B.astype(np.float32)
    key_generator = PaillierEncrypt()
    key_generator.generate_key()
    '''Init encoder'''
    max_value, element_mem_size, encode_slot_mem, encode_sign_bits = 1, 24, 168, 40
    max_element_num = int(1024 / encode_slot_mem)
    split_num = int(np.ceil(len(row_vec_A) / max_element_num))
    encoder = BatchEncoder(max_value, element_mem_size, encode_slot_mem, encode_sign_bits, max_element_num)
    row_vec_used_in_gpu_A = np.hstack((row_vec_A, np.zeros(max_element_num * split_num - elements_num))).astype(np.float32)
    row_vec_used_in_cpu_A = row_vec_used_in_gpu_A.reshape(split_num, max_element_num).astype(np.float32)
    row_vec_used_in_gpu_B = np.hstack((row_vec_B, np.zeros(max_element_num * split_num - elements_num))).astype(np.float32)
    row_vec_used_in_cpu_B = row_vec_used_in_gpu_B.reshape(split_num, max_element_num).astype(np.float32)
    '''GPU Enc'''
    fpn_store_A = FPN_store.init_from_arr(row_vec_A, key_generator.public_key.n, key_generator.public_key.max_int)
    fpn_store_B = FPN_store.init_from_arr(row_vec_B, key_generator.public_key.n, key_generator.public_key.max_int)
    pen_store_A = fpn_store_A.encrypt(key_generator.public_key)
    pen_store_A = pen_store_A.obfuscation()
    pen_store_B = fpn_store_B.encrypt(key_generator.public_key)
    pen_store_B = pen_store_B.obfuscation()
    '''GPU Enc with BatchEncode'''
    encrypter = BatchEncryption()
    fpn_with_batch_store_A = FPN_store.batch_encode(row_vec_used_in_gpu_A, encoder.scaling, encoder.size, encoder.slot_mem_size, encoder.bit_width, encoder.sign_bits, key_generator.public_key)
    fpn_with_batch_store_B = FPN_store.batch_encode(row_vec_used_in_gpu_B, encoder.scaling, encoder.size, encoder.slot_mem_size, encoder.bit_width, encoder.sign_bits, key_generator.public_key)
    pen_with_batch_store_A = encrypter.gpuBatchEncrypt(fpn_with_batch_store_A, encoder.scaling, encoder.size, key_generator.public_key)
    pen_with_batch_store_B = encrypter.gpuBatchEncrypt(fpn_with_batch_store_B, encoder.scaling, encoder.size, key_generator.public_key)

    print("\n--------------------------------------VecAdd Test Report:--------------------------------------")
    print("\n-------------------GPU:-------------------")
    start_time = time.time()
    res = pen_store_A + pen_store_B
    stop_time = time.time()
    print("Duration: ", stop_time - start_time)
    print("Throughput: ", int(elements_num / (stop_time - start_time)))
    print("\n-------------------GPU with BatchEncode:-------------------")
    start_time = time.time()
    res = pen_with_batch_store_A.value + pen_with_batch_store_B.value
    stop_time = time.time()
    print("Duration: ", stop_time - start_time)
    print("Throughput: ", int(elements_num / (stop_time - start_time)))

    print("\n--------------------------------------ScalarMul Test Report:--------------------------------------")
    c = np.random.uniform(-1, 1, 1).astype(np.float32)
    print("\n-------------------GPU:-------------------")
    start_time = time.time()
    res = pen_store_A * c
    stop_time = time.time()
    print("Duration: ", stop_time - start_time)
    print("Throughput: ", int(elements_num / (stop_time - start_time)))
    print("\n-------------------GPU with BatchEncode:-------------------")
    c = FPN_store.quantization(c, encoder.scaling, encoder.bit_width, encoder.sign_bits, key_generator.public_key) 
    start_time = time.time()
    res = pen_with_batch_store_A.value * c
    stop_time = time.time()
    print("Duration: ", stop_time - start_time)
    print("Throughput: ", int(elements_num / (stop_time - start_time)))

    print("\n--------------------------------------InnerSum Test Report:--------------------------------------")
    print("\n-------------------GPU:-------------------")
    start_time = time.time()
    res = pen_store_A.sum()
    stop_time = time.time()
    print("Duration: ", stop_time - start_time)
    print("Throughput: ", int(elements_num / (stop_time - start_time)))
    print("\n-------------------GPU with BatchEncode:-------------------")
    pen_with_batch_store_A.to_slot_based_value()
    start_time = time.time()
    res = [pen_with_batch_store_A.slot_based_value[idx].sum() for idx in range(pen_with_batch_store_A.size)]
    stop_time = time.time()
    print("Duration: ", stop_time - start_time)
    print("Throughput: ", int(elements_num / (stop_time - start_time)))

    print("\n--------------------------------------HadmrdProd Test Report:--------------------------------------")
    print("\n-------------------GPU:-------------------")
    start_time = time.time()
    res = pen_store_A * fpn_store_B
    stop_time = time.time()
    print("Duration: ", stop_time - start_time)
    print("Throughput: ", int(elements_num / (stop_time - start_time)))
    print("\n-------------------GPU with BatchEncode:-------------------")
    coefficients_list = []
    for split_idx in range(max_element_num):
        coefficients = [v[split_idx] for v in row_vec_used_in_cpu_B]
        coefficients = FPN_store.quantization(coefficients, encoder.scaling, encoder.bit_width, encoder.sign_bits, key_generator.public_key)       # encode
        coefficients_list.append(coefficients)
    pen_with_batch_store_A.to_slot_based_value()
    # begin computation
    start_time = time.time()
    res = [a * b for a, b in zip(pen_with_batch_store_A.slot_based_value, coefficients_list)]
    stop_time = time.time()
    print("Duration: ", stop_time - start_time)
    print("Throughput: ", int(elements_num / (stop_time - start_time)))

cpu_test()

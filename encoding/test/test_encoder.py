import numpy as np
import copy
from federatedml.FATE_Engine.python.BatchPlan.encoding.encoder import BatchEncoder

def encode():
    encoder = BatchEncoder(1, 64, 271, 69)        # encode [-1, 1] using 8 bits
    row_vec = np.random.uniform(-1, 1, 3)
    print("Original row vector: " + str(row_vec))
    batch_encode_number = encoder.batchEncode(row_vec)
    print("After batch encoding: " + str(batch_encode_number.value))
    batch_decode_number = encoder.batchDecode(batch_encode_number)
    print("After batch decoding: " + str(batch_decode_number))

def test_mul():
    encoder = BatchEncoder(1, 64, 271, 69, 3)        # encode [-1, 1] using 8 bits
    row_vec_A = np.random.uniform(-1, 1, 3)
    row_vec_B = np.random.uniform(-1, 1, 3)
    print("----------------Original vector:----------------")
    print(row_vec_A)
    print(row_vec_B)
    print("----------------Encode:----------------")
    batch_encode_A = encoder.batchEncode(row_vec_A)
    scalar_encode_B = encoder.scalarEncode(row_vec_B)
    print("encode A: ", '0x%x'%batch_encode_A)
    print("scalar B: ")
    for scalar in scalar_encode_B:
        print('0x%x'%scalar)
    print("----------------Multiplication:----------------")
    batch_data = [copy.deepcopy(batch_encode_A) for _ in range(len(scalar_encode_B))]
    for split_idx in range(len(scalar_encode_B)):
        batch_data[split_idx] = batch_data[split_idx] * scalar_encode_B[split_idx]
    # shift sum
    res = batch_data[2]
    res = res * pow(2, encoder.slot_mem_size)
    res += batch_data[1]
    res = res * pow(2, encoder.slot_mem_size)
    res += batch_data[0]
    res = res >> 2 * encoder.slot_mem_size
    print("----------------Decode:----------------")
    res = encoder.batchDecode(res, encoder.scaling*encoder.scaling, 3)
    print(res)
    temp = row_vec_A * row_vec_B
    temp = temp.sum()
    print(temp)

test_mul()
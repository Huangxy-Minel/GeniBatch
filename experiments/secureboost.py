import numpy as np
import time, random, multiprocessing
from federatedml.FATE_Engine.python.BatchPlan.planner.batch_plan import BatchPlan, PlanNode
from federatedml.FATE_Engine.python.BatchPlan.storage.data_store import DataStorage
from federatedml.FATE_Engine.python.BatchPlan.encoding.encoder import BatchEncoder
from federatedml.FATE_Engine.python.BatchPlan.encryption.encrypt import BatchEncryptedNumber, BatchEncryption
from federatedml.secureprotol.fate_paillier import PaillierKeypair, PaillierPublicKey, PaillierPrivateKey

from federatedml.secureprotol import PaillierEncrypt
from federatedml.util.fixpoint_solver import FixedPointEncoder

def cal_histogram(grad_enc, hess_enc, slot_mem, split_num, feature_num, matrix_partition):
    histogram = [[[0 for i in range(3)] for _ in range(split_num)] for _ in range(feature_num)]
    rnum = len(matrix_partition)
    # print(rnum)
    for rid in range(rnum):
        grad_value, slot_idx = grad_enc.get_batch_value(rid)
        hess_value, slot_idx = hess_enc.get_batch_value(rid)
        # v1 = grad_value * (1 << (slot_mem * slot_idx))
        # v2 = hess_value * (1 << (slot_mem * slot_idx))
        for fid in range(feature_num):
            split_id = random.randint(0,split_num-1)
            # histogram[fid][split_id][0] += v1
            # histogram[fid][split_id][1] += v2
            histogram[fid][split_id][0] += grad_value
            histogram[fid][split_id][1] += hess_value
            histogram[fid][split_id][2] += 1
    return histogram

def test_secure_matrix_mul():
    data_store = DataStorage()
    myBatchPlan = BatchPlan(data_store, vector_mem_size=512, element_mem_size=24, device_type='CPU', multi_process_flag=True, max_processes=40)
    matrixA = np.random.uniform(-1, 0, (1, 1000000)).astype(np.float32)
    matrixB = np.random.uniform(-1, 0, (1, 1000000)).astype(np.float32)
    matrixC = np.random.uniform(-1, 0, 1000000).astype(np.float32)
    encode_para, batch_scheme = myBatchPlan.generateBatchScheme(['batchSUM'], vec_len=matrixA.shape[1])
    print(encode_para)
    print(batch_scheme)
    myBatchPlan.setBatchScheme(batch_scheme, force_flag=True)
    myBatchPlan.setEncoder(encode_para)

    encrypter = PaillierEncrypt()
    encrypter.generate_key()
    myBatchPlan.setEncrypter()

    '''Enc'''
    encrypted_row_vec_A = myBatchPlan.encrypt(matrixA, batch_scheme[0], encrypter.public_key)
    encrypted_row_vec_B = myBatchPlan.encrypt(matrixB, batch_scheme[0], encrypter.public_key)
    encrypted_row_vec_A = encrypted_row_vec_A.split_to_partition(40)
    encrypted_row_vec_B = encrypted_row_vec_B.split_to_partition(40)
    matrixC_in_partition = BatchEncryptedNumber.align_instance(matrixC, encrypted_row_vec_A)
    # print(len(encrypted_row_vec_A), len(matrixC_in_partition))
    # print(encrypted_row_vec_A)
    split_num = 10
    feature_num = 50

    slot_mem = encode_para[2]

    processor_num = len(matrixC_in_partition)
    pool = multiprocessing.Pool(processes=40)

    time1 = time.time()
    print("start")
    sub_process = [pool.apply_async(cal_histogram, (encrypted_row_vec_A[idx], encrypted_row_vec_B[idx], slot_mem, split_num, feature_num, matrixC_in_partition[idx],)) for idx in range(processor_num)]
    pool.close()
    pool.join()
    res = [p.get() for p in sub_process]
    time2 = time.time()
    print(time2 - time1)

test_secure_matrix_mul()



    
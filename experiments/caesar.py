import numpy as np
import time
from federatedml.FATE_Engine.python.BatchPlan.planner.batch_plan import BatchPlan, PlanNode
from federatedml.FATE_Engine.python.BatchPlan.storage.data_store import DataStorage
from federatedml.FATE_Engine.python.BatchPlan.encoding.encoder import BatchEncoder
from federatedml.FATE_Engine.python.BatchPlan.encryption.encrypt import BatchEncryptedNumber, BatchEncryption
from federatedml.secureprotol.fate_paillier import PaillierKeypair, PaillierPublicKey, PaillierPrivateKey

from federatedml.secureprotol import PaillierEncrypt
from federatedml.util.fixpoint_solver import FixedPointEncoder

def test_secure_matrix_mul():
    row_vec_A = np.random.uniform(-1, 1, (1, 569))
    features = np.random.uniform(-1, 1, (569, 20))
    row_vec_A = row_vec_A.astype(np.float32)
    features = features.astype(np.float32)

    # row_vec_A = row_vec_A / len(row_vec_A[0])

    data_store = DataStorage()
    myBatchPlan = BatchPlan(data_store, vector_mem_size=1024, element_mem_size=24, device_type='CPU', multi_process_flag=True, max_processes=3)
    encode_para, batch_scheme = myBatchPlan.generateBatchScheme(['batchADD', 'batchMUL_SUM'], vec_len=len(row_vec_A[0]))
    myBatchPlan.setBatchScheme(batch_scheme, force_flag=True)
    myBatchPlan.setEncoder(encode_para)

    print(batch_scheme)

    encrypter = PaillierEncrypt()
    encrypter.generate_key()
    myBatchPlan.setEncrypter()

    '''Enc'''
    encrypted_row_vec_A = myBatchPlan.encrypt(row_vec_A, batch_scheme[0], encrypter.public_key)
    encrypted_row_vec_A.to_slot_based_value()

    '''Mul'''
    encrypted_g = []
    for row in features.T:
        encrypted_g.append(PlanNode.cpuBatchMUL_SUM(encrypted_row_vec_A, myBatchPlan.split_row_vec(row), myBatchPlan.encoder))

    '''Get secret sharing'''
    g_sharing_1 = np.random.uniform(-1, 1, len(encrypted_g))
    encrypted_g_sharing_2 = encrypted_g
    for v1, v2 in zip(myBatchPlan.encoder.scalarEncode(g_sharing_1 * (-1)), encrypted_g_sharing_2):
        v2.slot_based_value[-1][0] += int(v1 / myBatchPlan.encoder.scaling * len(row_vec_A[0]))

    '''Dec'''
    res = []
    for row in encrypted_g_sharing_2:
        temp = myBatchPlan.decrypt(row, encrypter.privacy_key)
        res_sum = 0
        for slot_v in temp:
            res_sum += slot_v[0]
        res.append(res_sum)

    res = np.array(res)
    # print(np.array(res))

    '''Re-construct'''
    print(g_sharing_1 + res / len(row_vec_A[0]))

    print(row_vec_A.dot(features) / len(row_vec_A[0]))

test_secure_matrix_mul()
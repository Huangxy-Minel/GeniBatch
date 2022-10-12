import numpy as np
import time, random
from federatedml.FATE_Engine.python.BatchPlan.planner.batch_plan import BatchPlan, PlanNode
from federatedml.FATE_Engine.python.BatchPlan.storage.data_store import DataStorage
from federatedml.FATE_Engine.python.BatchPlan.encoding.encoder import BatchEncoder
from federatedml.FATE_Engine.python.BatchPlan.encryption.encrypt import BatchEncryptedNumber, BatchEncryption
from federatedml.secureprotol.fate_paillier import PaillierKeypair, PaillierPublicKey, PaillierPrivateKey

from federatedml.secureprotol import PaillierEncrypt
from federatedml.util.fixpoint_solver import FixedPointEncoder

def test_secure_matrix_mul():
    np.random.seed(0)
    random.seed(0)
    data_store = DataStorage()
    myBatchPlan = BatchPlan(data_store, vector_mem_size=1024, element_mem_size=24, device_type='CPU', multi_process_flag=False, max_processes=40)
    matrixA = np.random.uniform(-1, 0, (1, 1)).astype(np.float32)
    np.random.seed(1)
    matrixC = np.random.uniform(-1, 0, (1, 1)).astype(np.float32)

    # row_vec_A = row_vec_A / len(row_vec_A[0])
    myBatchPlan.fromMatrix(matrixA, True)
    myBatchPlan.matrixMul([matrixC])
    myBatchPlan.weave()
    batch_scheme = myBatchPlan.getBatchScheme()
    # encode_para, batch_scheme = myBatchPlan.generateBatchScheme(['batchMUL_SUM'], vec_len=len(row_vec_A[0]))
    # myBatchPlan.setBatchScheme(batch_scheme, force_flag=True)
    # myBatchPlan.setEncoder(encode_para)

    # print(encode_para)
    # print(myBatchPlan.encoder.scaling)

    encrypter = PaillierEncrypt()
    encrypter.generate_key()
    myBatchPlan.setEncrypter()

    '''Enc'''
    encrypted_row_vec = myBatchPlan.encrypt(matrixA, batch_scheme[0], encrypter.public_key)
    myBatchPlan.assignEncryptedVector(0, 0, encrypted_row_vec)

    '''Mul'''
    # self.batch_data = self.cpuBatchMUL(batch_encrypted_vec, other_batch_data, encoder)
    outputs = [[]]
    for row in matrixC.T:
        outputs[0].append(PlanNode.cpuBatchMUL_SUM(encrypted_row_vec, myBatchPlan.split_row_vec(row), myBatchPlan.encoder))

    '''Get secret sharing'''
    # g_sharing_1 = np.random.uniform(-1, 1, len(encrypted_g)).astype(np.float32)
    # encrypted_g_sharing_2 = encrypted_g
    # for v1, v2 in zip(myBatchPlan.encoder.scalarEncode(g_sharing_1 * (-1)), encrypted_g_sharing_2):
    #     v2.slot_based_value[-1][0] += int(v1 / myBatchPlan.encoder.scaling)
        # v2.slot_based_value[-1][0] += int(v1 * int(np.log2(len(row_vec_A[0]))) / myBatchPlan.encoder.scaling)
        # v2.slot_based_value[-1][0] += int(v1 / myBatchPlan.encoder.scaling * len(row_vec_A[0]))

    '''Dec'''
    # res = []
    # for row in outputs:
    #     temp = myBatchPlan.decrypt(row, encrypter.privacy_key)
    #     print(temp)
    #     res_sum = 0
    #     for slot_v in temp:
    #         res_sum += sum(slot_v)
    #     res.append(res_sum)

    # res = np.array(res)
    # temp = np.array([0.0 for i in range(11)])
    # for v1, v2 in zip(myBatchPlan.split_row_vec(matrixA[0]), myBatchPlan.split_row_vec(matrixB.T[0])):
    #     temp += np.array(v1) * np.array(v2)
    # print(temp)
    # print(np.array(res))
    # print(row_vec_A.dot(features))

    '''Re-construct'''
    # print(g_sharing_1 + res / len(row_vec_A[0]))

    # print(row_vec_A.dot(features) / len(row_vec_A[0]))

    res = []
    for output in outputs:
        # each output represent the output of one root node
        row_vec = []
        for BatchEncryptedNumber in output:
            real_res = 0
            plain_vec = myBatchPlan.decrypt(BatchEncryptedNumber, encrypter.privacy_key)
            for element_vec in plain_vec:
                real_res += sum(element_vec)
            row_vec.append(real_res)
        res.append(row_vec)
    outputs = res
    row_num, col_num = myBatchPlan.matrix_shape
    output_matrix = np.zeros(myBatchPlan.matrix_shape)
    for row_id in range(row_num):
        output_matrix[row_id, :] = outputs[row_id][0:col_num]
    print("\n-------------------Batch Plan output:-------------------")
    print(output_matrix)
    result = matrixA.dot(matrixC)
    print(result)

test_secure_matrix_mul()
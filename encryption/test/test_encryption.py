import numpy as np
import time
from federatedml.FATE_Engine.python.BatchPlan.planner.batch_plan import BatchPlan
from federatedml.FATE_Engine.python.BatchPlan.storage.data_store import DataStorage
from federatedml.FATE_Engine.python.BatchPlan.encoding.encoder import BatchEncoder
from federatedml.FATE_Engine.python.BatchPlan.encryption.encrypt import BatchEncryptedNumber, BatchEncryption
from federatedml.secureprotol.fate_paillier import PaillierKeypair, PaillierPublicKey, PaillierPrivateKey

from federatedml.secureprotol import PaillierEncrypt
from federatedml.util.fixpoint_solver import FixedPointEncoder

def encrypt_decrypt():
    data_store = DataStorage()
    myBatchPlan = BatchPlan(data_store, vector_mem_size=1024, element_mem_size=24, device_type='CPU', multi_process_flag=True)
    matrixA = np.random.uniform(-1, 1, (1, 100))     # ciphertext
    matrixB = np.random.uniform(-1, 1, (100, 1))     # plaintext

    '''Contruct BatchPlan'''
    myBatchPlan.fromMatrix(matrixA, True)
    myBatchPlan.matrixMul([matrixB])
    print("\n-------------------Batch Plan before weave:-------------------")
    myBatchPlan.printBatchPlan()
    print("\n-------------------Batch Plan after weave:-------------------")
    myBatchPlan.weave()
    batch_scheme = myBatchPlan.getBatchScheme()
    max_element_num, split_num = batch_scheme[0]
    print("Element num in one vector: ", + max_element_num)
    print("Split num: ", + split_num)

    '''Encrypt'''
    print("\n-------------------Encryption:-------------------")
    print("Plaintext: ")
    print(matrixA)
    encrypter = PaillierEncrypt()
    encrypter.generate_key()
    myBatchPlan.setEncrypter()
    encrypted_row_vec = myBatchPlan.encrypt(matrixA, batch_scheme[0], encrypter.public_key)
    '''Decrypt'''
    decrypted_vec = myBatchPlan.decrypt(encrypted_row_vec, encrypter.privacy_key)
    print("-------------------After decryption:-------------------")
    print(decrypted_vec)

def encrypt_decrypt_with_gpu_encode():
    data_store = DataStorage()
    myBatchPlan = BatchPlan(data_store, vector_mem_size=1024, element_mem_size=24, device_type='GPU')
    matrixA = np.random.uniform(-1, 1, (1, 100))     # ciphertext
    matrixB = np.random.uniform(-1, 1, (100, 1))     # plaintext

    '''Contruct BatchPlan'''
    myBatchPlan.fromMatrix(matrixA, True)
    myBatchPlan.matrixMul([matrixB])
    print("\n-------------------Batch Plan before weave:-------------------")
    myBatchPlan.printBatchPlan()
    print("\n-------------------Batch Plan after weave:-------------------")
    myBatchPlan.weave()
    batch_scheme = myBatchPlan.getBatchScheme()
    max_element_num, split_num = batch_scheme[0]
    print("Element num in one vector: ", + max_element_num)
    print("Split num: ", + split_num)

    '''Encrypt'''
    print("\n-------------------Encryption:-------------------")
    print("Plaintext: ")
    print(matrixA)
    encrypter = PaillierEncrypt()
    encrypter.generate_key()
    myBatchPlan.setEncrypter()
    encrypted_row_vec = myBatchPlan.encrypt(matrixA, batch_scheme[0], encrypter.public_key)
    '''Decrypt'''
    decrypted_vec = myBatchPlan.decrypt(encrypted_row_vec, encrypter.privacy_key)
    print("-------------------After decryption:-------------------")
    print(decrypted_vec)

def encrypted_add():
    data_store = DataStorage()
    myBatchPlan = BatchPlan(data_store, vector_mem_size=1024, element_mem_size=32, device_type='GPU', multi_process_flag=True, max_processes=40)
    matrixA = np.random.uniform(-1, 1, (1, 100))     # ciphertext
    matrixB = np.random.uniform(-1, 1, (1, 100))     # plaintext

    '''Contruct BatchPlan'''
    myBatchPlan.fromMatrix(matrixA, True)
    myBatchPlan.matrixAdd([matrixB], [False])
    print("\n-------------------Batch Plan before weave:-------------------")
    myBatchPlan.printBatchPlan()
    print("\n-------------------Batch Plan after weave:-------------------")
    myBatchPlan.weave()
    batch_scheme = myBatchPlan.getBatchScheme()
    max_element_num, split_num = batch_scheme[0]
    print("Element num in one vector: ", + max_element_num)
    print("Split num: ", + split_num)

    '''Encrypt'''
    print("\n-------------------Encryption:-------------------")
    encrypter = PaillierEncrypt()
    encrypter.generate_key()
    myBatchPlan.setEncrypter()
    encrypted_row_vec = myBatchPlan.encrypt(matrixA, batch_scheme[0], encrypter.public_key)

    '''Assign encrypted vector'''
    myBatchPlan.assignEncryptedVector(0, 0, encrypted_row_vec)

    print("\n-------------------Begin to exec Batch Plan.-------------------")
    outputs = myBatchPlan.parallelExec()
    '''Decrypt'''
    outputs = [myBatchPlan.decrypt(output, encrypter.privacy_key) for output in outputs]
    row_num, col_num = myBatchPlan.matrix_shape
    output_matrix = np.zeros(myBatchPlan.matrix_shape)
    for row_id in range(row_num):
        output_matrix[row_id, :] = outputs[row_id][0:col_num]
    print("\n-------------------Batch Plan output:-------------------")
    print(output_matrix)
    print("\n-------------------Numpy output:-------------------")
    result = matrixA + matrixB
    print(result)
    if np.allclose(output_matrix, result):
        print("\n-------------------Test Pass!-------------------")
    else:
        print("\n-------------------Test Fail-------------------")
        print(output_matrix == result)

def encrypted_mul():
    data_store = DataStorage()
    myBatchPlan = BatchPlan(data_store, vector_mem_size=1024, element_mem_size=24, device_type='GPU', multi_process_flag=False, max_processes=None)
    matrixA = np.random.uniform(-1, 1, (1, 3000))     # ciphertext
    matrixB = np.random.uniform(-1, 1, (1, 3000))
    matrixC = np.random.uniform(-1, 1, (3000, 1))     # plaintext
    matrixA = matrixA.astype(np.float32)
    matrixB = matrixB.astype(np.float32)
    matrixC = matrixC.astype(np.float32)

    '''Contruct BatchPlan'''
    myBatchPlan.fromMatrix(matrixA, True)
    myBatchPlan.matrixAdd([matrixB], [False])
    fore_gradient_node = myBatchPlan.root_nodes[0]
    myBatchPlan.matrixMul([matrixC])
    print("\n-------------------Batch Plan before weave:-------------------")
    myBatchPlan.printBatchPlan()
    print("\n-------------------Batch Plan after weave:-------------------")
    myBatchPlan.weave()
    batch_scheme = myBatchPlan.getBatchScheme()
    max_element_num, split_num = batch_scheme[0]
    print("Element num in one vector: ", + max_element_num)
    print("Split num: ", + split_num)
    print("Memory of each slot: ", + myBatchPlan.encoder.slot_mem_size)
    print("Memory of extra sign bits: ", + myBatchPlan.encoder.sign_bits)

    '''Encrypt'''
    print("\n-------------------Encryption:-------------------")
    encrypter = PaillierEncrypt()
    encrypter.generate_key()
    myBatchPlan.setEncrypter()
    time1 = time.time()
    encrypted_row_vec = myBatchPlan.encrypt(matrixA, batch_scheme[0], encrypter.public_key)
    time2 = time.time()
    print("Encrypt costs: ", time2 - time1)

    '''Assign encrypted vector'''
    myBatchPlan.assignEncryptedVector(0, 0, encrypted_row_vec)

    print("\n-------------------Begin to exec Batch Plan.-------------------")
    outputs = myBatchPlan.parallelExec()
    print(fore_gradient_node.getBatchData().slot_based_value)
    '''Decrypt & shift sum'''
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
    print("\n-------------------Numpy output:-------------------")
    result = (matrixA + matrixB).dot(matrixC)
    print(result)
    if np.allclose(output_matrix, result):
        print("\n-------------------Test Pass!-------------------")
    else:
        print("\n-------------------Test Fail-------------------")
        print(output_matrix == result)

def scalar_mul():
    encoder = BatchEncoder(1, 64, 256, 64, 3)        # encode [-1, 1] using 64 bits
    row_vec_A = np.random.uniform(-1, 0, 3)
    row_vec_B = np.random.uniform(-1, 0, 3)
    row_vec_A = np.array([-1, -1, -1])
    print("----------------Original vector:----------------")
    print(row_vec_A)
    # print(row_vec_B)
    print("----------------Encode:----------------")
    batch_encode_A = encoder.batchEncode(row_vec_A)
    # scalar_encode_B = encoder.scalarEncode(row_vec_B)
    print("encode A: ", '0x%x'%batch_encode_A)
    # print("scalar B: ")
    # for scalar in scalar_encode_B:
    #     print('0x%x'%scalar)
    print("----------------Encrypt:----------------")
    key_generator = PaillierEncrypt()
    key_generator.generate_key()
    encrypter = BatchEncryption()
    encrypted_A = encrypter.gpuBatchEncrypt([batch_encode_A], encoder.scaling, encoder.size, key_generator.public_key)
    # shift
    encrypted_A.value = encrypted_A.value.mul_with_big_integer(int(pow(2, encoder.slot_mem_size)))
    encrypted_A.value = encrypted_A.value.mul_with_big_integer(int(pow(2, encoder.slot_mem_size)))

    print("----------------Decrypt:----------------")
    decrypted_A = encrypter.gpuBatchDecrypt(encrypted_A, key_generator.privacy_key)
    decrypted_A[0] = decrypted_A[0] >> encoder.slot_mem_size
    decrypted_A[0] = decrypted_A[0] >> encoder.slot_mem_size
    print("encode A: ", '0x%x'%decrypted_A[0])

def lr_procedure():
    data_store = DataStorage()
    myBatchPlan = BatchPlan(data_store, vector_mem_size=1024, element_mem_size=32)
    self_fore_gradient = np.random.uniform(-1, 1, (1, 300))     # ciphertext
    host_fore_gradient = np.random.uniform(-1, 1, (1, 300))
    self_feature = np.random.uniform(-1, 1, (300, 20))     # plaintext

    '''Contruct BatchPlan'''
    myBatchPlan.fromMatrix(self_fore_gradient, True)
    myBatchPlan.matrixAdd([host_fore_gradient], [False])
    fore_gradient_node = myBatchPlan.root_nodes[0]
    myBatchPlan.matrixMul([self_feature])
    print("\n-------------------Batch Plan before weave:-------------------")
    myBatchPlan.printBatchPlan()
    print("\n-------------------Batch Plan after weave:-------------------")
    myBatchPlan.weave()
    batch_scheme = myBatchPlan.getBatchScheme()
    max_element_num, split_num = batch_scheme[0]
    print("Element num in one vector: ", + max_element_num)
    print("Split num: ", + split_num)
    print("Memory of each slot: ", + myBatchPlan.encoder.slot_mem_size)
    print("Memory of extra sign bits: ", + myBatchPlan.encoder.sign_bits)

    '''Encrypt'''
    print("\n-------------------Encryption:-------------------")
    encrypter = PaillierEncrypt()
    encrypter.generate_key()
    myBatchPlan.setEncrypter()
    encrypted_row_vec = myBatchPlan.encrypt(self_fore_gradient, batch_scheme[0], encrypter.public_key)

    '''Assign encrypted vector'''
    myBatchPlan.assignEncryptedVector(0, 0, encrypted_row_vec)

    print("\n-------------------Begin to exec Batch Plan.-------------------")
    outputs = myBatchPlan.parallelExec()
    '''Decrypt & shift sum'''
    res = []
    for output in outputs:
        # each output represent the output of one root node
        row_vec = []
        for element in output:
            real_res = 0
            for batch_encrypted_number_idx in range(len(element)):
                temp = myBatchPlan.decrypt(element[batch_encrypted_number_idx], encrypter.privacy_key)
                real_res += temp[batch_encrypted_number_idx]
            row_vec.append(real_res)
        res.append(row_vec)
    outputs = res
    '''Calculate bias'''
    bias_middle_grad = fore_gradient_node.getBatchData()
    bias_middle_grad.value = bias_middle_grad.value.sum()
    bias_grad = sum(myBatchPlan.decrypt(bias_middle_grad, encrypter.privacy_key))

    row_num, col_num = myBatchPlan.matrix_shape
    output_matrix = np.zeros(myBatchPlan.matrix_shape)
    for row_id in range(row_num):
        output_matrix[row_id, :] = outputs[row_id][0:col_num]
    print("\n-------------------Batch Plan output:-------------------")
    print("unilateral_gradient: ", output_matrix)
    print("bias gradient: ", bias_grad)
    print("\n-------------------Numpy output:-------------------")
    result = (self_fore_gradient+host_fore_gradient).dot(self_feature)
    plain_bias = (self_fore_gradient+host_fore_gradient).sum()
    print(result)
    print(plain_bias)
    if np.allclose(output_matrix, result):
        print("\n-------------------Test Pass!-------------------")
    else:
        print("\n-------------------Test Fail-------------------")
        print(output_matrix == result)

def shift_sum():
    data_store = DataStorage()
    myBatchPlan = BatchPlan(data_store, vector_mem_size=1024, element_mem_size=32, device_type='CPU', multi_process_flag=True, max_processes=None)
    matrixA = np.random.uniform(-1, 1, (1, 10000))     # ciphertext
    '''Contruct BatchPlan'''
    myBatchPlan.fromMatrix(matrixA, True)
    myBatchPlan.shiftSum([1,1,1])
    myBatchPlan.weave()
    batch_scheme = myBatchPlan.getBatchScheme()
    max_element_num, split_num = batch_scheme[0]
    print("Element num in one vector: ", + max_element_num)
    print("Split num: ", + split_num)
    print("Memory of each slot: ", + myBatchPlan.encoder.slot_mem_size)
    print("Memory of extra sign bits: ", + myBatchPlan.encoder.sign_bits)

    '''Encrypt'''
    print("\n-------------------Encryption:-------------------")
    encrypter = PaillierEncrypt()
    encrypter.generate_key()
    myBatchPlan.setEncrypter()
    encrypted_row_vec = myBatchPlan.encrypt(matrixA, batch_scheme[0], encrypter.public_key)
    batch_size = encrypted_row_vec.size
    batch_scaling = encrypted_row_vec.scaling
    res = BatchEncryptedNumber([[0] for _ in range(batch_size)], batch_scaling, batch_size, lazy_flag=True)
    print(res.slot_based_value)
    for rid in range(matrixA.size):
        v, slot_idx = encrypted_row_vec.get_batch_value(rid)
        res.shift_add(v, slot_idx, 0)
    shift_sum_res = res.slot_based_value[-1][0]
    for slot_idx in range(1, batch_size):
        shift_sum_res = shift_sum_res * (1<<myBatchPlan.encoder.slot_mem_size) + res.slot_based_value[batch_size - 1 - slot_idx][0]
    res = BatchEncryptedNumber([shift_sum_res], batch_scaling, batch_size)
    '''Decrypt'''
    print("\n-------------------Decryption:-------------------")
    slot_based_v_sum = myBatchPlan.decrypt(res, encrypter.privacy_key)
    print(slot_based_v_sum)
    # v_sum = 0
    # for slot_v_list in slot_based_v_sum: v_sum += slot_v_list[0]
    print(slot_based_v_sum[0])
    print(matrixA.sum())

shift_sum()

import numpy as np
from federatedml.FATE_Engine.python.BatchPlan.planner.batch_plan import BatchPlan
from federatedml.FATE_Engine.python.BatchPlan.storage.data_store import DataStorage
from federatedml.FATE_Engine.python.BatchPlan.encoding.encoder import BatchEncoder
from federatedml.FATE_Engine.python.BatchPlan.encryption.encrypt import BatchEncryption
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
    myBatchPlan = BatchPlan(data_store, vector_mem_size=1024, element_mem_size=32, device_type='CPU', multi_process_flag=False)
    matrixA = np.random.uniform(-1, 1, (1, 1000000))     # ciphertext
    matrixB = np.random.uniform(-1, 1, (1, 1000000))     # plaintext

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
    myBatchPlan = BatchPlan(data_store, vector_mem_size=1024, element_mem_size=32, device_type='CPU', multi_process_flag=True, max_processes=None)
    matrixA = np.random.uniform(-1, 1, (1, 10000))     # ciphertext
    matrixB = np.random.uniform(-1, 1, (1, 10000))
    matrixC = np.random.uniform(-1, 1, (10000, 1))     # plaintext
    matrixA = matrixA.astype(np.float32)
    matrixB = matrixB.astype(np.float32)
    matrixC = matrixC.astype(np.float32)

    '''Contruct BatchPlan'''
    myBatchPlan.fromMatrix(matrixA, True)
    # myBatchPlan.matrixAdd([matrixB], [False])
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
    encrypted_row_vec = myBatchPlan.encrypt(matrixA, batch_scheme[0], encrypter.public_key)

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

    row_num, col_num = myBatchPlan.matrix_shape
    output_matrix = np.zeros(myBatchPlan.matrix_shape)
    for row_id in range(row_num):
        output_matrix[row_id, :] = outputs[row_id][0:col_num]
    print("\n-------------------Batch Plan output:-------------------")
    print(output_matrix)
    print("\n-------------------Numpy output:-------------------")
    result = (matrixA).dot(matrixC)
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

def split_sum():
    data_store = DataStorage()
    myBatchPlan = BatchPlan(data_store, vector_mem_size=1024, element_mem_size=32, device_type='CPU', multi_process_flag=True, max_processes=None)
    matrixA = np.random.uniform(-1, 1, (1, 1000))     # ciphertext
    '''Contruct BatchPlan'''
    myBatchPlan.fromMatrix(matrixA, True)
    myBatchPlan.splitSum([[1, 11, 111]])
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
    '''Assign encrypted vector'''
    myBatchPlan.assignEncryptedVector(0, 0, encrypted_row_vec)
    print("\n-------------------Begin to exec Batch Plan.-------------------")
    outputs = myBatchPlan.parallelExec()
    res = []
    for output in outputs:
        row_vec = []
        for split_sum_res in output:
            # reshape
            temp = []
            valid_idx = []
            for idx in range(len(split_sum_res.value)):
                if split_sum_res.value[idx] != 0: 
                    temp.append(split_sum_res.value[idx])
                    valid_idx.append(idx)
            split_sum_res.value = temp
            plaintext = myBatchPlan.decrypt(split_sum_res, encrypter.privacy_key)
            real_res = 0
            for batch_idx, split_idx in enumerate(valid_idx):
                real_res += plaintext[batch_idx*split_sum_res.size + split_idx]
            print(real_res)
            row_vec.append(real_res)
        res.append(row_vec)
    print(matrixA[0][1] + matrixA[0][11] + matrixA[0][111])

split_sum()

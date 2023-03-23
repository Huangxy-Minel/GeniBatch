from federatedml.FATE_Engine.python.bigintengine.gpu.gpu_store import FPN_store, PEN_store
import numpy as np
import time, sys, pickle, copy, math
from federatedml.FATE_Engine.python.BatchPlan.planner.batch_plan import BatchPlan, PlanNode
from federatedml.FATE_Engine.python.BatchPlan.storage.data_store import DataStorage
from federatedml.FATE_Engine.python.BatchPlan.encoding.encoder import BatchEncoder
from federatedml.FATE_Engine.python.BatchPlan.encryption.encrypt import BatchEncryptedNumber, BatchEncryption
from federatedml.secureprotol.fate_paillier import PaillierKeypair, PaillierPublicKey, PaillierPrivateKey

from federatedml.secureprotol import PaillierEncrypt
from federatedml.util.fixpoint_solver import FixedPointEncoder

def encrypt_decrypt():
    data_store = DataStorage()
    myBatchPlan = BatchPlan(data_store, vector_mem_size=1024, element_mem_size=24, device_type='GPU', multi_process_flag=True, max_processes=3)
    matrixA = np.random.uniform(-1, 1, (1, 1000000))     # ciphertext
    matrixB = np.random.uniform(-1, 1, (1000000, 1))     # plaintext

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
    ################Traffic Testing#################
    temp_bytes = pickle.dumps(encrypted_row_vec)
    print(f"Traffic size: {sys.getsizeof(temp_bytes)}")
    raise NotImplementedError("--------------------DEBUG--------------------")
    '''Decrypt'''
    decrypted_vec = myBatchPlan.decrypt(encrypted_row_vec, encrypter.privacy_key)
    print("-------------------After decryption:-------------------")
    print(decrypted_vec[0:3])

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
    myBatchPlan = BatchPlan(data_store, vector_mem_size=1024, element_mem_size=24, device_type='CPU', multi_process_flag=True, max_processes=40)
    matrixA = np.random.uniform(-1, 1, (1, 10000))     # ciphertext
    matrixB = np.random.uniform(-1, 1, (1, 10000))     # plaintext

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
    np.random.seed(0)
    data_store = DataStorage()
    myBatchPlan = BatchPlan(data_store, vector_mem_size=1024, element_mem_size=24, device_type='GPU', multi_process_flag=True, max_processes=40)
    matrixA = np.random.uniform(-1, 1, (1, 5000000)).astype(np.float32)     # ciphertext
    matrixB = np.random.uniform(-1, 1, (1, 5000000)).astype(np.float32)
    matrixC = np.random.uniform(-1, 1, (5000000, 9)).astype(np.float32)     # plaintext
    print(matrixA.dot(matrixC))

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
    # outputs = [[]]
    # for row in matrixC.T:
    #     outputs[0].append(PlanNode.cpuBatchMUL_SUM(encrypted_row_vec, myBatchPlan.split_row_vec(row), myBatchPlan.encoder))
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
    # result = matrixA.dot(matrixC)
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
    sample_num = 5000000
    feature_num = 9
    host_num = 1
    print("Please input 1 for GPU, 0 for CPU:")
    if_GPU = int(input())
    np.random.seed(0)
    self_fore_gradient = np.random.uniform(-1, 1, sample_num).astype(np.float32)     # ciphertext
    host_fore_gradient = np.random.uniform(-1, 1, sample_num).astype(np.float32)

    self_feature = np.random.uniform(0, 1, (sample_num, feature_num))     # plaintext

    '''Contruct BatchPlan'''
    max_value = math.ceil(max(np.max(np.abs(self_fore_gradient)), np.max(np.abs(host_fore_gradient))))
    data_store = DataStorage()
    if not if_GPU:
        '''CPU version'''
        print("CPU version")
        myBatchPlan = BatchPlan(data_store, vector_mem_size=1024, element_mem_size=24, max_value=max_value, device_type='CPU', multi_process_flag=True, max_processes=40)
    else:
        '''GPU version'''
        print("GPU version")
        myBatchPlan = BatchPlan(data_store, vector_mem_size=1024, element_mem_size=24, max_value=max_value, device_type='GPU')


    op_list = ['batchADD' for _ in range(host_num)]
    op_list.append('batchMUL_SUM')

    encode_para, batch_scheme = myBatchPlan.generateBatchScheme(op_list, len(self_fore_gradient))
    max_element_num, split_num = batch_scheme[0]
    myBatchPlan.setBatchScheme(batch_scheme, force_flag=True)
    myBatchPlan.setEncoder(encode_para)
    myBatchPlan.setEncrypter()

    print("\n-------------------Batch Plan Info:-------------------")
    print("encode_slot_mem: ", + encode_para[2])
    print("max_element_num: ", + max_element_num)
    print("Split num: ", + split_num)

    '''Encrypt'''
    time1 = time.time()
    print("\n-------------------Encryption:-------------------")
    encrypter = PaillierEncrypt()
    encrypter.generate_key()
    temp = host_fore_gradient.reshape((1, host_fore_gradient.shape[0]))
    encrypted_row_vec = myBatchPlan.encrypt(temp, batch_scheme[0], encrypter.public_key)
    time2 = time.time()
    print("Encryption time: ", + time2 - time1)

    print("\n-------------------Begin to exec Batch Plan.-------------------")
    # Addition
    if not if_GPU:
        fore_gradients = myBatchPlan.split_row_vec(self_fore_gradient)
        fore_gradients = myBatchPlan.BatchAddParallel(encrypted_row_vec, fore_gradients)
    else:
        fore_gradients = np.hstack((self_fore_gradient, np.zeros(max_element_num * split_num - len(self_fore_gradient))))
        fore_gradients = myBatchPlan.BatchAddGPU(encrypted_row_vec, fore_gradients)
    time1 = time.time()
    print("Addition time: ", + time1 - time2)
    # r1 = self_fore_gradient + host_fore_gradient
    # r2 = myBatchPlan.decrypt(fore_gradients, encrypter.privacy_key)
    # print(r1[0], r1[100], r1[299])
    # print(r2[0], r2[100], r2[299])

    # Multiplication
    bias_middle_grad = BatchEncryptedNumber(fore_gradients.value, fore_gradients.scaling, fore_gradients.size)
    if not if_GPU:
        bias_middle_grad.value = bias_middle_grad.sum()
        unilateral_middle_gradient = myBatchPlan.BatchMulParallel(fore_gradients, self_feature)
    else:
        bias_middle_grad.value = fore_gradients.value.sum()
        unilateral_middle_gradient = myBatchPlan.BatchMulGPU(fore_gradients, self_feature)
    unilateral_middle_gradient.append(bias_middle_grad)
    time2 = time.time()
    print("Multiplication time: ", + time2 - time1)

    '''Decrypt & shift sum'''
    guest_unilateral_gradient = []
    for ben_idx in range(len(unilateral_middle_gradient)):
        if ben_idx != len(unilateral_middle_gradient) - 1:
            # handle unilateral_middle_gradient
            plain_vec = myBatchPlan.decrypt(unilateral_middle_gradient[ben_idx], encrypter.privacy_key)
            real_res = 0
            for element_vec in plain_vec:
                real_res += sum(element_vec)
            guest_unilateral_gradient.append(real_res)
        else:
            # handle bias_middle_grad
            guest_unilateral_gradient.append(sum(myBatchPlan.decrypt(unilateral_middle_gradient[ben_idx], encrypter.privacy_key)))
    time1 = time.time()
    print("Decryption time: ", + time1 - time2)

    print("\n-------------------Result of ciphertexts:-------------------")
    print(np.array(guest_unilateral_gradient))

    '''Result of plaintexts'''
    print("\n-------------------Result of plaintexts:-------------------")
    res = (self_fore_gradient + host_fore_gradient).dot(self_feature)
    bias = (self_fore_gradient + host_fore_gradient).sum()
    res = np.hstack((res, bias))
    print(res)

    if np.allclose(guest_unilateral_gradient, res):
        print("\n-------------------Test Pass!-------------------")
    else:
        print("\n-------------------Test Fail-------------------")
        print(guest_unilateral_gradient == res)


    # res = []
    # for output in outputs:
    #     # each output represent the output of one root node
    #     row_vec = []
    #     for element in output:
    #         real_res = 0
    #         for batch_encrypted_number_idx in range(len(element)):
    #             temp = myBatchPlan.decrypt(element[batch_encrypted_number_idx], encrypter.privacy_key)
    #             real_res += temp[batch_encrypted_number_idx]
    #         row_vec.append(real_res)
    #     res.append(row_vec)
    # outputs = res
    # '''Calculate bias'''
    # bias_middle_grad = fore_gradient_node.getBatchData()
    # bias_middle_grad.value = bias_middle_grad.value.sum()
    # bias_grad = sum(myBatchPlan.decrypt(bias_middle_grad, encrypter.privacy_key))

    # row_num, col_num = myBatchPlan.matrix_shape
    # output_matrix = np.zeros(myBatchPlan.matrix_shape)
    # for row_id in range(row_num):
    #     output_matrix[row_id, :] = outputs[row_id][0:col_num]
    # print("\n-------------------Batch Plan output:-------------------")
    # print("unilateral_gradient: ", output_matrix)
    # print("bias gradient: ", bias_grad)
    # print("\n-------------------Numpy output:-------------------")
    # result = (self_fore_gradient+host_fore_gradient).dot(self_feature)
    # plain_bias = (self_fore_gradient+host_fore_gradient).sum()
    # print(result)
    # print(plain_bias)
    # if np.allclose(output_matrix, result):
    #     print("\n-------------------Test Pass!-------------------")
    # else:
    #     print("\n-------------------Test Fail-------------------")
    #     print(output_matrix == result)

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

def gpu_mul():
    # matrixA = np.random.uniform(-1, 1, 1000000)
    matrixA = np.array([0.5 for i in range(1000000)])
    matrixB = np.random.uniform(0, 1, (1000000, 50))     # plaintext
    encrypter = PaillierEncrypt()
    encrypter.generate_key()
    '''Encrypt'''
    enc_forward_grad = PEN_store.init_from_arr(matrixA, encrypter.get_public_key()).obfuscation()
    fixed_point_encoder = FixedPointEncoder()
    matrixB = fixed_point_encoder.encode(matrixB)
    time1 = time.time()
    temp = enc_forward_grad.protective_r_dot(np.ascontiguousarray(matrixB.T))
    time2 = time.time()
    print(time2 - time1)

def test_para_add():
    matrixA = np.random.uniform(-1, 1, (1, 1000))     # ciphertext
    matrixB = np.random.uniform(-1, 1, 1000)
    matrixC = np.random.uniform(0, 1, (1000, 50))     # plaintext
    '''Construct BatchPlan'''
    data_store = DataStorage()
    myBatchPlan = BatchPlan(data_store, vector_mem_size=1024, element_mem_size=24, max_value=1, device_type='CPU', multi_process_flag=True, max_processes=2)

    encode_para, batch_scheme = myBatchPlan.generateBatchScheme(['batchADD', 'batchMUL_SUM'], vec_len=matrixA.shape[1])
    myBatchPlan.setBatchScheme(batch_scheme, force_flag=True)
    myBatchPlan.setEncoder(encode_para)
    myBatchPlan.setEncrypter()

    cipher = PaillierEncrypt()
    cipher.generate_key()

    enc_A = myBatchPlan.encrypt(matrixA, batch_scheme[0], cipher.public_key)
    split_B = myBatchPlan.split_row_vec(matrixB)
    res = myBatchPlan.BatchAddParallel(enc_A, split_B)
    res = myBatchPlan.BatchMulParallel(res, matrixC)

    dec = []
    for row in res:
        temp = myBatchPlan.decrypt(row, cipher.privacy_key)
        res_sum = 0
        for slot_v in temp:
            res_sum += slot_v[0]
        dec.append(res_sum)
    print(np.array(dec))
    print((matrixA + matrixB).dot(matrixC))

lr_procedure()

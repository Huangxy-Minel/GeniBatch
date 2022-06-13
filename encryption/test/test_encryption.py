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
    myBatchPlan = BatchPlan(data_store, vector_mem_size=1024, element_mem_size=64)
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
    decrypted_vec = myBatchPlan.decrypt(encrypted_row_vec, encrypter.privacy_key)
    print("After decryption: ")
    print(decrypted_vec)


encrypt_decrypt()
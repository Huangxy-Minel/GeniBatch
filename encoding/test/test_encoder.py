import numpy as np
from federatedml.FATE_Engine.python.BatchPlan.encoding.encoder import BatchEncoder

def encode():
    encoder = BatchEncoder(1, 64, 200)        # encode [-1, 1] using 8 bits
    row_vec = np.random.uniform(-1, 1, 5)
    print("Original row vector: " + str(row_vec))
    batch_encode_number = encoder.batchEncode(row_vec)
    print("After batch encoding: " + str(batch_encode_number.value))
    batch_decode_number = encoder.batchDecode(batch_encode_number)
    print("After batch decoding: " + str(batch_decode_number))

encode()
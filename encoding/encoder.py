import numpy as np
class BatchEncodeNumber(object):
    def __init__(self, value, scaling, size):
        self.value = value
        self.scaling = scaling
        self.size = size

class BatchEncoder(object):
    '''
        Encode a list of number to a batch number
    '''
    def __init__(self, max_value, bit_width, slot_mem_size, sign_length):
        self.max_value = max_value
        self.bit_width = bit_width
        self.slot_mem_size = slot_mem_size
        self.sign_bits = sign_length

    def quantize(self, row_vec, scaling):
        if isinstance(row_vec, list):
            row_vec = np.array(row_vec)
        og_sign = np.sign(row_vec)      
        uns_row_vec = row_vec * og_sign     # abs
        zeros_idx = np.where(uns_row_vec == 0)[0]
        quantize_row_vector = np.ceil(uns_row_vec / scaling - 1)
        for idx in zeros_idx:
            quantize_row_vector[idx] = 0
        return og_sign * quantize_row_vector
    
    def de_quantize(self, quantize_row_vector, scaling):
        if isinstance(quantize_row_vector, list):
            quantize_row_vector = np.array(quantize_row_vector)
        return quantize_row_vector * scaling

    def squeeze(self, quantize_row_vector):
        res = 0
        for sign_num in quantize_row_vector:
            res = res << self.slot_mem_size
            complement = int(sign_num) & (pow(2, self.sign_bits + self.bit_width) - 1)   # transform to complement
            res += complement
        return res

    def de_squeeze(self, batch_number:BatchEncodeNumber):
        big_integer = batch_number.value
        quantize_row_vector = []
        for i in range(batch_number.size):
            complement = big_integer & (pow(2, self.sign_bits + self.bit_width) - 1)
            if complement & (1 << (self.sign_bits + self.bit_width - 1)):   # negative
                true_code = ~complement^(pow(2, self.sign_bits + self.bit_width) - 1)
                quantize_row_vector.insert(0, true_code)
            else:
                quantize_row_vector.insert(0, complement)
            big_integer = big_integer >> self.slot_mem_size
        return quantize_row_vector

    def batchEncode(self, row_vec):
        '''Encode a batch of data to a BatchNumber; row_vec should be 1-D array'''
        '''Quantize'''
        scaling = self.max_value / pow(2, self.bit_width - 1)
        quantize_row_vector = self.quantize(row_vec, scaling)
        batch_num = self.squeeze(quantize_row_vector)
        return BatchEncodeNumber(batch_num, scaling, len(row_vec))

    def batchDecode(self, batch_number:BatchEncodeNumber):
        '''Decode a BatchNumber to a batch of data'''
        quantize_row_vector = self.de_squeeze(batch_number)
        row_vector = self.de_quantize(quantize_row_vector, batch_number.scaling)
        return row_vector
        

        
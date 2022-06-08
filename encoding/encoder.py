class BatchEncoder(object):
    '''
        Encode a list of number to a batch number
    '''
    def __init__(self, max_value, bit_width):
        self.max_value = max_value
        self.bit_width = bit_width
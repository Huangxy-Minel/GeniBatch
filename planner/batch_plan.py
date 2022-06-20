from heapq import merge
import numpy as np
import copy, math
from federatedml.FATE_Engine.python.BatchPlan.planner.plan_node import PlanNode
from federatedml.FATE_Engine.python.BatchPlan.storage.data_store import DataStorage
from federatedml.FATE_Engine.python.BatchPlan.encoding.encoder import BatchEncoder
from federatedml.FATE_Engine.python.BatchPlan.encryption.encrypt import BatchEncryption, BatchEncryptedNumber
from federatedml.FATE_Engine.python.bigintengine.gpu.gpu_store import PEN_store

class BatchPlan(object):
    '''
    version 2.1:

    Purpose: Describe the computational topology: DAG.

    Description: 
        Provide common encrypted matrix operator such as Add and Multiplication.
        Create before calculation and lazy operate.
        Weave the current DAG to get a batch sheme, other parties can encrypt the share vector using this batch scheme.
        During execution, the memory of each slot of a batch number is enough. Therefore, no overflow in execution process.

    Note:
        1. Only root nodes can be operated
        2. Node type of encrypted data can only be row vector
        3. After multiplication, a batch encrypted number will be copied for multiple times. Therefore, MUL operation can occur for only once 

    For version 2.1, please call "matrixMul" for only once. 
    Exp:
        Target: ([A]*B+[C]) * D; [·] means homomorphic encryption
        Correct usage: [A]*(B*D) + [C]*D

    TODO: 
        1. Provide primitives which supports to be called for any times
        2. Memory optimization
    '''
    
    def __init__(self, data_storage:DataStorage, vector_mem_size=1024, element_mem_size=64, max_value=1):
        if vector_mem_size != 1024 and vector_mem_size!= 2048 and vector_mem_size != 4096:
            raise NotImplementedError("Vector memory size of batchplan should be 1024 or 2048 or 4096")
        '''Use for BatchPlan'''
        self.root_nodes = []                        # each root node represents one CompTree
        self.opera_nodes_list = []                  # each element in this list represents a level of operation nodes. nodes_list[0] is the lowest level in BatchPlan
        self.vector_nodes_list = []                 # store normal vector nodes
        self.encrypted_vector_nodes = {}            # store encrypted vector nodes
        self.matrix_shape = None                    # represents the shape of the output of this BatchPlan
        # self.encrypted_flag = False                 # represents if output matrix is encrypted or not. default: false
        '''Use for Weaver'''
        self.vector_mem_size = vector_mem_size      # represents memory size of each vector. default: 1024bits
        self.element_mem_size = element_mem_size    # the memory size of one slot number
        self.vector_size = 0                        # num of elements in each node
        self.merge_nodes = []                       # record merge nodes, since only node under merge nodes need be splitted
        self.mul_times = 0
        self.add_times = 0
        '''Use for data storage'''
        self.data_storage = data_storage            # database of the batch plan
        '''Use for interaction with other parties'''
        self.batch_scheme = []                      # list of (max_element_num. split_num). Each element represents the batch plan of a given root node
        '''Use for encoder'''
        self.encoder = None
        self.encode_slot_mem = 0
        self.encode_sign_bits = 0
        self.max_value = 1
        '''Use for encrypter'''
        self.encrypter = None


    def fromMatrix(self, matrixA:np.ndarray, encrypted_flag:bool=False):
        '''
        Initialize Batch Plan from a matrix
        Input:
            matrixA: ndarray, a matrix, should be 2-D array
            encrypted_flag: represents the matrix is encrypted or not
        '''
        if self.matrix_shape != None or self.root_nodes != []:
            raise NotImplementedError("This BatchPlan is not null. Don't use fromMatrix!")
        else:
            '''update to current data store'''
            self.matrix_shape = matrixA.shape
            matrix_id = self.data_storage.addMatrix(matrixA)
        for row_id in range(matrixA.shape[0]):
            '''Create a vector node'''     
            new_node = PlanNode.fromVector(matrix_id, row_id, matrixA.shape[1], 0, self.element_mem_size, encrypted_flag)
            # self.vector_nodes_list.append(new_node)
            self.root_nodes.append(new_node)        # each root node represents a row vector
        # self.encrypted_flag = encrypted_flag
        self.vector_size = matrixA.shape[1]
    
    def matrixAdd(self, matrix_list:list, encrypted_flag_list:list):
        '''
        Primitive for user: Matrix Add
        Input: 
            matrix_list: list of np.ndarray or list of BatchPlan; each matrix should be 2-D array
            encrypted_flag_list: bool list, represents encrypted_flag of each matrix
        Note: 
            matrice in matrix_list can be encrypted
            if type of matrix is BatchPlan, the encrypted_flag in encrypted_flag_list is None
        '''
        # check inputs
        other_BatchPlans = []
        for (matrixB, encrypted_flag) in zip(matrix_list, encrypted_flag_list):
            if isinstance(matrixB, np.ndarray):
                BatchPlanB = BatchPlan(self.data_storage, vector_mem_size=self.vector_mem_size, element_mem_size=self.element_mem_size)
                BatchPlanB.fromMatrix(matrixB, encrypted_flag)
            else:
                raise NotImplementedError("Input of matrixAdd should be ndarray")
            # check shape
            if self.matrix_shape != BatchPlanB.matrix_shape:
                raise NotImplementedError("Input shapes are different!")
            '''Merge two BatchPlans'''
            # self.vector_nodes_list += BatchPlanB.vector_nodes_list
            other_BatchPlans.append(BatchPlanB)

        # Construct computational relationships
        add_nodes_list = []
        for row_id in range(self.matrix_shape[0]):
            new_opera_node = PlanNode.fromOperator("ADD")
            new_opera_node.addChild(self.root_nodes[row_id])       # add one row vector of self as a child of new operator
            new_opera_node.shape = self.root_nodes[row_id].shape   # add operation does not change shape
            max_bit_list = [self.root_nodes[row_id].max_slot_size]
            for iter_plan in other_BatchPlans:
                # if iter_plan.encrypted_flag:
                #     self.encrypted_flag = True
                new_opera_node.addChild(iter_plan.root_nodes[row_id])
                max_bit_list.append(iter_plan.root_nodes[row_id].max_slot_size)
            new_opera_node.max_slot_size = max(max_bit_list) + len(max_bit_list) - 1     # update the current max slot bits
            add_nodes_list.append(new_opera_node)
            self.root_nodes[row_id] = new_opera_node    # replace root node
        self.add_times += 1
        # self.opera_nodes_list.append(add_nodes_list)   # record the operation level

    def matrixMul(self, matrix_list:list):
        '''
        Primitive for user: Matrix Mul
        Input: matrix_list: list of np.ndarray; each matrix should be 2-D array
        Note: matrix in matrix_list must be unenrypted, the encrypted matrix can multiple for only once
        '''
        # check inputs
        other_BatchPlans = []
        last_output_shape = self.matrix_shape   # use to check input shape
        for matrixB in matrix_list:
            if isinstance(matrixB, np.ndarray):
                BatchPlanB = BatchPlan(self.data_storage, vector_mem_size=self.vector_mem_size, element_mem_size=self.element_mem_size)
                BatchPlanB.fromMatrix(matrixB.T)    # transform col vector to row vector
            else:
                raise TypeError("Input of matrixAdd should be ndarray!")
            # check shape
            if last_output_shape[1] != BatchPlanB.matrix_shape[1]:
                raise NotImplementedError("Input shapes are different!")
            '''Merge two BatchPlans'''
            # self.vector_nodes_list += BatchPlanB.vector_nodes_list
            other_BatchPlans.append(BatchPlanB)
            last_output_shape = (last_output_shape[0], BatchPlanB.matrix_shape[0])
        
        for BatchPlanB in other_BatchPlans:
            mul_nodes_list = []
            merge_nodes_list = []
            # Construct computational relationships
            for row_id in range(self.matrix_shape[0]):
                new_merge_node = PlanNode.fromOperator("Merge")
                for col_id in range(BatchPlanB.matrix_shape[0]):           # for each row vector of self, it should be times for col_num of matrixB
                    new_mul_operator = PlanNode.fromOperator("MUL")     # each MUl operator just output 1 element
                    new_mul_operator.addChild(self.root_nodes[row_id])   
                    new_mul_operator.addChild(BatchPlanB.root_nodes[col_id])
                    new_mul_operator.max_slot_size = self.root_nodes[row_id].max_slot_size + BatchPlanB.root_nodes[col_id].max_slot_size   # Adds in MUL matrix operation are ignored
                    new_mul_operator.shape = (1,1)
                    new_merge_node.addChild(new_mul_operator)           # using merge operator splice elements from MUL operators
                    if new_merge_node.max_slot_size < new_mul_operator.max_slot_size:
                        new_merge_node.max_slot_size = new_mul_operator.max_slot_size
                    mul_nodes_list.append(new_mul_operator)
                new_merge_node.shape = (1, BatchPlanB.matrix_shape[0])
                merge_nodes_list.append(new_merge_node)
                self.root_nodes[row_id] = new_merge_node

            # modify current batch plan shape
            self.matrix_shape = (self.matrix_shape[0], BatchPlanB.matrix_shape[0])
        # self.opera_nodes_list.append(mul_nodes_list)
        # self.opera_nodes_list.append(merge_nodes_list)
        self.merge_nodes = merge_nodes_list
        self.mul_times += 1
        

    def weave(self):
        '''
        Use to modify BatchPlan, make sure there is no overflow when executing.
        Note:
            Split nodes below Merge. Any nodes over Merge will not use batch-wise encryption
        '''
        if self.batch_scheme == []:
            # handle BatchPlan which contains MUL operator
            for merge_node in self.merge_nodes:
                merge_node.max_slot_size += math.ceil(math.log2(self.vector_size))      # elements will sum up in matrix mul
                merge_node.max_slot_size += 8 - merge_node.max_slot_size % 8            # transform to multiple of 8
                self.encode_sign_bits = merge_node.max_slot_size - self.element_mem_size    # each element will be quantized using self.element_mem_size, and joint with self.encode_sign_bits for its sign
                self.encode_sign_bits += 8 - self.encode_sign_bits % 8
                self.encode_slot_mem = merge_node.max_slot_size + merge_node.max_slot_size * self.mul_times + self.add_times + self.mul_times * math.ceil(math.log2(self.vector_size))  # the final memory for each slot
                self.encode_slot_mem += 8 - self.encode_slot_mem % 8
                max_element_num = int(self.vector_mem_size / self.encode_slot_mem)     # max element num in one vector
                if self.vector_size > max_element_num:
                    split_num = math.ceil(self.vector_size / max_element_num)   # represents for this CompTree, each vector can be splited to split_num
                    # merge_node.splitTree(max_element_num, split_num)
                    merge_node.recursionUpdateDataIdx(max_element_num, split_num)
                    self.batch_scheme.append((max_element_num, split_num))

            # handle BatchPlan which only contains ADD operator
            if self.merge_nodes == []:
                for root in self.root_nodes:
                    self.encode_sign_bits = root.max_slot_size - self.element_mem_size    # each element will be quantized using self.element_mem_size, and joint with self.encode_sign_bits for its sign
                    self.encode_slot_mem = root.max_slot_size + self.add_times      # the final memory for each slot
                    max_element_num = int(self.vector_mem_size / self.encode_slot_mem)     # max element num in one vector
                    if self.vector_size > max_element_num:
                        split_num = math.ceil(self.vector_size / max_element_num)   # represents for this CompTree, each vector can be splited to split_num
                        # root.splitTree(max_element_num, split_num)
                        root.recursionUpdateDataIdx(max_element_num, split_num)
                        self.batch_scheme.append((max_element_num, split_num))
        else:
            if self.merge_nodes == []:
                if len(self.batch_scheme) != len(self.root_nodes):
                    raise NotImplementedError("The length of batch_scheme is not equal to the num of root nodes!")
                for root_node, (max_element_num, split_num) in zip(self.root_nodes, self.batch_scheme):
                    root_node.recursionUpdateDataIdx(max_element_num, split_num)
            else:
                if len(self.batch_scheme) != len(self.merge_nodes):
                    raise NotImplementedError("The length of batch_scheme is not equal to the num of merge nodes!")
                for merge_node, (max_element_num, split_num) in zip(self.merge_nodes, self.batch_scheme):
                    # merge_node.splitTree(max_element_num, split_num)
                    merge_node.recursionUpdateDataIdx(max_element_num, split_num)
        self.setEncoder()
        self.traverseDAG()      # update node vectors

    def traverseDAG(self):
        '''Update self.vector_nodes_list and self.opera_nodes_list'''
        node_in_level = self.root_nodes
        while len(node_in_level) > 0:
            nodes_next_level = []
            opera_nodes_list = []
            for node in node_in_level:
                if node.encrypted_flag and node.operator == None:     # encrypted vector data
                    matrix_id, row_id, _, _ = node.getDataIdx(0)
                    self.encrypted_vector_nodes[(matrix_id, row_id)] = node
                elif node.operator == None:   # vector data
                    self.vector_nodes_list.append(node)
                else:
                    opera_nodes_list.append(node)   # operation node
                for child in node.children:
                    if child not in nodes_next_level:
                        nodes_next_level.append(child)
            if opera_nodes_list != []:
                self.opera_nodes_list.insert(0, opera_nodes_list)
            node_in_level = nodes_next_level

    def setBatchScheme(self, batch_scheme):
        if len(batch_scheme) != len(self.merge_nodes):
            raise NotImplementedError("The length of batch_scheme is not equal to the num of merge nodes!")
        else:
            self.batch_scheme = batch_scheme

    def getBatchScheme(self):
        return self.batch_scheme

    def assignVector(self):
        '''
            Assign unencrypted vector data to vector nodes
            Note: each vector data should be [array(...), array(...), ...], that means original vector has been splitted to several batch (array)
        '''
        if self.vector_nodes_list == []:
            raise NotImplementedError("Please update vector nodes list firstly!")
        for vec_node in self.vector_nodes_list:
            batch_data = [self.data_storage.getDataFromIdx(split_data_idx) for split_data_idx in vec_node.getDataIdxList()]
            vec_node.setBatchData(batch_data)
    
    def assignEncryptedVector(self, matrix_id, row_id, encrypted_row_vector:BatchEncryptedNumber):
        '''
            Assign encrypted vector data to vector nodes
            Note: encrypted_row_vector should be BatchEncryptedNumber, its value contains [PaillierEncryptedNumber, PaillierEncryptedNumber, ...]; each PaillierEncryptedNumber represents a splitted of original vector (batch encrypted number)
        '''
        if (matrix_id, row_id) in self.encrypted_vector_nodes.keys():
            self.encrypted_vector_nodes[(matrix_id, row_id)].setBatchData(encrypted_row_vector)
        else:
            raise NotImplementedError("Wrong (matrix_id, row_id, slot_start_idx)!")

    def getEncodePara(self):
        return (self.max_value, self.element_mem_size, self.encode_slot_mem, self.encode_sign_bits)

    def setEncoder(self, encode_para=None):
        if encode_para:
            self.max_value, self.element_mem_size, self.encode_slot_mem, self.encode_sign_bits = encode_para
        
        self.encoder = BatchEncoder(self.max_value, self.element_mem_size, self.encode_slot_mem, self.encode_sign_bits, self.batch_scheme[0][0])

    def setEncrypter(self, public_key=None, private_key=None):
        self.encrypter = BatchEncryption(public_key, private_key)

    def encode(self, row_vec):
        '''Batch encode given row vector; row_vec should be 2-D array'''
        if self.encoder == None:
            raise NotImplementedError("Please set encoder before encoding!")
        return self.encoder.batchEncode(row_vec)

    def encrypt(self, row_vec:np.array, row_batch_scheme, pub_key=None):
        '''
            According to the batch scheme, encrypt given row_vec
            Input:
                row_vec: a 2-D array. shape: (1, length of this row vector)
                row_batch_scheme: the batch scheme (max_element_num, split_num) of this row vector
            Return:
                a BatchEncryptedNumber, which contains a PEN_store, stores a list of PaillierEncryptedNumber in GPU
        '''
        # make up zeros
        max_element_num, split_num = row_batch_scheme
        col_num = row_vec.shape[1]
        row_vec = np.hstack((row_vec, np.zeros((1, max_element_num * split_num - col_num))))
        row_vec = row_vec.reshape(split_num, max_element_num)
        # encode
        encode_number_list = [self.encoder.batchEncode(slot_number) for slot_number in row_vec]    # a list of BatchEncodeNumber
        return self.encrypter.gpuBatchEncrypt(encode_number_list, self.encoder.scaling, self.encoder.size, pub_key)

    def decrypt(self, encrypted_data:BatchEncryptedNumber, private_key=None):
        '''
            Decrypt and decode given BatchEncryptedNumber
            Input: 
                encrypted_data: BatchEncryptedNumber, which contains pen_store and current scaling, size for each batch number
                private_key: used in decrypt
        '''
        batch_encoding_values = self.encrypter.gpuBatchDecrypt(encrypted_data, private_key)
        plaintext_list = np.array([self.encoder.batchDecode(ben, encrypted_data.scaling, encrypted_data.size) for ben in batch_encoding_values])
        # reshape
        plaintext_row_vec = plaintext_list.reshape(plaintext_list.size)
        return plaintext_row_vec

    def serialExec(self):
        '''Serially execute each operator, from bottom of the DAG. Call it when use CPUs'''
        self.assignVector()
        outputs = []
        # Execute from operation level 0 
        for nodes_in_level in self.opera_nodes_list:
            for node in nodes_in_level:
                node.serialExec()
        for root in self.root_nodes:
            outputs.append(root.getBatchData())
        return outputs

    def parallelExec(self):
        '''Parallel execute each operator, from bottom of the DAG. Call it when use GPUs'''
        self.assignVector()
        outputs = []
        for one_level_opera_nodes in self.opera_nodes_list:
            for node in one_level_opera_nodes:
                node.parallelExec(self.encoder)
        for root in self.root_nodes:
            outputs.append(root.getBatchData())
        return outputs


    def printBatchPlan(self):
        '''Use to debug'''
        node_in_level = [self.root_nodes]
        level = 0
        while len(node_in_level[level]) > 0:
            print("level: " + str(level))
            nodes_next_level = []
            for root in node_in_level[level]:
                root.printNode()
                for child in root.children:
                    if child not in nodes_next_level:
                        nodes_next_level.append(child)
            node_in_level.append(nodes_next_level)
            print("\n")
            level += 1



        
        
        


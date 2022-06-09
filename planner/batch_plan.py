from heapq import merge
import numpy as np
import copy, math
from federatedml.FATE_Engine.python.BatchPlan.planner.plan_node import PlanNode
from federatedml.FATE_Engine.python.BatchPlan.storage.data_store import DataStorage
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
    
    def __init__(self, data_storage:DataStorage, vector_mem_size=1024, element_mem_size=64):
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
        self.mul_flag = False                       # record if current batch plan includes mul operations or not
        self.merge_nodes = []                       # record merge nodes, since only node under merge nodes need be splitted
        self.mul_times = 0
        self.add_times = 0
        '''Use for data storage'''
        self.data_storage = data_storage            # database of the batch plan
        '''Use for interaction with other parties'''
        self.batch_scheme = []                      # list of (max_element_num. split_num). Each element represents the batch plan of a given root node
        '''Use for encoder'''
        self.encode_element_size = 0


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
        self.mul_flag = True
        self.mul_times += 1
        

    def weave(self):
        '''
        Use to modify BatchPlan, make sure there is no overflow when executing.
        Note:
            Split nodes below Merge. Any nodes over Merge will not use batch-wise encryption
        '''
        if self.batch_scheme == []:
            for merge_node in self.merge_nodes:
                if self.mul_flag:
                    merge_node.max_slot_size += math.ceil(math.log2(self.vector_size))      # elements will sum up in matrix mul
                self.encode_sign_bits = merge_node.max_slot_size - self.element_mem_size    # each element will be quantized using self.element_mem_size, and joint with self.encode_sign_bits for its sign
                self.encode_slot_mem = merge_node.max_slot_size + merge_node.max_slot_size * self.mul_times + self.add_times + self.mul_times * math.ceil(math.log2(self.vector_size))  # the final memory for each slot
                max_element_num = int(self.vector_mem_size / self.encode_slot_mem)     # max element num in one vector
                if self.vector_size > max_element_num:
                    split_num = math.ceil(self.vector_size / max_element_num)   # represents for this CompTree, each vector can be splited to split_num
                    # merge_node.splitTree(max_element_num, split_num)
                    merge_node.recursionUpdateDataIdx(max_element_num, split_num)
                    self.batch_scheme.append((max_element_num, split_num))
        else:
            if len(self.batch_scheme) != len(self.merge_nodes):
                raise NotImplementedError("The length of batch_scheme is not equal to the num of merge nodes!")
            for merge_node, (max_element_num, split_num) in zip(self.merge_nodes, self.batch_scheme):
                # merge_node.splitTree(max_element_num, split_num)
                merge_node.recursionUpdateDataIdx(max_element_num, split_num)
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
    
    def assignEncryptedVector(self, matrix_id, row_id, encrypted_row_vector):
        '''
            Assign encrypted vector data to vector nodes
            Note: encrypted_row_vector should be [PaillierEncryptedNumber, PaillierEncryptedNumber, ...]; each PaillierEncryptedNumber represents a splitted of original vector (batch encrypted number)
        '''
        if (matrix_id, row_id) in self.encrypted_vector_nodes.keys():
            self.encrypted_vector_nodes[(matrix_id, row_id)].setBatchData(encrypted_row_vector)
        else:
            raise NotImplementedError("Wrong (matrix_id, row_id, slot_start_idx)!")

    def serialExec(self):
        '''Serial execution'''
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
        '''Parallel execution'''
        self.assignVector()
        outputs = []
        for one_level_opera_nodes in self.opera_nodes_list:
            self.parallelExecOneLevel(one_level_opera_nodes)
        for root in self.root_nodes:
            outputs.append(root.getBatchData())
        return outputs

    def parallelExecOneLevel(self, one_level_opera_nodes):
        '''Current support 2 children'''
        nodes_type = one_level_opera_nodes[0].operator   # get node type
        print(nodes_type)
        if nodes_type == "ADD":
            '''make up inputs'''
            A_list = []
            B_list = []
            for node in one_level_opera_nodes:
                A_list.extend(node.children[0].getBatchData())
                # TODO: encode 
                B_list.extend(node.children[1].getBatchData())
            # A_list = PEN_store.set_from_PaillierEncryptedNumber(A_list)
            '''calculation'''
            # res = A_list + B_list
            # res = res.get_PEN_ndarray()
            res = np.array(A_list) + np.array(B_list)
            slot_start_idx = 0
            for node in one_level_opera_nodes:
                node.setBatchData(res[slot_start_idx:slot_start_idx + self.batch_scheme[0][0]])
                slot_start_idx += self.batch_scheme[0][0]
        elif nodes_type == "MUL":
            '''make up inputs'''
            A_list = []
            B_list = []
            for node in one_level_opera_nodes:
                A_list.extend(node.children[0].getBatchData())
                # TODO: encode 
                B_list.extend(node.children[1].getBatchData())
            # A_list = PEN_store.set_from_PaillierEncryptedNumber(A_list)
            # res = A_list * B_list
            # res = res.get_PEN_ndarray()
            '''calculation'''
            res = np.array(A_list) * np.array(B_list)
            slot_start_idx = 0
            for node in one_level_opera_nodes:
                sum_list = res[slot_start_idx:slot_start_idx + self.batch_scheme[0][0]]
                slot_start_idx += self.batch_scheme[0][0]
                # sum_list = PEN_store.set_from_PaillierEncryptedNumber(sum_list)
                # sum_list.accumulate_sum()
                # node.setBatchData(sum_list.get_PEN_ndarray())
                node.setBatchData([np.sum(sum_list)])
        elif nodes_type == "Merge":
            for node in one_level_opera_nodes:
                batch_data = []
                for i in range(0, node.size):
                    batch_data.append(node.children[i].getBatchData())
                print(batch_data)
                node.setBatchData(np.array(batch_data))
    
    # def makeupInputs(self, one_level_opera_nodes):
    #     res_list = []
    #     for node in one_level_opera_nodes:
    #         for i in range(0, node.size):




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

    def execBatchPlan(self):
        '''Use to debug'''
        outputs = []
        for root in self.root_nodes:
            outputs.append(root.execNode())
        return outputs


        
        
        


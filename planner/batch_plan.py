import numpy as np
import copy, math
from federatedml.FATE_Engine.python.BatchPlan.planner.plan_node import PlanNode
from federatedml.FATE_Engine.python.BatchPlan.storage.data_store import DataStorage

class BatchPlan(object):
    '''
    version: 2.0
    Update:
    1. Update the data storage and interfaces
    2. Update data struture to support parallel execution
    3. Update computation typology to DAG

    Note: 
    1. Make sure the encryption multiplication only occurs once
    Exp:
        Target: ([A]*B+[C]) * D; [·] means homomorphic encryption
        Correct usage: [A]*(B*D) + [C]*D
    -------------------------------------------------------------
    Description in version 1.1:

    Describe the computational typology.
    Provide common encrypted matrix operator such as Add and Multiplication.
    Create before calculation and lazy operate.

    Note:
    1. Only root nodes can be operated
    2. Node type of encrypted data can only is row vector
    3. Currently, suppose one input of each operator is encrypted
    4. All elements should be integer

    For version 1.1, please call "matrixMul" for only once. 
    Exp:
        Goal: A*B*C
        Incorrect usage:
            myBatchPlan.fromMatrix(A)
            myBatchPlan.matrixMul(B)
            myBatchPlan.matrixMul(C)
        Correct usage:
            myBatchPlan.fromMatrix(A)
            myBatchPlan.matrixMul([B,C])

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
        self.vector_nodes_list = []
        # self.encrypted_vector_node = []
        self.matrix_shape = None                    # represents the shape of the output of this BatchPlan
        # self.encrypted_flag = False                 # represents if output matrix is encrypted or not. default: false
        '''Use for Weaver'''
        self.vector_mem_size = vector_mem_size      # represents memory size of each vector. default: 1024bits
        self.element_mem_size = element_mem_size    # the memory size of one slot number
        self.vector_size = 0                        # num of elements in each node
        self.mul_flag = False
        self.merge_nodes = []
        '''Use for data storage'''
        self.data_storage = data_storage
        '''Use for interaction with other parties'''
        self.batch_scheme = []                      # list of (max_element_num. split_num)


    def fromMatrix(self, matrixA:np.ndarray, encrypted_flag:bool=False):
        '''
        Initialize Batch Plan from a matrix
        Input:
            matrixA: ndarray, a matrix
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
            matrix_list: list of np.ndarray or list of BatchPlan
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
        # self.opera_nodes_list.append(add_nodes_list)   # record the operation level

    def matrixMul(self, matrix_list:list):
        '''
        Primitive for user: Matrix Mul
        Input: matrix_list: list of np.ndarray
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
        

    def weave(self):
        '''
        Use to modify BatchPlan, make sure there is no overflow when executing.
        Note:
            Split nodes below Merge. Any nodes over Merge will not use batch-wise encryption
        '''
        if self.batch_scheme == []:
            for merge_node in self.merge_nodes:
                max_element_num = int(self.vector_mem_size / merge_node.max_slot_size)     # max element num in one vector
                if self.vector_size > max_element_num:
                    # re-calculate slot memory
                    if self.mul_flag and max_element_num != int(self.vector_mem_size / (merge_node.max_slot_size + max_element_num)):
                        max_element_num -= 1
                    split_num = math.ceil(self.vector_size / max_element_num)   # represents for this CompTree, each vector can be splited to split_num
                    merge_node.max_slot_size += max_element_num
                    merge_node.splitTree(max_element_num, split_num)
                    self.batch_scheme.append((max_element_num, split_num))
        else:
            if len(self.batch_scheme) != len(self.merge_nodes):
                raise NotImplementedError("The length of batch_scheme is not equal to the num of merge nodes!")
            for merge_node, (max_element_num, split_num) in zip(self.merge_nodes, self.batch_scheme):
                merge_node.splitTree(max_element_num, split_num)
        self.traverseDAG()      # update node vectors

    def traverseDAG(self):
        '''Update self.vector_nodes_list and self.opera_nodes_list'''
        node_in_level = self.root_nodes
        while len(node_in_level) > 0:
            nodes_next_level = []
            opera_nodes_list = []
            for node in node_in_level:
                if node.operator == None:   # vector data
                    self.vector_nodes_list.append(node)
                    # if node.encrypted_flag:
                    #     self.encrypted_vector_node.append(node)
                else:
                    opera_nodes_list.append(node)   # operation node
                for child in node.children:
                    if child not in nodes_next_level:
                        nodes_next_level.append(child)
            if opera_nodes_list != []:
                self.opera_nodes_list.insert(0, opera_nodes_list)
            node_in_level = nodes_next_level

    def getBatchScheme(self):
        # scheme = []
        # for node in self.encrypted_vector_node:
        #     scheme.append(node.data_idx)
        # return scheme
        return self.batch_scheme

    def assignVector(self):
        '''Assign vector data to vector nodes'''
        if self.vector_nodes_list == []:
            raise NotImplementedError("Please update vector nodes list firstly!")
        for vec_node in self.vector_nodes_list:
            batch_data = self.data_storage.getDataFromIdx(vec_node.getDataIdx())
            vec_node.setBatchData(batch_data)

    def serialExec(self):
        '''Serial execution'''
        self.assignVector()
        outputs = []
        # Execute from operation level 0 
        for nodes_in_level in self.opera_nodes_list:
            for node in nodes_in_level:
                node.serialExec()
        for root in self.root_nodes:
            outputs.append(root.batch_data)
        return outputs

    # def parallelExec(self):
    #     '''Parallel execution'''

    # def parallelExecOneLevel(self, opera_nodes_list):
    #     nodes_type = opera_nodes_list[0].operator
    #     if nodes_type == "ADD":


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


        
        
        


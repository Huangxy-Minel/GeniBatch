import numpy as np
import copy, math
from plan_node import PlanNode

class BatchPlan(object):
    '''
    version: 2.0
    Update:
    1. Update the data storage and interfaces
    2. Update data struture to support parallel execution

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
    
    def __init__(self, vector_mem_size=1024):
        if vector_mem_size != 1024 and vector_mem_size!= 2048 and vector_mem_size != 4096:
            raise NotImplementedError("Vector memory size of batchplan should be 1024 or 2048 or 4096")
        '''Use for BatchPlan'''
        self.root_nodes = []                        # each root node represents one CompTree
        self.matrix_shape = None                    # represents the shape of the output of this BatchPlan
        self.encrypted_flag = False                 # represents if output matrix is encrypted or not. default: false
        self.nodes_list = []                        # each element in this list represents a level of nodes. nodes_list[0] is the lowest level in BatchPlan
        self.data_node_map = {}                     # input_matrix:[plan_nodes]
        '''Use for Weaver'''
        self.vector_mem_size = vector_mem_size      # represents memory size of each vector. default: 1024bits
        self.vector_size = 0

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
            self.matrix_shape = matrixA.shape
            self.data_node_map[matrixA] = []        # create a map space for input matrix to store related plan node
        for vec in matrixA:     
            new_node = PlanNode.fromVector(vec, encrypted_flag)
            self.root_nodes.append(new_node)        # each root node represents a row vector
        self.encrypted_flag = encrypted_flag
        self.vector_size = len(matrixA[0])
        
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
                BatchPlanB = BatchPlan()
                BatchPlanB.fromMatrix(matrixB, encrypted_flag)
            elif isinstance(matrixB, BatchPlan):
                BatchPlanB = matrixB
            else:
                raise NotImplementedError("Input of matrixAdd should be ndarray or BatchPlan")
            # check shape
            if self.matrix_shape != BatchPlanB.matrix_shape:
                raise NotImplementedError("Input shapes are different!")
            other_BatchPlans.append(BatchPlanB)

        # Construct computational relationships
        for row_id in range(self.matrix_shape[0]):
            new_opera_node = PlanNode.fromOperator("ADD")
            new_opera_node.addChild(self.root_nodes[row_id])       # add one row vector of self as a child of new operator
            new_opera_node.shape = self.root_nodes[row_id].shape   # add operation does not change shape
            max_bit_list = [self.root_nodes[row_id].max_element_size]
            for iter_plan in other_BatchPlans:
                if iter_plan.encrypted_flag:
                    self.encrypted_flag = True
                new_opera_node.addChild(iter_plan.root_nodes[row_id])
                max_bit_list.append(iter_plan.root_nodes[row_id].max_element_size)
            new_opera_node.max_element_size = max(max_bit_list) + len(max_bit_list) - 1
            self.root_nodes[row_id] = new_opera_node    # replace root node

    def matrixMul(self, matrix_list:list):
        '''
        Primitive for user: Matrix Mul
        Input: matrix_list: list of np.ndarray
        Note: matrix in matrix_list must be unenrypted
        '''
        # check inputs
        other_BatchPlans = []
        last_output_shape = self.matrix_shape   # use to check input shape
        for matrixB in matrix_list:
            if isinstance(matrixB, np.ndarray):
                BatchPlanB = BatchPlan()
                BatchPlanB.fromMatrix(matrixB.T)    # transform col vector to row vector
            else:
                raise TypeError("Input of matrixAdd should be ndarray!")
            # check shape
            if last_output_shape[1] != BatchPlanB.matrix_shape[1]:
                raise NotImplementedError("Input shapes are different!")
            other_BatchPlans.append(BatchPlanB)
            last_output_shape = (last_output_shape[0], BatchPlanB.matrix_shape[0])
        
        for BatchPlanB in other_BatchPlans:
            # Construct computational relationships
            for row_id in range(self.matrix_shape[0]):
                new_merge_node = PlanNode.fromOperator("Merge")
                for col_id in range(BatchPlanB.matrix_shape[0]):           # for each row vector of self, it should be times for col_num of matrixB
                    new_mul_operator = PlanNode.fromOperator("MUL")     # each MUl operator just output 1 element
                    new_mul_operator.addChild(copy.deepcopy(self.root_nodes[row_id]))   
                    new_mul_operator.addChild(copy.deepcopy(BatchPlanB.root_nodes[col_id]))
                    new_mul_operator.max_element_size = self.root_nodes[row_id].max_element_size + BatchPlanB.root_nodes[col_id].max_element_size   # Adds in MUL matrix operation are ignored
                    new_mul_operator.shape = (1,1)
                    new_merge_node.addChild(new_mul_operator)           # using merge operator splice elements from MUL operators
                    if new_merge_node.max_element_size < new_mul_operator.max_element_size:
                        new_merge_node.max_element_size = new_mul_operator.max_element_size
                new_merge_node.shape = (1, BatchPlanB.matrix_shape[0])
                self.root_nodes[row_id] = new_merge_node

            # modify current batch plan shape
            self.matrix_shape = (self.matrix_shape[0], BatchPlanB.matrix_shape[0])

    def weave(self):
        '''
        Use to modify BatchPlan, make sure there is no overflow when executing.
        '''
        new_root_nodes = []
        for root in self.root_nodes:
            max_element_num = int(self.vector_mem_size / root.max_element_size)     # max element num in one vector
            if self.vector_size > max_element_num:
                split_num = math.ceil(self.vector_size / max_element_num)   # represents for this CompTree, each vector can be splited to split_num
                tail_zero_num = split_num * max_element_num - self.vector_size
                new_root_nodes.extend(root.splitTree(max_element_num, split_num, tail_zero_num))
            else:
                new_root_nodes.append(root)
        self.root_nodes = new_root_nodes

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


        
        
        


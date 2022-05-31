from heapq import merge
import numpy as np
import copy, math

class PlanNode(object):
    '''
    versionL 2.0
    Note:
        Upate the typology to DAG
    version: 1.1
    Node in the computational typology, which is called BatchPlan.
    Two types of Node: vector & operator
    Vector type: only row vector

    TODO: Better encapsulation. Specify which variables are inaccessible, exp: PlanNode._operatror
    '''

    def __init__(self, operator=None, batch_data=None, max_slot_size=0, encrypted_flag=False):
        '''Node properties'''
        self.operator = operator            # ADD, MUL or Merge
        self.batch_data = batch_data        # vector data of this node; type: np.ndarray
        self.data_idx = ()
        '''Vector properties'''
        self.max_slot_size = max_slot_size          # represent max memory bits of each slot
        self.shape = (0,0)
        '''DAG Graph'''
        # self.parent = []             # The parent node, for root node, parent = []
        self.children = []           # A list of children
        self.size = 0                # children num
        '''Node attributes'''
        self.state = 0               # represents if the output data has been prepared or not. 0: not finished; 1: finished
        self.encrypted_flag = encrypted_flag             # represents if batch_data is encrypted or not. default: false

    @staticmethod
    def fromVector(matrix_id, vector_id, vector_len, slot_start_idx, slot_mem, encrypted_flag:bool):
        '''
        Create a node from a vector but do not set the batch data. Only allow the row vector
        Input:

        '''
        new_node = PlanNode(max_slot_size=slot_mem, encrypted_flag=encrypted_flag)
        new_node.data_idx = (matrix_id, vector_id, slot_start_idx, vector_len)
        new_node.shape = (1, vector_len)
        new_node.encrypted_flag = encrypted_flag
        return new_node

    @staticmethod
    def fromOperator(operator:str):
        '''
        Create a node from a operator (ADD, MUL or Merge)
        '''
        if operator == "ADD" or operator == "MUL" or operator == "Merge":
            new_node = PlanNode(operator=operator)
        return new_node

    def getBatchData(self):
        if self.state == 0:
            raise NotImplementedError("Current node has not gotten the batch data, please check the execution logic or data assignment process!")
        return self.batch_data

    def getDataIdx(self):
        return self.data_idx

    def getShape(self):
        return self.shape

    def getVectorLen(self):
        return self.shape[1]

    def setBatchData(self, data):
        self.batch_data = data
        self.state = 1

    # def changeParent(self, parent_node):
    #     '''
    #     Change current parent for this node
    #     Input:
    #         parent_node: class PlanNode
    #     '''
    #     if self.parent == None:
    #         raise NotImplementedError("Current node is root, do not call this function!")
    #     self.parent = parent_node

    # def addParent(self, parent_node):
    #     '''
    #     Add a parent for current root node in Batch Plan
    #     Input:
    #         parent_node: class PlanNode
    #     '''
    #     self.parent.append(parent_node)
    
    def addChild(self, child_node):
        '''
        Add a child node for current root node
        Input:
            child_node: class PlanNode
        Note: Only operation node can add a child
        '''
        # if self.parent != []:
        #     raise NotImplementedError("Current node is not root, all operations should at root node!")
        if self.operator == None:
            raise NotImplementedError("Only operation node can add a child!")
        self.children.append(child_node)
        # child_node.addParent(self)
        if child_node.encrypted_flag:
            self.encrypted_flag = True      # For any operations, if one child is encrypted, the output of this operation node must be encrypted
        self.size += 1

    def delChild(self, child_node):
        '''
        Delete a specified child node
        Input:
            child_node: class PlanNode
        Return: bool, True - Success, False - can not find child_node
        '''
        if child_node in self.children:
            child_node.parent = None
            self.children.remove(child_node)
            return True
        else:
            return False

    
    def splitTree(self, max_element_num, split_num):
        '''
        Split CompTree, make sure vector size of each node is smaller than max_element_num
        Properties:
            1. Only one level of CompTree contains Merge Operator and the operation above Merge node can only be ADD
            2. Only one level of sub-CompTree (root node is Merge Operator) contains MUL Operator
        For Merge Operator, keep same
        For MUL Operator, split to multi-MUL + one ADD
        Call from a merge node
        Input:
            max_element_num: int
        Return:
            list of PlanNode, represents several CompTree
        '''
        '''Split node start at Merge node'''
        '''delete mul nodes below merge node firstly'''
        mul_nodes = []          # store the children (mul nodes) of the merge node
        for mul_node in self.children:
            mul_nodes.append(mul_node)
        self.children = []
        self.size = 0
        '''
        Insight: Mul nodes of a fixed Merge node share the same row vector
        Therefore, split the row vector only once
        '''
        element_idx = 0         # means all vectors start at element_idx
        for i in range(len(mul_nodes)):
            new_add_node = PlanNode.fromOperator("ADD")     # One MUL will split to one ADD and multi-MUL
            new_add_node.shape = (1,1)
            self.addChild(new_add_node)
        for i in range(split_num):
            '''split row vector secondly'''
            row_vec_node = copy.deepcopy(mul_nodes[0].children[0])
            row_vec_node.recursionUpdate(max_element_num, element_idx)
            '''split mul operators and modify typology thirdly'''
            add_node_idx = 0
            for mul_node in mul_nodes:
                new_mul_node = PlanNode.fromOperator("MUL")
                new_mul_node.shape = (1,1)
                new_mul_node.addChild(row_vec_node)
                for child_idx in range(1, mul_node.size):
                    child = copy.deepcopy(mul_node.children[child_idx])        # get one child of original mul node. the child node must be vector node (col vector)
                    child.recursionUpdate(max_element_num, element_idx)
                    new_mul_node.addChild(child)
                self.children[add_node_idx].addChild(new_mul_node)
                add_node_idx += 1
            element_idx += max_element_num      # update start idx


    # def recursionMakeUpZeros(self, tail_zero_num):
    #     if self.operator == None:   # vector node
    #         self.batch_data = np.hstack((self.batch_data, np.zeros((1, tail_zero_num))))
    #     else:
    #         for child in self.children:
    #             child.recursionMakeUpZeros(tail_zero_num)


    def recursionUpdate(self, max_element_num, element_idx):
        if self.operator == None:   # vector node
            self.shape = (1, max_element_num)
            matrix_id, vector_id = self.data_idx[0], self.data_idx[1]
            self.data_idx = (matrix_id, vector_id, element_idx, max_element_num)
            # self.batch_data = self.batch_data[element_idx : element_idx + max_element_num - 1]
        else:   # operator node
            if self.operator == "ADD":
                self.shape = (1, max_element_num)
            else:
                raise NotImplementedError("recursionUpdate should only be called at ADD or Vector!")
            for child in self.children:
                child.recursionUpdate(max_element_num, element_idx)

    def printNode(self):
        '''Use to debug'''
        if self.operator != None:
            print(self.operator, end='  ')
        else:
            print(self.encrypted_flag, end='  ')

    def execNode(self):
        '''
        Execute node function recursively
        '''
        if self.state == 1:             # for vector node or operator node which has been executed
            return self.batch_data
        else:                           # node has not operated
            if self.operator == "ADD":
                self.batch_data = copy.deepcopy(self.children[0].execNode())
                for i in range(1, self.size):
                    self.batch_data = self.batch_data + self.children[i].execNode()
            elif self.operator == "MUL":
                self.batch_data = copy.deepcopy(self.children[0].execNode())
                for i in range(1, self.size):
                    self.batch_data = self.batch_data * self.children[i].execNode()
                    self.batch_data = self.batch_data.sum()
            elif self.operator == "Merge":
                self.batch_data = np.zeros(self.shape)
                for i in range(0, self.size):
                   self.batch_data[0][i] = self.children[i].execNode() 
            else: 
                raise NotImplementedError("Invalid operator node!")
            self.state = 1              # change state
            return self.batch_data

    def serialExec(self):
        '''
        Execute node only base on its children, not recursively
        '''
        if self.operator == "ADD":
            self.batch_data = copy.deepcopy(self.children[0].getBatchData())
            # print("ADD inputs")
            # print(self.children[0].getBatchData())
            # print(self.children[1].getBatchData())
            for i in range(1, self.size):
                self.batch_data = self.batch_data + self.children[i].getBatchData()
        elif self.operator == "MUL":
            self.batch_data = copy.deepcopy(self.children[0].getBatchData())
            # print("MUL inputs")
            # print(self.children[0].getBatchData())
            # print(self.children[1].getBatchData())
            for i in range(1, self.size):
                self.batch_data = self.batch_data * self.children[i].getBatchData()
                self.batch_data = self.batch_data.sum()
            # print("Mul output " + str(self.batch_data))
        elif self.operator == "Merge":
            self.batch_data = np.zeros(self.shape)
            for i in range(0, self.size):
                self.batch_data[0][i] = self.children[i].getBatchData()
        else: 
            raise NotImplementedError("Invalid operator node!")
        self.state = 1
        return self.batch_data

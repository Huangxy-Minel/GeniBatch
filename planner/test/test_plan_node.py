import sys, math
sys.path.append("..")
sys.path.append("../..")
from plan_node import PlanNode
from batch_plan import BatchPlan
from storage.data_store import DataStorage

import numpy as np

def init_batch_plan_from_matrix():
    data_store = DataStorage()
    myBatchPlan = BatchPlan(data_store, vector_mem_size=1024, element_mem_size=64)
    matrixA = np.random.rand(4,3)
    print(matrixA)
    myBatchPlan.fromMatrix(matrixA, True)
    print("matrix shape: " + str(myBatchPlan.matrix_shape))
    print("BatchPlan typology: ")
    myBatchPlan.printBatchPlan()
    myBatchPlan.assignVector()
    print("each root node:")
    for root in myBatchPlan.root_nodes:
        print("batch data: " + str(root.getBatchData()))
        print("batch shape: " + str(root.getShape()))

def matrix_add():
    data_store = DataStorage()
    myBatchPlan = BatchPlan(data_store, vector_mem_size=1024, element_mem_size=64)
    matrix_list = []
    for i in range(4):
        matrix_list.append(np.random.rand(4,3))
    print("\n-------------------Input Matrix: -------------------")
    for matrix in matrix_list:
        print(matrix)
    myBatchPlan.fromMatrix(matrix_list[0], True)
    myBatchPlan.matrixAdd([matrix_list[1], matrix_list[2]], [False, False])
    myBatchPlan.matrixAdd([matrix_list[3]], [False])
    print("BatchPlan typology")
    myBatchPlan.printBatchPlan()
    print("\n-------------------Begin to exec Batch Plan.-------------------")
    outputs = myBatchPlan.serialExec()
    row_num, col_num = myBatchPlan.matrix_shape
    output_matrix = np.zeros(myBatchPlan.matrix_shape)
    for row_id in range(row_num):
        output_matrix[row_id, :] = outputs[row_id]
    print("\n-------------------Batch Plan output:-------------------")
    print(output_matrix)
    print("\n-------------------Numpy output:-------------------")
    result = matrix_list[0]
    for i in range(1, len(matrix_list)):
        result += matrix_list[i]
    print(result)

    if np.allclose(output_matrix, result):
        print("\n-------------------Test Pass!-------------------")
    else:
        print("\n-------------------Test Fail-------------------")
        print(output_matrix == result)

def matrix_mul():
    data_store = DataStorage()
    myBatchPlan = BatchPlan(data_store, vector_mem_size=1024, element_mem_size=64)
    matrixA = np.random.rand(1, 10000)
    print("\n-------------------Matrix A: -------------------")
    print(matrixA)
    matrixB = np.random.rand(10000,20)
    print("\n-------------------Matrix B: -------------------")
    print(matrixB)
    myBatchPlan.fromMatrix(matrixA, True)
    myBatchPlan.matrixMul([matrixB])
    print("BatchPlan typology")
    myBatchPlan.printBatchPlan()
    print("\n-------------------Begin to exec Batch Plan.-------------------")
    outputs = myBatchPlan.serialExec()
    row_num, col_num = myBatchPlan.matrix_shape
    output_matrix = np.zeros(myBatchPlan.matrix_shape)
    for row_id in range(row_num):
        output_matrix[row_id, :] = outputs[row_id]
    print("\n-------------------Batch Plan output:-------------------")
    print(output_matrix)
    print("\n-------------------Numpy output:-------------------")
    result = matrixA.dot(matrixB)
    print(result)

    if np.allclose(output_matrix, result):
        print("\n-------------------Test Pass!-------------------")
    else:
        print("\n-------------------Test Fail-------------------")
        print(output_matrix == result)

def weaver():
    data_store = DataStorage()
    myBatchPlan = BatchPlan(data_store, vector_mem_size=1024, element_mem_size=64)
    matrixA = np.random.randint(63, size=(1,8))
    matrixB = np.random.randint(63, size=(1,8))
    matrixC = np.random.randint(63, size=(8,2))


    print("\n-------------------Test Report:-------------------")
    myBatchPlan.fromMatrix(matrixA, True)
    print("In matrixA, max_slot_size of each vector is:", end=" ")
    for root in myBatchPlan.root_nodes:
        print(root.max_slot_size, end=" ")
    myBatchPlan.matrixAdd([matrixB], [False])
    print("\nAfter adding matrixB, max_slot_size of each vector is:", end=" ")
    for root in myBatchPlan.root_nodes:
        print(root.max_slot_size, end=" ")
    myBatchPlan.matrixMul([matrixC])
    print("\nAfter timing matrixC, max_slot_size of each vector is:", end=" ")
    for root in myBatchPlan.root_nodes:
        print(root.max_slot_size, end=" ")
    print("\n")
    idx = 0
    for root in myBatchPlan.root_nodes:
        vector_size = int(myBatchPlan.vector_mem_size / root.max_slot_size)
        print("Maximum compression rate in vector " + str(idx) + " is: " + str(vector_size))
        idx += 1
    print("\n-------------------Batch Plan before weave:-------------------")
    myBatchPlan.printBatchPlan()
    print("\n-------------------Batch Plan after weave:-------------------")
    myBatchPlan.weave()
    myBatchPlan.printBatchPlan()

    print("\n-------------------Begin to exec Batch Plan.-------------------")
    outputs = myBatchPlan.serialExec()
    row_num, col_num = myBatchPlan.matrix_shape
    output_matrix = np.zeros(myBatchPlan.matrix_shape)
    for row_id in range(row_num):
        output_matrix[row_id, :] = outputs[row_id]
    print("\n-------------------Batch Plan output:-------------------")
    print(output_matrix)
    print("\n-------------------Numpy output:-------------------")
    result = matrixA + matrixB
    result = result.dot(matrixC)
    print("\n-------------------Numpy output:-------------------")
    print(result)
    if np.allclose(output_matrix, result):
        print("\n-------------------Test Pass!-------------------")
    else:
        print("\n-------------------Test Fail-------------------")
        print(output_matrix == result)

weaver()


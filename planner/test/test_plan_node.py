import sys, math
sys.path.append("..")
from plan_node import PlanNode
from batch_plan import BatchPlan

import numpy as np

def init_batch_plan_from_matrix():
    myBatchPlan = BatchPlan(1024)
    matrixA = np.random.rand(4,3)
    print(matrixA)
    myBatchPlan.fromMatrix(matrixA)
    print("matrix shape: " + str(myBatchPlan.matrix_shape))
    print("each root node:")
    for root in myBatchPlan.root_nodes:
        print("batch data: " + str(root.getBatchData()))
        print("batch shape: " + str(root.getShape()))

def matrix_add():
    MyBatchPlan_1 = BatchPlan(1024)
    MyBatchPlan_2 = BatchPlan(1024)
    matrix_list = []
    for i in range(5):
        matrix_list.append(np.random.rand(4,3))
    print("\n-------------------Input Matrix: -------------------")
    for matrix in matrix_list:
        print(matrix)
    MyBatchPlan_1.fromMatrix(matrix_list[0], False)
    MyBatchPlan_2.fromMatrix(matrix_list[2], False)
    MyBatchPlan_1.matrixAdd([matrix_list[1]], [False])
    MyBatchPlan_2.matrixAdd([matrix_list[3], matrix_list[4]], [False, False])
    MyBatchPlan_1.matrixAdd([MyBatchPlan_2], [None])
    print("\n-------------------Begin to exec Batch Plan.-------------------")
    outputs = MyBatchPlan_1.execBatchPlan()
    row_num, col_num = MyBatchPlan_1.matrix_shape
    output_matrix = np.zeros(MyBatchPlan_1.matrix_shape)
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
    myBatchPlan = BatchPlan(1024)
    matrixA = np.random.rand(4,3)
    print("\n-------------------Matrix A: -------------------")
    print(matrixA)
    matrixB = np.random.rand(3,4)
    print("\n-------------------Matrix B: -------------------")
    print(matrixB)
    matrixC = np.random.rand(4,5)
    print("\n-------------------Matrix B: -------------------")
    print(matrixC)
    myBatchPlan.fromMatrix(matrixA, True)
    myBatchPlan.matrixMul([matrixB, matrixC])
    print(myBatchPlan.encrypted_flag)
    print("\n-------------------Begin to exec Batch Plan.-------------------")
    outputs = myBatchPlan.execBatchPlan()
    row_num, col_num = myBatchPlan.matrix_shape
    output_matrix = np.zeros(myBatchPlan.matrix_shape)
    for row_id in range(row_num):
        output_matrix[row_id, :] = outputs[row_id]
    print("\n-------------------Batch Plan output:-------------------")
    print(output_matrix)
    print("\n-------------------Numpy output:-------------------")
    result = matrixA.dot(matrixB)
    result = result.dot(matrixC)
    print(result)

    if np.allclose(output_matrix, result):
        print("\n-------------------Test Pass!-------------------")
    else:
        print("\n-------------------Test Fail-------------------")
        print(output_matrix == result)

def weaver():
    myBatchPlan = BatchPlan(1024)
    matrixA = np.ones([1, 100], dtype=np.uint32) * 4294967295
    matrixB = np.random.randint(15, size=(1,100))
    matrixC = np.random.randint(15, size=(100,2))


    print("\n-------------------Test Report:-------------------")
    myBatchPlan.fromMatrix(matrixA, True)
    print("In matrixA, max_element_size of each vector is:", end=" ")
    for root in myBatchPlan.root_nodes:
        print(root.max_element_size, end=" ")
    myBatchPlan.matrixAdd([matrixB], [False])
    print("\nAfter adding matrixB, max_element_size of each vector is:", end=" ")
    for root in myBatchPlan.root_nodes:
        print(root.max_element_size, end=" ")
    myBatchPlan.matrixMul([matrixC])
    print("\nAfter timing matrixC, max_element_size of each vector is:", end=" ")
    for root in myBatchPlan.root_nodes:
        print(root.max_element_size, end=" ")
    print("\n")
    idx = 0
    for root in myBatchPlan.root_nodes:
        vector_size = int(myBatchPlan.vector_mem_size / root.max_element_size)
        print("Maximum compression rate in vector " + str(idx) + " is: " + str(vector_size))
        idx += 1
    print("\n-------------------Batch Plan before weave:-------------------")
    myBatchPlan.printBatchPlan()
    print("\n-------------------Batch Plan after weave:-------------------")
    myBatchPlan.weave()
    myBatchPlan.printBatchPlan()

    print("\n-------------------Begin to exec Batch Plan.-------------------")
    outputs = myBatchPlan.execBatchPlan()
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


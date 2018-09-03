import numpy as np

#-------------------------------------------------------------------------
'''
    Problem 2: getting familiar with numpy package.
    In this problem, please install the following python package:
        * numpy 
    Numpy is the library for scientific computing in Python. It provides a high-performance multidimensional array object, and tools for working with these arrays. 
    To install numpy using pip, you could type `pip install numpy` in the terminal.
    Then start implementing function matrix_vector_multiplication()
    You could test the correctness of your code by typing `nosetests -v test2.py` in the terminal.

'''

#--------------------------
def matrix_vector_multiplication(X, y):
    ''' 
        Given a matrix X and a vector y, compute the product X*y = z
        Input: 
                X: a numpy matrix of float values, such as np.mat([1., 2.],[3., 4.],[5. ,6. ]), which is a (3 by 2) matrix;
                y: a numpy vector of float values, such as np.mat([[1.], [2.]]), which is a (2 by 1) vector.
        Output: 
                z: the numpy vector of float values, such as np.mat([[5.], [11.], [17.]]), which is a (3 by 1) vector.
    '''
    matrows, matcols = X.shape
    vecrows, veccols = y.shape
    z = np.zeros((matrows,veccols))
    for i in range(matrows):
        for j in range(veccols):
            for u,v in zip(range(matcols),range(vecrows)):
                z[i,j] += X[i,u]*y[v,j]
    z = np.asmatrix(z)
    #numpy's matmul can also be used for matrix multiplication
    #z = np.matmul(X,y)
    return z


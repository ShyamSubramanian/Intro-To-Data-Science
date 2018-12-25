#-------------------------------------------------------------------------
# Note: please don't use any additional package except the following packages
import numpy as np
import numpy.matlib
import math
#-------------------------------------------------------------------------
'''
    Problem 1: PCA 
    In this problem, you will implement a version of the principal component analysis method to reduce the dimensionality of data.
    You could test the correctness of your code by typing `nosetests -v test1.py` in the terminal.

    Notations:
            ---------- input data ------------------------
            n: the number of data instances (for example, # of images), an integer scalar.
            p: the number of dimensions (for example, # of pixels in each image), an integer scalar.
            X: the feature matrix, a float numpy matrix of shape n by p. 
            ---------- computed data ----------------------
            mu: the average vector of matrix X, a numpy float matrix of shape 1 by p. 
                Each element mu[0,i] represents the average value in the i-th column of matrix X.
            Xc: the centered matrix X, a numpy float matrix of shape n by p. 
                Each column has a zero mean value.
            C:  the covariance matrix of matrix X, a numpy float matrix of shape p by p. 
            k:  the number of dimensions to reduce to (k should be smaller than p), an integer scalar
            E:  the eigen vectors of matrix X, a numpy float matrix of shape p by p. 
                Each column of E corresponds to an eigen vector of matrix X.
            v:  the eigen values of matrix X, a numpy float array of length p. 
                Each element corresponds to an eigen value of matrix X. E and v are paired.
            Xp: the projected feature matrix with reduced dimensions, a numpy float matrix of shape n by k. 
             P: the projection matrix, a numpy float matrix of shape p by k. 
            -----------------------------------------------
'''

#--------------------------
def centering_X(X):
    '''
        Centering matrix X, so that each column has zero mean.
        Input:
            X:  the feature matrix, a float numpy matrix of shape n by p. Here n is the number of data records, p is the number of dimensions.
        Output:
            Xc:  the centered matrix X, a numpy float matrix of shape n by p. 
            mu:  the average row vector of matrix X, a numpy float matrix of shape 1 by p. 
        Note: please don't use the np.cov() function. There seems to be a bug in their code which will result in an error in later test cases. 
              Please implement this function only using basic numpy functions, such as np.mean().
    '''

    mu = np.asmatrix(np.mean(X,axis=0))
    Xc = X - mu
    return Xc, mu


#--------------------------
def compute_C(Xc):
    '''
        Compute the covariance matrix C. 
        Input:
            Xc:  the centered feature matrix, a float numpy matrix of shape n by p. Here n is the number of data records, p is the number of dimensions.
        Output:
            C:  the covariance matrix, a numpy float matrix of shape p by p. 
        Note: please don't use the np.cov() function here. Implement the function using matrix multiplication.
    '''
    Xc_centered,_ = centering_X(Xc)
    C = np.matmul(Xc_centered.T,Xc_centered) / Xc.shape[1]
    #C = (C+C.T)/2
    return C


#--------------------------
def compute_eigen_pairs(C):
    '''
        Compute the eigen vectors and eigen values of C. 
        Input:
            C:  the covariance matrix, a numpy float matrix of shape p by p. 
        Output:
            E:  the eigen vectors of matrix C, a numpy float matrix of shape p by p. Each column of E corresponds to an eigen vector of matrix C.
            v:  the eigen values of matrix C, a numpy float array of length p. Each element corresponds to an eigen value of matrix C. E and v are paired.
        Hint: you could use np.linalg.eig() to compute the eigen vectors of a matrix. 
    '''

    v,E = np.linalg.eig(C)
    return E,v

#--------------------------
def compute_P(E,v,k):
    '''
        Compute the projection matrix P by combining the eigen vectors with the top k largest eigen values. 
        Input:
            E:  the eigen vectors of matrix X, a numpy float matrix of shape p by p. Each column of E corresponds to an eigen vector of matrix X.
            v:  the eigen values of matrix X, a numpy float array of length p. Each element corresponds to an eigen value of matrix X. E and v are paired.
            k:  the number of dimensions to reduce to (k should be smaller than p), an integer scalar
        Output:
            P: the projection matrix, a numpy float matrix of shape p by k. 
    '''

    #P = np.matlib.zeros((E.shape[1],k))
    sorted_indices = np.argsort(-v)
    P = E[:, sorted_indices[:k]]
    #for i in range(k):
    #    P[:,i] = E[:,sorted_indices[i]]
    return P


#--------------------------
def compute_Xp(Xc,P):
    '''
        Compute the projected feature matrix Xp by projecting data Xc using matrix P. 
        Input:
            Xc:  the feature matrix after centering, a float numpy matrix of shape n by p. Here n is the number of data records, p is the number of dimensions.
             P: the projection matrix, a numpy float matrix of shape p by k. 
        Output:
            Xp: the feature matrix after projection (dimension reduced), a numpy float matrix of shape p by k. 
    '''

    Xp = np.matmul(Xc,P)
    return Xp



#--------------------------
def PCA(X, k=1):
    '''
        Compute PCA of matrix X. 
        Input:
            X:  the feature matrix, a float numpy matrix of shape n by p. Here n is the number of data records, p is the number of dimensions.
            k:  the number of dimensions to output (k should be smaller than p)
        Output:
            Xp: the feature matrix with reduced dimensions, a numpy float matrix of shape n by k. 
             P: the projection matrix, a numpy float matrix of shape p by k. 
        Note: in this problem, you cannot use existing package for PCA, such as scikit-learn
    '''

    Xc,_ = centering_X(X)

    C = compute_C(Xc)

    E,v = compute_eigen_pairs(C)

    P = compute_P(E,v,k)

    Xp = compute_Xp(Xc,P)

    return Xp, P 



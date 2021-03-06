import math
import numpy as np
from problem3 import Tree
#-------------------------------------------------------------------------
'''
    Problem 4: Decision Tree (with continous attributes)
    In this problem, you will implement the decision tree method for classification problems.
    You could test the correctness of your code by typing `nosetests -v test4.py` in the terminal.
'''

#--------------------------
class Node:
    '''
        Decision Tree Node (with continous attributes)
        Inputs: 
            X: the data instances in the node, a numpy matrix of shape p by n.
               Each element can be int/float.
               Here n is the number data instances in the node, p is the number of attributes.
            Y: the class labels, a numpy array of length n.
               Each element can be int/float/string.
            i: the index of the attribute being tested in the node, an integer scalar 
            th: the threshold on the attribute, a float scalar.
            C1: the child node for values smaller than threshold
            C2: the child node for values larger than threshold
            isleaf: whether or not this node is a leaf node, a boolean scalar
            p: the label to be predicted on the node (i.e., most common label in the node).
    '''
    def __init__(self,X=None,Y=None, i=None,th=None,C1=None, C2=None, isleaf= False,p=None):
        self.X = X
        self.Y = Y
        self.i = i
        self.th = th 
        self.C1= C1
        self.C2= C2
        self.isleaf = isleaf
        self.p = p


#-----------------------------------------------
class DT(Tree):
    '''
        Decision Tree (with contineous attributes)
        Hint: DT is a subclass of Tree class in problem1. So you can reuse and overwrite the code in problem 1.
    '''

    #--------------------------
    @staticmethod
    def cutting_points(X,Y):
        '''
            Find all possible cutting points in the continous attribute of X. 
            (1) sort unique attribute values in X, like, x1, x2, ..., xn
            (2) consider splitting points of form (xi + x(i+1))/2 
            (3) only consider splitting between instances of different classes
            (4) if there is no candidate cutting point above, use -inf as a default cutting point.
            Input:
                X: a list of values, a numpy array of int/float values.
                Y: a list of values, a numpy array of int/float/string values.
            Output:
                cp: the list of  potential cutting points, a float numpy vector. 
        '''
        X,Y = zip(*sorted(zip(X, Y), key=lambda x: x[0]))
        cp = []
        lastcp_X = X[0]
        lastcp_Y = Y[0]
        for index in range(len(X)):
            ly = lastcp_Y
            lx = lastcp_X
            if lastcp_X != X[index] and lastcp_Y != Y[index]:
                cp.append((lastcp_X+X[index])/2)
                lastcp_Y = Y[index]
                lastcp_X = X[index]
            if ly == Y[index]:
                lastcp_X = X[index]
            if lx == X[index] and ly != Y[index]:
                lastcp_Y = None
        cp = np.asarray(cp)
        return cp
    
    #--------------------------
    @staticmethod
    def conditional_entropy(Y,X,cp):
        '''
            Compute the conditional entropy of y given x.
            Input:
                Y: a list of values, a numpy array of int/float/string values.
                X: a list of values, a numpy array of int/float/string values.
            Output:
                ce: the conditional entropy of y given x, a float scalar
        '''
        x_places1 = X >= cp
        x_places2 = X < cp
        y1 = Y[x_places1]
        y2 = Y[x_places2]
        
        
        ce = 0.
        ce += len(y1)/len(Y)*Tree.entropy(y1)
        ce += len(y2)/len(Y)*Tree.entropy(y2)
        return ce
    
    #--------------------------
    @staticmethod
    def information_gain(Y,X,cp):
        '''
            Compute the information gain of y after spliting over attribute x
            Input:
                X: a list of values, a numpy array of int/float/string values.
                Y: a list of values, a numpy array of int/float/string values.
            Output:
                g: the information gain of y after spliting over x, a float scalar
        '''
        g = Tree.entropy(Y) - DT.conditional_entropy(Y,X,cp)
        return g
    
    #--------------------------
    @staticmethod
    def best_threshold(X,Y):
        '''
            Find the best threshold among all possible cutting points in the continous attribute of X. 
            Input:
                X: a list of values, a numpy array of int/float values.
                Y: a list of values, a numpy array of int/float/string values.
            Output:
                th: the best threhold, a float scalar. 
                g: the information gain by using the best threhold, a float scalar. 
            Hint: you can reuse your code in problem 1.
        '''
        cp = DT.cutting_points(X,Y)
        g = -1
        th = float('-inf')
        for index in range(cp.shape[0]):
            gain = DT.information_gain(Y,X,cp[index])
            print(gain,cp[index])
            if gain > g:
                g = gain
                th = cp[index]
        print(th,g)
        return th,g
    
    
    #--------------------------
    def best_attribute(self,X,Y):
        '''
            Find the best attribute to split the node. The attributes have continous values (int/float).
            Here we use information gain to evaluate the attributes. 
            If there is a tie in the best attributes, select the one with the smallest index.
            Input:
                X: the feature matrix, a numpy matrix of shape p by n. 
                   Each element can be int/float/string.
                   Here n is the number data instances in the node, p is the number of attributes.
                Y: the class labels, a numpy array of length n. Each element can be int/float/string.
            Output:
                i: the index of the attribute to split, an integer scalar
                th: the threshold of the attribute to split, a float scalar
        '''
        gain = -1
        th = -float('inf')
        index = -1
        for x in X:
            index += 1
            threshold,g = DT.best_threshold(x,Y)
            if g > gain:
                th = threshold
                gain = g
                i = index
        return i, th
    


        
    #--------------------------
    @staticmethod
    def split(X,Y,i,th):
        '''
            Split the node based upon the i-th attribute and its threshold.
            (1) split the matrix X based upon the values in i-th attribute and threshold
            (2) split the labels Y 
            (3) build children nodes by assigning a submatrix of X and Y to each node
    
            Input:
                X: the feature matrix, a numpy matrix of shape p by n.
                   Each element can be int/float.
                   Here n is the number data instances in the node, p is the number of attributes.
                Y: the class labels, a numpy array of length n.
                   Each element can be int/float/string.
                i: the index of the attribute to split, an integer scalar
            Output:
                C1: the child node for values smaller than threshold
                C2: the child node for values larger than (or equal to) threshold
        '''
        x = X[i]
        x_places1 = x < th
        x_places2 = ~x_places1
        C1 = Node(X[:,x_places1],Y[x_places1])
        C2 = Node(X[:,x_places2],Y[x_places2])
        return C1, C2
    
    
    
    #--------------------------
    def build_tree(self, t):
        '''
            Recursively build tree nodes.
            Input:
                t: a node of the decision tree, without the subtree built.
                t.X: the feature matrix, a numpy float matrix of shape n by p.
                   Each element can be int/float/string.
                    Here n is the number data instances, p is the number of attributes.
                t.Y: the class labels of the instances in the node, a numpy array of length n.
                t.C1: the child node for values smaller than threshold
                t.C2: the child node for values larger than (or equal to) threshold
        '''
        if Tree.stop1(t.Y) or Tree.stop2(t.X):
            t.isleaf = True
            t.p = Tree.most_common(t.Y)
        else:
            t.i,t.th = self.best_attribute(t.X,t.Y)
            t.p = Tree.most_common(t.Y)
            t.C1,t.C2 = DT.split(t.X,t.Y,t.i,t.th)
            self.build_tree(t.C1)
            self.build_tree(t.C2)
    
    #--------------------------
    @staticmethod
    def inference(t,x):
        '''
            Given a decision tree and one data instance, infer the label of the instance recursively. 
            Input:
                t: the root of the tree.
                x: the attribute vector, a numpy vectr of shape p.
                   Each attribute value can be int/float
            Output:
                y: the class labels, a numpy array of length n.
                   Each element can be int/float/string.
        '''
        y = t.p
        if t.isleaf != True:
            if x[t.i] < t.th:
                y = DT.inference(t.C1,x)
            else:
                y = DT.inference(t.C2,x)
        return y
    
    
    #--------------------------
    @staticmethod
    def predict(t,X):
        '''
            Given a decision tree and a dataset, predict the labels on the dataset. 
            Input:
                t: the root of the tree.
                X: the feature matrix, a numpy matrix of shape p by n.
                   Each element can be int/float.
                   Here n is the number data instances in the dataset, p is the number of attributes.
            Output:
                Y: the class labels, a numpy array of length n.
                   Each element can be int/float/string.
        '''
        Y = []
        for x in X.T:
            Y.append(DT.inference(t,x))
        Y = np.asarray(Y)
        return Y

        return Y
    
    
    
    #--------------------------
    def train(self, X, Y):
        '''
            Given a training set, train a decision tree. 
            Input:
                X: the feature matrix, a numpy matrix of shape p by n.
                   Each element can be int/float/string.
                   Here n is the number data instances in the training set, p is the number of attributes.
                Y: the class labels, a numpy array of length n.
                   Each element can be int/float/string.
            Output:
                t: the root of the tree.
        '''
        t = Node(X,Y)
        self.build_tree(t)
        return t


    #--------------------------
    @staticmethod
    def load_dataset(filename='data2.csv'):
        '''
            Load dataset 2 from the CSV file: data2.csv. 
            The first row of the file is the header (including the names of the attributes)
            In the remaining rows, each row represents one data instance.
            The first column of the file is the label to be predicted.
            In remaining columns, each column represents an attribute.
            Input:
                filename: the filename of the dataset, a string.
            Output:
                X: the feature matrix, a numpy matrix of shape p by n.
                   Each element is a float scalar.
                   Here n is the number data instances in the dataset, p is the number of attributes.
                Y: the class labels, a numpy array of length n.
                   Each element is a string.
        '''
        dataset = np.genfromtxt(filename,delimiter=',',skip_header=1,dtype=object)
        Y = dataset[:,0].astype('str')
        X = dataset[:,1:dataset.shape[1]].T.astype('float')
        return X,Y
    




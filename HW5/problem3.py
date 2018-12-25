#-------------------------------------------------------------------------
# Note: please don't use any additional package except the following packages
import numpy as np
from problem1 import *
from problem2 import UCBplayer 
#-------------------------------------------------------------------------
'''
    Problem 3: Monte Carlo Tree Search (MCTS) 
    In this problem, you will implement a MCTS player for TicTacToe.
    You could test the correctness of your code by typing `nosetests -v test2.py` in the terminal.
'''
            
#-----------------------------------------------
class Node():
    '''
        Search Tree Node
        Inputs: 
            s: the current state of the game, an integer matrix of shape 3 by 3. 
                s[i,j] = 0 denotes that the i-th row and j-th column is empty
                s[i,j] = 1 denotes that the i-th row and j-th column is taken by you. (for example, if you are the "O" player, then i, j-th slot is taken by "O") 
                s[i,j] = -1 denotes that the i-th row and j-th column is taken by the opponent.
            isleaf: whether or not this node is a leaf node (an end of the game), a boolean scalar
    '''
    def __init__(self,s, parent=None, isleaf= False):
        self.s = s
        self.isleaf = isleaf
        self.parent = parent 
        self.children= []
        self.N=0 # number of times being selected
        self.w=0 # sum of results

		


#-------------------------------------------------------
class PlayerMCTS:
    '''a player, that chooses optimal moves by Monte Carlo tree search. '''

    # ----------------------------------------------
    @staticmethod
    def expand(node):
        '''
         Expand the current tree node.
         Add one child node for each possible next move in the game.
         Inputs:
                node: the current tree node to be expanded 
         Outputs:
                c.children: a list of children nodes. 
        '''
        if TicTacToe.check(node.s) != None:
            node.isleaf = True
        else:
            rows,cols = np.where(node.s==0)
            for index in range(rows.shape[0]):
                r,c = rows[index],cols[index]
                sc = np.copy(node.s)
                sc[r,c] = 1
                child_node = Node(-sc,parent=node)
                node.children.append(child_node)
                if TicTacToe.check(child_node.s) != None:
                    child_node.isleaf = True

    # ----------------------------------------------
    @staticmethod
    def rollout(s):
        '''
         Monte Carlo simulation (rollout).
          Starting from the state s, simulates a randomized game from the current node until it reaches an end of the game.
          Inputs:
                s: the current state of the game, an integer matrix of shape 3 by 3. 
                    s[i,j] = 0 denotes that the i-th row and j-th column is empty
                    s[i,j] = 1 denotes that the i-th row and j-th column is taken by you. (for example, if you are the "O" player, then i, j-th slot is taken by "O") 
                    s[i,j] = -1 denotes that the i-th row and j-th column is taken by the opponent.
            Outputs:
                w: the result of the game (win:1, tie:0, lose: -1), an integer scalar. 
            Hint: you could use PlayerRandom in problem 1.
        '''
        p = PlayerRandom()
        sc = np.copy(s)
        player = 1

        while TicTacToe.check(sc) == None:
            r,c = p.play(sc)
            sc[r,c] = player
            player = -player

        w = TicTacToe.check(sc)
        
        return w



    # ----------------------------------------------
    @staticmethod
    def backprop(c,w):
        '''
         back propagation, update the game result in parent nodes recursively until reaching the root node. 
          Inputs:
                c: the current tree node to be updated
                w: the result of the game (win:1, tie:0, lose: -1), an integer scalar. 
        '''
        player = 1
        while c != None:
            c.N += 1
            c.w += player*w
            c = c.parent
            player = -player
        

    # ----------------------------------------------
    @staticmethod
    def selection(c):
        '''
         select the child node with the highest bound recursively until reaching a leaf node or an unexpanded node. 
          Inputs:
                c: the current tree node to be updated
          Outputs:
                node: the leaf/unexpanded node 
        
        '''
        
        if c.isleaf == True or len(c.children) == 0:
            node = c
        else:
            n = len(c.children)
            ucb = UCBplayer(n)
            for index in range(n):
                ucb.ni[index] = c.children[index].N
                ucb.w[index] = c.children[index].w
            ucb.N = c.N
            a = ucb.policy()
            node = PlayerMCTS.selection(c.children[a])            
        return node


    # ----------------------------------------------
    @staticmethod
    def build_tree(s,n=100):
        '''
        Given the current state of a game, build a search tree by n iteration of (selection->expand(selection)->rollout->backprop).
        After expanding a node, you need to run another selection operation starting from the expanded node, before performing rollout.
        Inputs: 
            s: the current state of the game, an integer matrix of shape 3 by 3. 
                s[i,j] = 0 denotes that the i-th row and j-th column is empty
                s[i,j] = 1 denotes that the i-th row and j-th column is taken by you. (for example, if you are the "O" player, then i, j-th slot is taken by "O") 
                s[i,j] = -1 denotes that the i-th row and j-th column is taken by the opponent.
            n: number of iterations, an interger scalar
          Outputs:
                root: the root node of the search tree 
        
        '''
        root = Node(s)
        for i in range(n):
            node = PlayerMCTS.selection(root)
            PlayerMCTS.expand(node)
            node = PlayerMCTS.selection(node)
            w = -PlayerMCTS.rollout(node.s)
            PlayerMCTS.backprop(node,w)
        
        return root


    # ----------------------------------------------
    def play(self,s,n=100):
        '''
           the policy function of the MCTS player, which chooses one move in the game.  
           Build a search tree with the current state as the root. Then find the most visited child node as the next action.
           Inputs:
                s: the current state of the game, an integer matrix of shape 3 by 3. 
                    s[i,j] = 0 denotes that the i-th row and j-th column is empty
                    s[i,j] = 1 denotes that the i-th row and j-th column is taken by you. (for example, if you are the "O" player, then i, j-th slot is taken by "O") 
                    s[i,j] = -1 denotes that the i-th row and j-th column is taken by the opponent.
            n: number of iterations when building the tree, an interger scalar
           Outputs:
                r: the row number, an integer scalar with value 0, 1, or 2. 
                c: the column number, an integer scalar with value 0, 1, or 2. 
          Hint: you could solve this problem using 3 lines of code.
        '''
        root = self.build_tree(s,n)
        max_n = 0
        max_index = 0
        index = -1
        for child in root.children:
            index += 1
            if child.N > max_n:
                max_index = index
                max_n = child.N
        diff = root.children[max_index].s + root.s
        rows,cols = np.where(diff != 0)
        r,c = rows[0],cols[0]
        return r,c







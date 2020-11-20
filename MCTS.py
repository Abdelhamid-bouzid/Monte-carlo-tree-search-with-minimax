# -*- coding: utf-8 -*-
"""
Created on Thu Nov 19 03:16:07 2020

@author: Abdelhamid
"""
import copy
import numpy as np

class MCTS():
    def __init__(self, state, br, parent=None, is_terminal=False,depth=0):
        
        self.state       = state
        self.parent      = parent
        self.children    = []
        self.is_expanded = False
        self.br_factor   = br
        self.is_terminal = False
        self.n_visit     = 0
        self.Q           = 0
        self.U           = -np.NINF
        self.depth       = depth
        
    '''##################################### add Node Function ###########################################'''
    def addNode(self,n_state,hist_sel):
        if len(hist_sel)==0:
            self.children.append(MCTS(n_state, self.br_factor, self.state,self.is_terminal,1))
        else:
            depth = len(hist_sel)
            for index in hist_sel:
                self = self.children[index]
            self.children.append(MCTS(n_state, self.br_factor, self.state,self.is_terminal,depth))
                
    '''##################################### Selction Function ###########################################'''            
    def selection(self):
        self.n_visit += 1
        hist_sel      = []
        while len(self.children):
            values        = np.array([node.Q +node.U for node in self.children])
            index         = np.argmax(values)
            self          = self.children[index]
            self.n_visit += 1  
            hist_sel.append(index)
            
        return hist_sel
    
    '''#################################### simulation Function ###########################################'''
    def simulation(self, hist_sel):
        current = copy.deepcopy(self)
        for index in hist_sel:
            current = current.children[index]
        
        i  = 4
        while not self.is_terminal and i>0:
            n_state     = 4    #just the new state
            is_terminal = True #suppose it is true 
            reward      = 1
            
            current.children.append(MCTS(n_state, self.br_factor, self.state, is_terminal))
            current = current.children[0]
            i -=1
            
        return reward
    
    '''################################## backprobagation Function ##########################################'''
    def backprobagation(self, hist_sel, reward, C, N):
        
        #Update root 
        self.Q  = ((self.n_visit-1)*self.Q + reward)/self.n_visit
        self.U   = C*np.sqrt(np.log(N)/self.n_visit)
        
        #update other selected nodes
        for index in hist_sel:
            self   = self.children[index]
            self.Q = ((self.n_visit-1)*self.Q + reward)/self.n_visit  
            self.U = C*np.sqrt(np.log(N)/self.n_visit)
    
    '''##################################### minimax Function ##############################################'''
    def minimax(self, depth, maximizingPlayer):
        
        if depth == 0 or len(self.children)==0:
            return self.Q + self.U
        
        if maximizingPlayer:
            max_val = np.NINF
            for child in self.children:
                val     = child.minimax(depth-1, False)
                max_val = max(val,max_val)
            return max_val
        else:
            min_val = -np.NINF
            for child in self.children:
                val     = child.minimax(depth-1, True)
                min_val = min(val,min_val)
            return min_val


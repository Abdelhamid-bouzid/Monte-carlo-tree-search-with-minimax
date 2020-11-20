# -*- coding: utf-8 -*-
"""
Created on Thu Nov 19 23:08:37 2020

@author: Abdelhamid
"""
from MCTS import MCTS

'''##################### create the root ################################'''
tree = MCTS(1,3)

'''################### add Nodes to the tree ############################'''
tree.addNode(2,[])
tree.addNode(3,[])
tree.addNode(4,[])

tree.addNode(-2,[0])
tree.addNode(-3,[1])
tree.addNode(-4,[2])

'''##################### Selection path  ################################'''
hist_sel = tree.selection()

'''########### Simulation  from after selection  ########################'''
reward   = tree.simulation(hist_sel)

'''########### backprobagation after Simulation  ########################'''
tree.backprobagation(hist_sel,reward, 0.01,5)

'''############# iterate many times in a loop  ##########################'''
for i in range(1,10):
    hist_sel = tree.selection()
    reward   = tree.simulation(hist_sel)
    tree.backprobagation(hist_sel,reward, 0.01,i)
    
value = tree.minimax(2,True)
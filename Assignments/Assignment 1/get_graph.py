#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 14 17:18:12 2021

@author: vijetadeshpande
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

def read_txt(filepath):
    
    #
    df_ = pd.read_csv(filepath, sep = " ", header = None)
    
    
    return df_

def get_size(node_connections):
    
    size_ = node_connections.max().max()
    
    return size_

def get_adjacency(node_connections):
    
    # size of the network
    size_ = get_size(node_connections)
    
    # create a empty square matrix of the size = size_
    adj_ = np.multiply(0, np.ones((size_, size_))) 
    
    # iterate over the list of given node connections
    for row_ in node_connections.index:
        
        # change the i-j to 1
        i = node_connections.iloc[row_, 0]
        j = node_connections.iloc[row_, 1]
        adj_[i-1][j-1] = 1
        adj_[j-1][i-1] = 1
    
    
    return adj_

def plot_graph(adj_, lab_ = None):
    
    # where are the connections
    rows, cols = np.where(adj_ == 1)
    edges = zip(rows.tolist(), cols.tolist())
    
    # initiate a nx.Graph object
    gr = nx.Graph()
    gr.add_edges_from(edges)
    
    # plot graph
    #nx.draw(gr, node_size=500, labels=mylabels, with_labels=True)
    nx.draw(gr, node_size = 50, with_labels = True)
    plt.show()
    
    return


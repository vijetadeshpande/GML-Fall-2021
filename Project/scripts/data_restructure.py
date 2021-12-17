#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 20 11:03:40 2021

@author: vijetadeshpande
"""
"""
This file processes the given input data into the required dat structuctures,

1. categories.txt: {label idx: label name}.json
2. network.txt: adjacency matrix in nested list format
3. titles: {node idx: {title: '', BERT: [], Glove: []}}
3. train, val, test.txt: [node]

"""
import pandas as pd
import os
import numpy as np
import torch
import torch.nn as nn
import json
#from TextEncoder import EmbModel as Enc
from tqdm import tqdm
import pickle
import train_utils as t_utils
from sklearn.decomposition import PCA
from RandomWalk import RandWalk
from sklearn.model_selection import train_test_split
import sys
import networkx as nx

# locations to import code from
path_import = []
path_file = os.path.dirname(os.path.abspath(__file__))
path_project = os.path.abspath(os.path.join(path_file, os.pardir))
path_data = os.path.join(path_project, 'Data')
path_import.append(os.path.join(path_project, 'decagon', 'polypharmacy'))
for i in path_import:
    sys.path.insert(1, i)

# import other scripts
from utility import *

def to_adjacency(path_data):
    
    """
    This function converts the given data of node connections into an adjacency
    matrix. 
    
    The final matrix (neested list) is returned as a adj_mat (not saved)
    
    """
    
    # read
    net, node2idx = load_ppi(path_data)
    
    #
    ppi_graph = nx.convert.to_dict_of_dicts(net)
    adj_mat = np.zeros((len(ppi_graph), len(ppi_graph)))
    for protein in ppi_graph:
        source_idx = node2idx[protein]
        target_idx = [node2idx[p] for p in ppi_graph[protein].keys()]
        
        # unweighted, undirected graph
        adj_mat[source_idx, target_idx] = 1
        adj_mat[target_idx, source_idx] = 1
    
    
    return adj_mat, net, node2idx

def node_to_embeddings(adj_mat, emb_dim = 768):
    
    # randomized embeddings to start with
    node_emb = np.random.standard_normal((adj_mat.shape[0], emb_dim))
    
    
    return node_emb


def label_to_name(path_):
    
    """
    This function converts the given information about the label names to a
    dictionary of following format,
    label_idx_to_name = {'label_idx': label_name}
    
    The dictionary will be saved as label_to_name.json file
    
    """
    
    # read
    data_ = pd.read_csv(os.path.join(path_, 'categories.txt'), sep = ' ', header = None)
    
    # fill the dictionary
    dict_ = {}
    for row_ in data_.index:
        dict_[int(data_.loc[row_, 0])] = data_.loc[row_, 1]
    
    # save
    filename_ = os.path.join(path_, 'label_to_name.json')
    with open(filename_, 'w') as f:
        json.dump(dict_, f)
    
    return


def create_train_test_val_unsup(path_, adj_mat, net, node2idx, graph_exp_depth = 2, negative_sample_size = 5):
    
    """
    
    """
    # create positive and negative sample set
    if not ((os.path.exists(os.path.join(path_, 'sample_set_negative.csv'))) or (os.path.exists(os.path.join(path_, 'sample_set_positive.csv')))):
        # create positive sample set
        sample_set, neighbor_map = RandWalk(walk_length = graph_exp_depth+1).create_sample_set(adj_mat, net, node2idx)
        # save
        #for set_ in sample_set:
        #    pd.DataFrame(sample_set[set_]).to_csv(os.path.join(path_, 'sample_set_'+set_+'.csv'))
    
    # create train, val and test dataset for unsupervised learning
    nodes_idx = [node2idx[node] for node in list(net.nodes)]
    data_ = pd.DataFrame(0, index = nodes_idx, columns = ['node', 'target'])
    data_.loc[:, 'node'] = nodes_idx
    
    #
    X_train, X_val, y_train, y_val = train_test_split(data_.loc[:, ['node']], data_.loc[:, ['target']], test_size = 0.2)
    X_train['target'] = y_train.values
    X_val['target'] = y_val.values
    
    # save
    X_train.to_csv(os.path.join(path_, 'train_GSage.csv'))
    X_val.to_csv(os.path.join(path_, 'val_GSage.csv'))
    
    
    return sample_set, (X_train, X_val), neighbor_map


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
from TextEncoder import EmbModel as Enc
from tqdm import tqdm
import pickle
import train_utils as t_utils
from sklearn.decomposition import PCA

def to_adjacency(path_):
    
    """
    This function converts the given data of node connections into an adjacency
    matrix. As the given graph is a directed graph, for node pair (i, j), we only
    update the [i][j] value of Adj mat.
    
    The final matrix (neested list) is saved as a adj_mat.json file
    
    """
    
    # read
    data_ = pd.read_csv(os.path.join(path_, 'network.txt'), sep = ' ', header = None)
    nodes = list(set(data_.loc[:, 0].unique().tolist() + data_.loc[:, 1].unique().tolist()))
    
    # create empty square matrix of size total_nodes
    adj_ = np.zeros((len(nodes), len(nodes))) #{} #np.zeros((len(nodes), len(nodes)))
    
    # iterate through every row of the data and update values in adj_
    for row_ in data_.index:
        adj_[data_.loc[row_, 0]][data_.loc[row_, 1]] = 1
        #if str(data_.loc[row_, 0]) in adj_:
        #    adj_[str(data_.loc[row_, 0])].append(data_.loc[row_, 1])
        #else:
        #    adj_[str(data_.loc[row_, 0])] = [data_.loc[row_, 1]]
    
    # save the matrix as .json 
    adj_ = adj_.tolist()
    filename_ = os.path.join(path_, 'data', 'adj_mat.p')
    with open(filename_, 'wb') as fp:
        pickle.dump(adj_, fp, protocol=pickle.HIGHEST_PROTOCOL)
    
    
    return


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

def node_to_properties(path_):
    
    """
    This function converts the title of a Wiki article that corresponds to a node
    in the given graph, to vector representation. For converting the input
    text sequence to a vector representation we use BERT with average pooling.
    These vectors will be saved in a dictionary, 
    {'node idx': {'title': '', 'BERT', []}}
    
    The dictionary will be saved as node_to_properties.json
    
    """
    
    # read
    data_ = pd.read_csv(os.path.join(path_, 'titles.txt'), sep = 'delimiter', header = None)
    data_[['node','text']] = data_.loc[:, 0].str.split(" ", 1, expand = True)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    #
    models = ['bert-base-uncased']
    n_to_feat = {}
    for model in models:
        m_ = Enc(model, device)
        m_ = m_.to(device)
        
        #
        n_to_feat[model] = pd.DataFrame(0, index = data_.loc[:, 'node'], columns = np.arange(768 + 1))
        
        with tqdm(total = len(data_), position=0, leave=True) as pbar:
            for row_ in tqdm(data_.index, position=0, leave=True):
                
                # save node value
                n_to_feat[model].iloc[row_, 0] = int(data_.loc[row_, 'node'])
                
                # encode text
                vec_ = m_(data_.loc[row_, 'text'])
                n_to_feat[model].iloc[row_, 1:] = vec_
                
                #
                pbar.update()

        # 
        df_ = pd.DataFrame(0, index = data_.loc[:, 'node'], columns = np.arange(300 + 1))
        df_.iloc[:, 0] = n_to_feat[model].iloc[:, 0].values
        df_.iloc[:, 1:] = PCA(n_components = 300).fit_transform(n_to_feat[model].iloc[:, 1:])#.values.tolist()

        #
        filename_ = os.path.join(path_, 'data', '%s_node_to_features_300.csv'%(model))
        df_.to_csv(filename_)
    

    return

def create_train_test_val(path_):
    
    #
    data = {}
    data['train'] = pd.read_csv(os.path.join(path_, 'train.txt'), sep = ' ', header = None)
    data['val'] = pd.read_csv(os.path.join(path_, 'val.txt'), sep = ' ', header = None)
    data['test'] = pd.read_csv(os.path.join(path_, 'test.txt'), sep = ' ', header = None)
    feat_all = pd.read_csv(os.path.join(path_, 'bert-base-uncased_node_to_features_300.csv'))
    
    feat = {}
    for set_ in data:
        feat[set_] =  pd.DataFrame(0, index = np.arange(len(data[set_])), columns = np.arange(300+1))
    
    #
    for set_ in data:
        #
        nodes = data[set_].loc[:, 0].values.tolist()
        feats = feat_all.iloc[nodes, 2:].values
        feat[set_].iloc[:, :-1] = feats
        
        #
        if not set_ == 'test':
            labels = data[set_].loc[:, 1].values.tolist()
            feat[set_].iloc[:, -1] = labels


    #
    for set_ in feat:
        feat[set_].to_csv(os.path.join(path_, '%s_.csv'%(set_)))
    
    return 


# test
    
##
#dir_ = r'/Users/vijetadeshpande/Documents/GitHub/GraphML-Fall-2021/Assignments/Assignment 4/data'
#to_adjacency(dir_)
#node_to_properties(dir_)
#data = pd.read_csv(os.path.join('data', 'bert-base-uncased_node_to_features.csv'))
#data_ = t_utils.transform_data(data, 500)
#data_.to_csv(os.path.join('data', 'bert-base-uncased_node_to_features_500.csv'))

#with open(os.path.join('data', 'node_to_features.json'), 'rb') as fp:
#    data = pickle.loads(fp)

#with open(os.path.join('data', 'adj_mat_.p'), 'rb') as fp:
#    data = pickle.load(fp)

#create_train_test_val(dir_)

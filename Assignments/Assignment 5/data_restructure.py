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

def to_adjacency(path_data):
    
    """
    This function converts the given data of node connections into an adjacency
    matrix. As the given graph is a directed graph, for node pair (i, j), we only
    update the [i][j] value of Adj mat.
    
    The final matrix (neested list) is saved as a adj_mat.json file
    
    """
    
    # read
    data_ = pd.read_csv(os.path.join(path_data, 'train.txt'), sep = ',', header = None)
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
    filename_ = os.path.join(path_data, 'adj_mat.p')
    with open(filename_, 'wb') as fp:
        pickle.dump(adj_, fp, protocol=pickle.HIGHEST_PROTOCOL)
    
    
    return

def node_to_embeddings(path_data):
    
    # read
    data_ = pd.read_csv(os.path.join(path_data, 'node-feat.txt'), sep = '\t', header = None)
    data_out = pd.DataFrame(0, index = data_.index, columns = np.arange(129))
    
    data_out.iloc[:, 0] = data_.iloc[:, 0].values.astype(int).tolist()
    for row in data_out.index:
        data_out.iloc[row, 1:] = data_.iloc[row, 1].split(',')
    
    #
    filename_ = os.path.join(path_data, 'node_embeddings.csv')
    data_out.to_csv(filename_)
    
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

def node_to_BERT(path_):
    
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

def create_train_test_val(path_, negative_sample_size = 5):
    
    #
    data_train_pos = pd.read_csv(os.path.join(path_, 'train.txt'), sep = ',', header = None)
    data_train_pos.columns = ['ni', 'nj']
    data_train_pos['link'] = 1
    
    # read adjacency matrix
    adj_mat = np.array(t_utils.load_pickle(os.path.join(path_, 'adj_mat.p'))).astype(int)
    
    # sample negative examples
    if os.path.exists(os.path.join(path_, 'sample_set_negative.csv')):
        sample_set_neg = pd.read_csv(os.path.join(path_, 'sample_set_negative.csv'))
    else:
        sample_set_neg = RandWalk(walk_length = 3).get_negative_samples(np.arange(len(adj_mat)), adj_mat)
        sample_set_neg = pd.DataFrame(sample_set_neg)
        sample_set_neg.to_csv(os.path.join(path_, 'sample_set_negative.csv'))
        
    # lage haat sample_set_positive bhi save kar lete he
    if os.path.exists(os.path.join(path_, 'sample_set_positive.csv')):
        sample_set_pos = pd.read_csv(os.path.join(path_, 'sample_set_positive.csv'))
    else:
        sample_set_pos = RandWalk(walk_length = 3).get_positive_samples(np.arange(len(adj_mat)), adj_mat)
        sample_set_pos = pd.DataFrame(sample_set_pos)
        sample_set_pos.to_csv(os.path.join(path_, 'sample_set_positive.csv'))
    
    
    # for creating training data we need to include negative examples for every node (ni) present in the dataset
    data_train_neg = pd.DataFrame(0, index = np.arange(negative_sample_size * data_train_pos.shape[0]), columns = ['ni', 'nj', 'link'])
    idx = 0
    for row in data_train_pos.index:
        case_ = data_train_pos.loc[row, 'ni']
        controls_ = sample_set_neg.loc[case_, :].sample(negative_sample_size).values.astype(int).tolist()
        
        #
        data_train_neg.loc[idx:idx+negative_sample_size-1, 'ni'] = case_
        data_train_neg.loc[idx:idx+negative_sample_size-1, 'nj'] = controls_
        data_train_neg.loc[idx:idx+negative_sample_size-1, 'link'] = 0
        
        idx += negative_sample_size
    
    # join dataframes and shuffle
    data_train = pd.concat([data_train_pos, data_train_neg]).sample(frac=1).reset_index(drop=True)
    
    #
    X_train, X_val, y_train, y_val = train_test_split(data_train.loc[:, ['ni','nj']], data_train.loc[:, ['link']], test_size = 0.3)
    
    X_train['link'] = y_train.values
    X_val['link'] = y_val.values
    
    # save 
    X_train.to_csv(os.path.join(path_, 'train_.csv'))
    X_val.to_csv(os.path.join(path_, 'val_.csv'))
    
    #
    data_test = pd.read_csv(os.path.join(path_, 'test.txt'), sep = ',', header = None)
    data_test.columns = ['ni', 'nj']
    data_test.to_csv(os.path.join(path_, 'test_.csv'))
    
    # data for GSage unsupervised learning
    data_GS = pd.DataFrame(0, index = np.arange(len(adj_mat)), columns = ['node', 'placeholder'])
    data_GS['node'] = np.arange(len(adj_mat)).astype(int)
    data_GS = data_GS.sample(frac=1).reset_index(drop=True)
    
    #
    X_train, X_val, y_train, y_val = train_test_split(data_GS.loc[:, ['node']], data_GS.loc[:, ['placeholder']], test_size = 0.2)
    X_train['target'] = y_train.values
    X_val['target'] = y_val.values
    
    # save
    X_train.to_csv(os.path.join(path_, 'train_GSage.csv'))
    X_val.to_csv(os.path.join(path_, 'val_GSage.csv'))
    
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

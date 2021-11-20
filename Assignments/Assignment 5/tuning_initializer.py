#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  3 11:18:09 2021

@author: vijetadeshpande
"""

import sys
import training_initializer as t_init_
import train_utils as t_utils_
import itertools
from ModelData import ModelData
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import torch
import pickle
import os
import json


def main(*args):
    
    """
    # path variables
    path_adj_ = r'/Users/vijetadeshpande/Documents/GitHub/GraphML-Fall-2021/Assignments/Assignment 4/data/adj_mat.p'
    path_bert_ = r'/Users/vijetadeshpande/Documents/GitHub/GraphML-Fall-2021/Assignments/Assignment 4/data/bert-base-uncased_node_to_features_300.csv'
    path_data_ = r'/Users/vijetadeshpande/Documents/GitHub/GraphML-Fall-2021/Assignments/Assignment 4/data'
    
    # load data
    with open(path_adj_, 'rb') as fp:
        adj_mat = pickle.load(fp) 
    data_feat = pd.read_csv(path_bert_).iloc[:, 2:]
    data_object = ModelData(path_data_, batch_size = 16)
    
    # create a folder to save the trained models
    path_res_ = os.path.join(r'/Users/vijetadeshpande/Documents/GitHub/GraphML-Fall-2021/Assignments/Assignment 4', 'results', 'experiments', 'RB experiment', 'Depth_3', 'wRB')
    if not os.path.exists(path_res_):
        os.makedirs(path_res_)
    
    # Grid for hyper-par search
    hidden_dims = [[256, 128, 64]]
    depths = [3]
    l_rates = [0.001] * 100#[0.03, 0.01, 0.009, 0.007, 0.005, 0.003, 0.001]#np.power(10, np.random.normal(loc=-3.5, scale=0.7, size=10))
    n_epochs = [50]
    dropouts = [0.03]#[0.03, 0.05, 0.1]
    sample_sizes = [[5, 5, 5]]
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    """
    
    # paths
    path_ = args[0].path_
    path_data_ = args[0].path_data_
    path_res_ = args[0].path_res_
    
    
    # data
    adj_mat = args[0].adjacency_mat
    data_feat = args[0].node_feature_vectors
    #data_object = args[0].data_ttv
    batch_ = args[0].batch_size
    node_feature_dim = args[0].node_feature_dim
    data_object = ModelData(path_data_, 
                            batch_size = batch_,
                            data_train = args[0].data_train_,
                            data_val = args[0].data_val_,
                            data_test = args[0].data_test_,
                            bert_feature_dim = node_feature_dim)
    
    
    # hyperparameters
    repeat_ = args[0].experiments_repeat_number if args[0].experiments_repeat else 1
    depths = args[0].gnn_depth * repeat_
    hidden_dims = args[0].gnn_hidden_dims
    dropouts = args[0].gnn_hidden_dropout
    sample_sizes = args[0].gnn_neighbor_sample_size
    l_rates = args[0].lr_start
    l_rate_reductions = args[0].lr_reduction
    l_rate_reduction_steps = args[0].lr_reduction_step
    n_epochs = args[0].training_epochs
    device = args[0].device
    
    #
    pars = list(itertools.product(hidden_dims, depths, dropouts, l_rates, n_epochs, sample_sizes, l_rate_reductions, l_rate_reduction_steps))
    results = {}
    idx = 0
    for par in pars:
        idx += 1
        
        # print 
        print(('\nStarting training for hyperparameter set: %d')%(idx))
    
        # create dictionary of parameter values
        (hidden_dim, depth, dropout, l_rate, n_epoch, sample_size, l_rate_red, l_rate_red_step) = par
        par_dict = {'input layer size': node_feature_dim,
                    'hidden layer size': hidden_dim,
                    'depth': depth,
                    'dropout rate': dropout,
                    'learning rate': l_rate,
                    'learning rate reduction': l_rate_red,
                    'learning rate reduction step': l_rate_red_step,
                    'epochs': n_epoch,
                    'device': device,
                    'adjacency matrix': adj_mat,
                    'BERT representations': data_feat,
                    'number of classes': data_object.output_features,
                    'aggregator': 'mean',
                    'sample size': sample_size,
                    'data': data_object,
                    'model filename': os.path.join(path_res_, 'GSage_for_hpar_set_%d.pt'%(idx))
                    }
        
        # train with current set of paramters
        results[idx] = {}
        results[idx]['results'] = t_init_.main(par_dict)
        #_ = par_dict.pop('device')
        _ = par_dict.pop('data')
        _ = par_dict.pop('BERT representations')
        _ = par_dict.pop('adjacency matrix')
        results[idx]['hyperparameters'] = par_dict
    
    # save tuning results
    result_best = t_utils_.save_tuning_results(results, path_res_,
                                               save_all_ = False,
                                               save_top3_ = True if len(pars) >= 3 else False)
    
    # plot train and validation error for best result
    t_utils_.plot_train_val(result_best, path_res_)
    
    # plot embeddings
    t_utils_.embedding_visualization(embeddings = result_best['results']['node embeddings'], 
                                     train = args[0].data_train_,
                                     val = args[0].data_val_,
                                     save_path = path_res_)
    
    # print out the prediction.txt file
    t_utils_.write_predictions(result_best, os.path.join(path_, 'predictions.txt'))
    
    
    return

#
#xx, xx1 = t_utils_.load_results(path_res_)
    
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 10 23:22:13 2021

@author: vijetadeshpande
"""
import sys
import os
import train_utils as t_utils_
import pandas as pd
import argparse
import torch
import tuning_initializer as hpt__init__
import data_restructure as pre_pro

def main(*args, **kwargs):
    
    #
    #print(args)
    
    """
    This function primarily initializes the training and testing of the 
    GraphSage model.     
    
    """    
    
    # 
    # path variables
    path_ = os.getcwd()
    path_data_ = os.path.join(path_, 'data_ast4')
    path_adj_ = os.path.join(path_data_, 'adj_mat.p')
    path_feat_ = os.path.join(path_data_, 'node_embeddings.csv')
    
    # create adj_mat
    if not os.path.exists(path_adj_):
        print('\nCreating adjacency matrix: ')
        pre_pro.to_adjacency(path_data_)
        print('DONE')
    
    # create node embeddings
    if not os.path.exists(path_feat_):
        print('\nCreating embeddings for nodes: ')
        pre_pro.node_to_embeddings(path_data_)
        print('DONE')
    
    # create train, val and testing set
    print('\nCreating training, validation and testing datasets: ')
    pre_pro.create_train_test_val(path_data_)
    print('DONE')
    
    # load data
    adj_mat = t_utils_.load_pickle(path_adj_)
    data_feat = pd.read_csv(path_bert_).iloc[:, 2:]
    
    # create a folder to save the trained models
    path_res_ = os.path.join(path_, 'results_')
    if not os.path.exists(path_res_):
        os.makedirs(path_res_)
    
    # read the best model
    try: 
        best_model = t_utils_.load_pickle(os.path.join(path_data_, 'best_result.p'))
    except:
        best_model = {}
    
    # set all arguments
    parser = argparse.ArgumentParser(description = 'COMP5800_Assignment_4_GraphSAGE')
    parser.add_argument('--adjacency_mat', default = adj_mat)
    parser.add_argument('--node_feature_dim', default = 300)
    parser.add_argument('--node_feature_vectors', default = data_feat)
    parser.add_argument('--data_train_', default = args[0][-3])
    parser.add_argument('--data_val_', default = args[0][-2])
    parser.add_argument('--data_test_', default = args[0][-1])
    #parser.add_argument('--data_ttv', default = data_object)
    parser.add_argument('--best_model', default = best_model)
    parser.add_argument('--batch_size', default = 32)
    parser.add_argument('--path_', default = path_)
    parser.add_argument('--path_data_', default = path_data_)
    parser.add_argument('--path_res_', default = path_res_)
    
    
    # following hyperparameters are read as list so that they can be iterated 
    # over for tuning 
    parser.add_argument('--gnn_depth', default = [1])
    parser.add_argument('--gnn_hidden_dims', default = [[50]])#[[32], [50], [64], [100], [128]])
    parser.add_argument('--gnn_hidden_dropout', default = [0.03])#[0, 0.01, 0.03, 0.05, 0.1])
    parser.add_argument('--gnn_neighbor_sample_size', default = [[5]]) #[[5], [7], [10]])
    parser.add_argument('--lr_start', default = [0.01])#[0.0001, 0.001, 0.003, 0.005, 0.007, 0.009, 0.01, 0.03])
    parser.add_argument('--lr_reduction', default = [0.9])
    parser.add_argument('--lr_reduction_step', default = [10])
    parser.add_argument('--training_epochs', default = [200])
    parser.add_argument('--device', default = torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
    parser.add_argument('--experiments_repeat', default = False)
    parser.add_argument('--experiments_repeat_number', default = 5)
    args_, unknown_ = parser.parse_known_args()
    
    # initialize tuning/training and inference
    #print('here')
    #print(args_)
    hpt__init__.main(args_)
    
    return


if __name__ == '__main__':
    #main(sys.argv[1:])
    main(('network.txt', 'categories.txt', 'titles.txt', 'train.txt', 'val.txt', 'test.txt'))

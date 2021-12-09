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
import numpy as np
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
        print('\nReading node feature vectors: ')
        pre_pro.node_to_embeddings(path_data_)
        print('DONE')
    
    # create train, val and testing set
    if not (os.path.exists(os.path.join(path_data_, 'train_.csv')) or os.path.exists(os.path.join(path_data_, 'val_.csv')) or os.path.exists(os.path.join(path_data_, 'train_GSage.csv')) or os.path.exists(os.path.join(path_data_, 'val_GSage.csv'))):
        print('\nCreating training, validation and testing datasets: ')
        pre_pro.create_train_test_val(path_data_)
        print('DONE')
    
    # load data
    adj_mat = np.array(t_utils_.load_pickle(path_adj_)).astype(int).tolist()
    node_emb_lan = pd.read_csv(path_feat_).iloc[:, 2:]
    try:
        node_emb_graph = pd.read_csv(os.path.join(path_data_, 'node_embeddings_GSage.csv')).iloc[:, 2:]
    except:
        node_emb_graph = []
    
    # create a folder to save the trained models
    path_res_ = os.path.join(path_data_, 'results_')
    if not os.path.exists(path_res_):
        os.makedirs(path_res_)
    
    # read the best model
    try: 
        best_model = t_utils_.load_pickle(os.path.join(path_data_, 'best_result.p'))
    except:
        best_model = {}
    
    # set all arguments
    parser = argparse.ArgumentParser(description = 'COMP5800_Assignment_5_GraphSAGE')
    parser.add_argument('--adjacency_mat', default = adj_mat)
    parser.add_argument('--node_feature_dim', default = 128)
    parser.add_argument('--node_feature_vectors', default = node_emb_lan)
    #parser.add_argument('--data_train_', default = args[0][-3])
    #parser.add_argument('--data_val_', default = args[0][-2])
    #parser.add_argument('--data_test_', default = args[0][-1])
    #parser.add_argument('--data_ttv', default = data_object)
    parser.add_argument('--best_model', default = best_model)
    parser.add_argument('--path_', default = path_)
    parser.add_argument('--path_data_', default = path_data_)
    parser.add_argument('--path_res_', default = path_res_)
    
    
    # following hyperparameters are read as list so that they can be iterated 
    # over for tuning 
    
    # GraphSage hyper parameters
    parser.add_argument('--gnn_batch_size', default = 1024)
    parser.add_argument('--gnn_depth', default = 3)
    parser.add_argument('--gnn_input_layer_size', default = [128])
    parser.add_argument('--gnn_hidden_layer_size', default = [[128, 128, 128]])#[[32], [50], [64], [100], [128]])
    parser.add_argument('--gnn_hidden_layer_dropout', default = [0])#[0, 0.01, 0.03, 0.05, 0.1])
    parser.add_argument('--gnn_output_layer_size', default = [128])
    parser.add_argument('--gnn_neighborhood_sample_size', default = [[5, 7, 10]]) #[[5], [7], [10]])
    parser.add_argument('--gnn_aggregator', default = ['mean'])
    parser.add_argument('--gnn_unsupervised_random_walk_length', default = [3])
    parser.add_argument('--gnn_unsupervised_positive_sample_size', default = [3])
    parser.add_argument('--gnn_unsupervised_negative_sample_size', default = [10])
    parser.add_argument('--gnn_lr_start', default = [0.0001])#[0.0001, 0.001, 0.003, 0.005, 0.007, 0.009, 0.01, 0.03])
    parser.add_argument('--gnn_lr_reduction', default = [0.97])
    parser.add_argument('--gnn_lr_reduction_step', default = [30])
    parser.add_argument('--gnn_training_epochs', default = [1000])
    
    
    # Classifier for link classification task
    parser.add_argument('--classifier_batch_size', default = 32)
    parser.add_argument('--classifier_use_language_embeddings', default = [True])
    parser.add_argument('--classifier_use_graph_embeddings', default = [False])
    parser.add_argument('--classifier_layers_lan', default = [2])
    parser.add_argument('--classifier_input_layer_size_lan', default = [128])
    parser.add_argument('--classifier_hidden_layer_size_lan', default = [[64, 32]])#[[32], [50], [64], [100], [128]])
    parser.add_argument('--classifier_hidden_layer_dropout_lan', default = [0.01])#[0, 0.01, 0.03, 0.05, 0.1])
    
    parser.add_argument('--classifier_layers_graph', default = [2])
    parser.add_argument('--classifier_input_layer_size_graph', default = [128])
    parser.add_argument('--classifier_hidden_layer_size_graph', default = [[64, 32]])#[[32], [50], [64], [100], [128]])
    parser.add_argument('--classifier_hidden_layer_dropout_graph', default = [0.01])#[0, 0.01, 0.03, 0.05, 0.1])
    
    parser.add_argument('--classifier_output_layer_size', default = 1)
    parser.add_argument('--classifier_lr_start', default = [0.0001])#[0.0001, 0.001, 0.003, 0.005, 0.007, 0.009, 0.01, 0.03])
    parser.add_argument('--classifier_lr_reduction', default = [0.5])
    parser.add_argument('--classifier_lr_reduction_step', default = [4])
    parser.add_argument('--classifier_training_epochs', default = [30])
    parser.add_argument('--classifier_node_embeddings_graph', default = node_emb_graph)
    
    #nAdditinal parameters
    parser.add_argument('--device', default = torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
    parser.add_argument('--experiments_repeat', default = False)
    parser.add_argument('--experiments_repeat_number', default = 5)
    
    # initialize unsupervised learning of GSage
    if not os.path.exists(os.path.join(path_data_, 'node_embeddings_GSage.csv')):
        parser.add_argument('--ML_TASK', default = 'unsupervised_learning')
    else:
        parser.add_argument('--ML_TASK', default = 'supervised_learning')
    args_, unknown_ = parser.parse_known_args()
    
    #
    best_ = hpt__init__.main(args_)
    
    # initialize supervised learning of the classifier
    if args_.ML_TASK == 'unsupervised_learning':
        args_.ML_TASK = 'supervised_learning'
        best_ = hpt__init__.main(args_)
    
    #
    
    return best_


if __name__ == '__main__':
    #main(sys.argv[1:])
    #main(('network.txt', 'categories.txt', 'titles.txt', 'train.txt', 'val.txt', 'test.txt'))
    best_res = main(('data/'))
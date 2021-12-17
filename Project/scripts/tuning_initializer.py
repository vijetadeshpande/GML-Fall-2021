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
    
    # prepare data for training of the model
    data_object = get_torch_data_object(args[0])
    
    #
    pars = get_hyperpar_set_list(args[0])
    results = {}
    idx = 0
    for par in pars:
        idx += 1
        
        # print 
        print(('\nStarting training for hyperparameter set: %d')%(idx))
        
        # create dictionary of parameter values
        if args[0].ML_TASK == "unsupervised_learning":
        #(hidden_dim, depth, dropout, l_rate, n_epoch, sample_size, l_rate_red, l_rate_red_step) = par
            par_dict = {'gnn_input_layer_size': par['gnn_input_layer_size'],
                        'gnn_hidden_layer_size': par['gnn_hidden_layer_size'],
                        'gnn_depth': args[0].gnn_depth,
                        'gnn_hidden_layer_dropout': par['gnn_hidden_layer_dropout'],
                        'gnn_lr_start': par['gnn_lr_start'],
                        'gnn_lr_reduction': par['gnn_lr_reduction'],
                        'gnn_lr_reduction_step': par['gnn_lr_reduction_step'],
                        'training_epochs': par['gnn_training_epochs'],
                        'device': args[0].device,
                        'adjacency_matrix': args[0].adjacency_mat,
                        'node_feature_vectors': args[0].node_feature_vectors,
                        'gnn_output_layer_size': par['gnn_output_layer_size'],
                        'gnn_aggregator': par['gnn_aggregator'],
                        'gnn_neighborhood_sample_size': par['gnn_neighborhood_sample_size'],
                        'gnn_unsupervised_negative_sample_size': par['gnn_unsupervised_negative_sample_size'],
                        'gnn_unsupervised_positive_sample_size': par['gnn_unsupervised_positive_sample_size'],
                        'gnn_unsupervised_random_walk_length': par['gnn_unsupervised_random_walk_length'],
                        
                        'gnn_unsupervised_positive_sample_set': args[0].sample_set['positive'],
                        'gnn_unsupervised_negative_sample_set': args[0].sample_set['negative'],
                        'gnn_unsupervised_neighbor_map': args[0].neighbor_map,
                        'data': data_object,
                        'model_filename': os.path.join(args[0].path_res_, 'GSage_for_hpar_set_%d.pt'%(idx)),
                        'path_': args[0].path_,
                        'path_data_': args[0].path_data_,
                        'path_res_': args[0].path_res_,
                        'ML_TASK': args[0].ML_TASK}
        
        elif args[0].ML_TASK == "supervised_learning":
            
            par_dict = {
                        'device': args[0].device,
                        'adjacency_matrix': args[0].adjacency_mat,
                        'node_feature_vectors': args[0].node_feature_vectors,
                        'node_embeddings_graph': args[0].classifier_node_embeddings_graph,
                        
                        'data': data_object,
                        'model_filename': os.path.join(args[0].path_res_, 'link_pred_for_hpar_set_%d.pt'%(idx)),
                        'path_': args[0].path_,
                        'path_data_': args[0].path_data_,
                        'path_res_': args[0].path_res_,
                        'ML_TASK': args[0].ML_TASK,
                        
                        'classifier_batch_size': args[0].classifier_batch_size,
                        'classifier_use_language_embeddings': par['classifier_use_language_embeddings'],
                        'classifier_use_graph_embeddings': par['classifier_use_graph_embeddings'],
                        
                        'classifier_layers_lan': par['classifier_layers_lan'],
                        'classifier_input_layer_size_lan': par['classifier_input_layer_size_lan'],
                        'classifier_hidden_layer_size_lan': par['classifier_hidden_layer_size_lan'],
                        'classifier_hidden_layer_dropout_lan': par['classifier_hidden_layer_dropout_lan'],
                        'classifier_layers_graph': par['classifier_layers_graph'],
                        'classifier_input_layer_size_graph': par['classifier_input_layer_size_graph'],
                        'classifier_hidden_layer_size_graph': par['classifier_hidden_layer_size_graph'],
                        'classifier_hidden_layer_dropout_graph': par['classifier_hidden_layer_dropout_graph'],
                        
                        #
                        'classifier_output_layer_size': args[0].classifier_output_layer_size,
                        'classifier_lr_start': par['classifier_lr_start'],
                        'classifier_lr_reduction': par['classifier_lr_reduction'],
                        'classifier_lr_reduction_step': par['classifier_lr_reduction_step'],
                        'classifier_training_epochs': par['classifier_training_epochs']}
            
        else:
            par_dict = {}

        # train with current set of paramters
        results[idx] = {}
        results[idx]['results'] = t_init_.main(par_dict)
        #_ = par_dict.pop('device')
        _ = par_dict.pop('data')
        _ = par_dict.pop('node_feature_vectors')
        _ = par_dict.pop('adjacency_matrix')
        results[idx]['hyperparameters'] = par_dict
    
    # save tuning results
    result_best = t_utils_.save_tuning_results(results, args[0].path_res_,
                                               save_all_ = False,
                                               save_top3_ = True if len(pars) >= 3 else False)
    
    # plot train and validation error for best result
    t_utils_.plot_train_val(result_best, args[0].path_res_)
    
    # plot embeddings
    #t_utils_.embedding_visualization(embeddings = result_best['results']['node embeddings'], 
    #                                 train = args[0].data_train_,
    #                                 val = args[0].data_val_,
    #                                 save_path = args[0].path_res_)
    
    # print out the prediction.txt file
    if args[0].ML_TASK == 'supervised_learning':
        t_utils_.write_predictions(result_best, os.path.join(args[0].path_, 'predictions.txt'))
    
    
    return result_best

def product_dict(**kwargs):
    
    #
    keys = kwargs.keys()
    vals = kwargs.values()
    
    #
    out = []
    for instance in itertools.product(*vals):
        out.append(dict(zip(keys, instance)))
    
    return out


def get_hyperpar_set_list(*args):
    
    

    if args[0].ML_TASK == 'unsupervised_learning':
        hp_set_list = product_dict(gnn_input_layer_size = args[0].gnn_input_layer_size,
                                   gnn_hidden_layer_size = args[0].gnn_hidden_layer_size,
                                   gnn_hidden_layer_dropout = args[0].gnn_hidden_layer_dropout,
                                   gnn_output_layer_size = args[0].gnn_output_layer_size,
                                   gnn_neighborhood_sample_size = args[0].gnn_neighborhood_sample_size,
                                   gnn_aggregator = args[0].gnn_aggregator,
                                   gnn_unsupervised_random_walk_length = args[0].gnn_unsupervised_random_walk_length,
                                   gnn_unsupervised_positive_sample_size = args[0].gnn_unsupervised_positive_sample_size,
                                   gnn_unsupervised_negative_sample_size = args[0].gnn_unsupervised_negative_sample_size,
                                   gnn_lr_start = args[0].gnn_lr_start,
                                   gnn_lr_reduction = args[0].gnn_lr_reduction,
                                   gnn_lr_reduction_step = args[0].gnn_lr_reduction_step,
                                   gnn_training_epochs = args[0].gnn_training_epochs)
    
    elif args[0].ML_TASK == 'supervised_learning':
        hp_set_list = product_dict(classifier_use_language_embeddings = args[0].classifier_use_language_embeddings,
                                   classifier_use_graph_embeddings = args[0].classifier_use_graph_embeddings,
                               
                                   classifier_layers_lan = args[0].classifier_layers_lan,
                                   classifier_input_layer_size_lan = args[0].classifier_input_layer_size_lan,
                                   classifier_hidden_layer_size_lan = args[0].classifier_hidden_layer_size_lan,
                                   classifier_hidden_layer_dropout_lan = args[0].classifier_hidden_layer_dropout_lan,
                                   classifier_layers_graph = args[0].classifier_layers_graph,
                                   classifier_input_layer_size_graph = args[0].classifier_input_layer_size_graph,
                                   classifier_hidden_layer_size_graph = args[0].classifier_hidden_layer_size_graph,
                                   classifier_hidden_layer_dropout_graph = args[0].classifier_hidden_layer_dropout_graph,
                                   
                                   classifier_lr_start = args[0].classifier_lr_start,
                                   classifier_lr_reduction = args[0].classifier_lr_reduction,
                                   classifier_lr_reduction_step = args[0].classifier_lr_reduction_step,
                                   classifier_training_epochs = args[0].classifier_training_epochs)
    else:
        hp_set_list = []

    
    return hp_set_list

def get_torch_data_object(*args):
    
    data_dir = args[0].path_data_
    
    if args[0].ML_TASK == 'unsupervised_learning':
        
        torch_data = ModelData(data_dir,
                               task = args[0].ML_TASK,
                               batch_size = args[0].gnn_batch_size,
                               filename_train = os.path.join(data_dir, 'train_GSage.csv'),
                               filename_val = os.path.join(data_dir, 'val_GSage.csv'),
                               filename_test = os.path.join(data_dir, 'val_GSage.csv'),
                               node_feature_dim = args[0].node_feature_dim)
        
    elif args[0].ML_TASK == 'supervised_learning':
        
        torch_data = ModelData(data_dir,
                               task = args[0].ML_TASK,
                               batch_size = args[0].classifier_batch_size,
                               filename_train = os.path.join(data_dir, 'train_.csv'),
                               filename_val = os.path.join(data_dir, 'val_.csv'),
                               filename_test = os.path.join(data_dir, 'test_.csv'),
                               node_feature_dim = args[0].node_feature_dim)
    
    
    
    return torch_data
        
    
    
    
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



return 

"""
    

#
#xx, xx1 = t_utils_.load_results(path_res_)
    
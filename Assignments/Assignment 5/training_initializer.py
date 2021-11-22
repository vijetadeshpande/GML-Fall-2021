#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  3 09:50:37 2021

@author: vijetadeshpande
"""
import torch
import torch.nn as nn
from Models import GSageUnsup as GSUn
from Models import NNClassifier as NNCls
from GSUnsupLoss import GSUnsupLoss as UnsupLoss
import train_utils as utils_
import torch.optim as optim
import time
import os
import sys
from tqdm import tqdm
import numpy as np
from train_and_test import train, evaluate, inference



def main(parameter_dict):
    
    # some required parameters
    device = parameter_dict['device']
    data = parameter_dict['data']
    model_filename = parameter_dict['model_filename']
    clip = 1
    
    if parameter_dict['ML_TASK'] == 'unsupervised_learning':
        epochs = parameter_dict['training_epochs']
    else:
        epochs = parameter_dict['classifier_training_epochs']
    
    try:
        depth = parameter_dict['gnn_depth']
    except:
        depth = 0
    
    #
    __task__ = parameter_dict['ML_TASK']
    
    #%% UN_SUPERVISED NODE EMBEDDING DEVELOPMENT
    
    # training, validation and testing data
    data_train, data_val, data_test = data.train, data.val, data.test
    
    # define and initialize model
    model = get_model(parameter_dict, __task__)
    
    # define objective function
    obj_f = get_objective_function(parameter_dict, __task__)
    
    # define solution method
    model, sol_m, sch_lr = get_solution_method(parameter_dict, model, __task__)
    
    # start training
    losses_train, losses_val = [], []
    accs_train, accs_val = [], []
    time_total = 0
    model_state_dict = None
    best_valid_loss = 1000
    #with tqdm(total = epochs, position=0, leave=True) as pbar:
    for epoch in tqdm(range(epochs), position=0, leave=True):
        time_start = time.time()
        
        # train the model
        loss_train, acc_train = train(model, data_train, sol_m, obj_f, clip, device, __task__)
        loss_val, acc_val, _ = evaluate(model, data_val, obj_f, device, __task__)
        
        time_end = time.time()
        
        # print progress
        #utils_.print_train_progress(epoch, epochs, loss_train, acc_train, loss_val, acc_val, time_start, time_end)
        
        # save losses
        losses_train.append(loss_train)
        losses_val.append(loss_val)
        accs_train.append(acc_train)
        accs_val.append(acc_val)
        
        # update values
        time_total += (time_end - time_start)
        sch_lr.step()
        #pbar.update()
        
        # save model
        model_state_dict, best_valid_loss = utils_.update_model(loss_val, best_valid_loss, model, model_state_dict)

    
    # print loss and accuracy during last epoch
    utils_.print_train_progress(epoch, epochs, loss_train, acc_train, loss_val, acc_val, time_start, time_end)
    
    # test the best model
    try:
        #model.load_state_dict(torch.load(model_filename))
        model.load_state_dict(model_state_dict)
    except:
        print('Failed to load the saved model. Proceeding with the model available in last epoch.')
    
    # get final embeddings
    node_embeddings, all_embeddings, ij_link = [], [], []
    if __task__ == 'unsupervised_learning':
        all_embeddings = model.hidden
        node_embeddings = model.hidden['layer %d'%(depth)]#inference(GNN_model, data_test, device)
    elif __task__ == 'supervised_learning':
        ij_link = inference(model, data_test, device)
    
    return {'train losses': losses_train,
            'validation losses': losses_val,
            'train accuracies': accs_train,
            'validation accuracies': accs_val,
            'best training loss': min(losses_train),
            'best training accuracy': accs_train[np.where(np.array(losses_train) == min(losses_train))[0][0]],
            'best validation loss': min(losses_val),
            'best validation accuracy': accs_val[np.where(np.array(losses_val) == min(losses_val))[0][0]],
            'inference': ij_link,
            'total time': time_total,
            'model state dict': model_state_dict,
            'model filename': model_filename,
            'all embeddings': all_embeddings,
            'node embeddings': node_embeddings}

#%% SOME HELPER FUNCTION

def get_model(parameter_dict, task):
    
    # define and initialize model
    if task == 'unsupervised_learning':
        model = GSUn(parameter_dict['adjacency_matrix'],
                     parameter_dict['node_feature_vectors'],
                     parameter_dict['gnn_depth'],
                     parameter_dict['gnn_input_layer_size'],
                     parameter_dict['gnn_hidden_layer_size'],
                     parameter_dict['gnn_output_layer_size'],
                     parameter_dict['gnn_hidden_layer_dropout'],
                     parameter_dict['device'],
                     parameter_dict['gnn_aggregator'],
                     parameter_dict['gnn_neighborhood_sample_size'])
    elif task == 'supervised_learning':
        model = NNCls(parameter_dict['classifier_layers_lan'],
                      parameter_dict['classifier_input_layer_size_lan'],
                      parameter_dict['classifier_input_layer_size_lan'],
                      parameter_dict['classifier_hidden_layer_size_lan'],
                      #
                      parameter_dict['classifier_layers_graph'],
                      parameter_dict['classifier_input_layer_size_graph'],
                      parameter_dict['classifier_input_layer_size_graph'],
                      parameter_dict['classifier_hidden_layer_size_graph'],
                      #
                      parameter_dict['classifier_output_layer_size'],
                      parameter_dict['classifier_hidden_layer_dropout_graph'],
                      parameter_dict['node_feature_vectors'],
                      parameter_dict['node_embeddings_graph'],
                      parameter_dict['device'],
                      parameter_dict['classifier_use_language_embeddings'])
    
    #
    model = utils_.init_parameters(model)
    model = model.to(parameter_dict['device'])   
    
    # count and print total number of parameters
    _ = utils_.count_params(model)
    
    # print model structure
    #utils_.print_model_structure(model)
        
    return model

def get_objective_function(parameter_dict, task):
    
    if task == 'unsupervised_learning':
        objective_f = UnsupLoss(parameter_dict['adjacency_matrix'],
                                parameter_dict['path_data_'],
                                parameter_dict['device'],
                                parameter_dict['gnn_unsupervised_random_walk_length'],
                                parameter_dict['gnn_unsupervised_positive_sample_size'],
                                parameter_dict['gnn_unsupervised_negative_sample_size'])
    elif task == 'supervised_learning':
        objective_f = nn.CrossEntropyLoss()
    
    #
    objective_f = objective_f.to(parameter_dict['device'])
    
    return objective_f

def get_solution_method(parameter_dict, model_, task):
    
    if task == 'unsupervised_learning':
        l_rate = parameter_dict['gnn_lr_start']
        l_rate_red = parameter_dict['gnn_lr_reduction']
        l_rate_red_step = parameter_dict['gnn_lr_reduction_step']
    elif task == 'supervised_learning':
        l_rate = parameter_dict['classifier_lr_start']
        l_rate_red = parameter_dict['classifier_lr_reduction']
        l_rate_red_step = parameter_dict['classifier_lr_reduction_step']
    
    # solution method
    solution_m = optim.Adam(model_.parameters(), lr = l_rate)
    
    # gradual reduction in the learning rate
    lr_scheduler = optim.lr_scheduler.StepLR(solution_m, 
                                             step_size = l_rate_red_step, 
                                             gamma = l_rate_red)
    
    return model_, solution_m, lr_scheduler
    
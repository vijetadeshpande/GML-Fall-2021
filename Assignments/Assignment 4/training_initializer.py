#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  3 09:50:37 2021

@author: vijetadeshpande
"""
import torch
import torch.nn as nn
from GSage import GraphSage
import train_utils as utils_
import torch.optim as optim
import time
import os
import sys
from tqdm import tqdm
import numpy as np
from train_and_test import train, evaluate, inference

def main(parameter_dict):
    
    # unroll
    adj_mat = parameter_dict['adjacency matrix']
    node_feature_mat = parameter_dict['BERT representations']
    depth = parameter_dict['depth']
    dim_input = parameter_dict['input layer size']
    dim_hidden = parameter_dict['hidden layer size']
    dim_output = parameter_dict['number of classes']
    device = parameter_dict['device']
    aggregator = parameter_dict['aggregator']
    sample_size = parameter_dict['sample size']
    learning_rate = parameter_dict['learning rate']
    data = parameter_dict['data']
    epochs = parameter_dict['epochs']
    model_filename = parameter_dict['model filename']
    clip = 1
    dropout = parameter_dict['dropout rate']
    lr_reduction = parameter_dict['learning rate reduction']
    lr_reduction_step = parameter_dict['learning rate reduction step']
    
    # training, validation and testing data
    data_train, data_val, data_test = data.train, data.val, data.test
    
    # define and initialize model
    model = GraphSage(adj_mat,
                      node_feature_mat,
                      depth,
                      dim_input,
                      dim_hidden,
                      dim_output,
                      dropout,
                      device,
                      aggregator,
                      sample_size)
    model = utils_.init_parameters(model)
    model = model.to(device)   
    
    #
    utils_.print_model_structure(model)
    
    # count and print total number of parameters
    _ = utils_.count_params(model)
    
    # define objective function
    obj_f = nn.CrossEntropyLoss()
    obj_f = obj_f.to(device)
    
    # define solution method
    sol_m = optim.Adam(model.parameters(), lr = learning_rate)
    
    # define scheduler function to reduce learning rate over the training period
    sch_lr = optim.lr_scheduler.StepLR(sol_m, step_size = lr_reduction_step, gamma = lr_reduction)
    
    
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
        loss_train, acc_train = train(model, data_train, sol_m, obj_f, clip, device)
        loss_val, acc_val, _ = evaluate(model, data_val, obj_f, device)
        
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
    
    # forward pass
    node_to_class = inference(model, data_test, device)
    
    # print results
    #utils_.print_test_report(loss_test, time_total)
    
    # extract node embeddings
    embeddings = {}
    for layer in model.hidden:
        embeddings[layer] = model.hidden[layer].detach().numpy()
    
    return {'train losses': losses_train,
            'validation losses': losses_val,
            'train accuracies': accs_train,
            'validation accuracies': accs_val,
            'best training loss': min(losses_train),
            'best training accuracy': accs_train[np.where(np.array(losses_train) == min(losses_train))[0][0]],
            'best validation loss': min(losses_val),
            'best validation accuracy': accs_val[np.where(np.array(losses_val) == min(losses_val))[0][0]],
            'inference': node_to_class,
            'total time': time_total,
            'model state dict': model_state_dict,
            'model filename': model_filename,
            'node embeddings': embeddings}
    
    
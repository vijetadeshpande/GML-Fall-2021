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
    depth = parameter_dict['GSage depth']
    data = parameter_dict['data']
    epochs = parameter_dict['epochs']
    model_filename = parameter_dict['model filename']
    clip = 1
    
    #
    __task__ = parameter_dict['task']
    
    #%% UN_SUPERVISED NODE EMBEDDING DEVELOPMENT
    
    # training, validation and testing data
    data_train, data_val, data_test = data.train, data.val, data.test
    
    # define and initialize model
    model = get_model(parameter_dict, __task__)
    
    # define objective function
    obj_f = get_objective_function(parameter_dict, __task__)
    
    # define solution method
    model, sol_m, sch_lr = get_solution_method(parameter_dict, model)
    
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
    
    # get final embeddings
    node_embeddings, ij_link = None, None
    if __task__ == 'unsupervised learning':
        all_embeddings = model.hidden
        node_embeddings = model.hidden['layer %d'%(depth)]#inference(GNN_model, data_test, device)
    elif __task__ == 'supervised learning':
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
    if task == 'unsupervised learning':
        model = GSUn(parameter_dict['adjacency matrix'],
                     parameter_dict['node feature mat'],
                     parameter_dict['GSage depth'],
                     parameter_dict['GSage input layer size'],
                     parameter_dict['GSage hidden layer size'],
                     parameter_dict['GSage output layer size'],
                     parameter_dict['GSage dropout'],
                     parameter_dict['device'],
                     parameter_dict['GSage aggregator'],
                     parameter_dict['GSage neighborhood sample size'])
    elif task == 'supervised learning':
        model = NNCls(parameter_dict['classifier layers'],
                 parameter_dict['node feature dim'],
                 parameter_dict['classifier input layer size'],
                 parameter_dict['classifier hidden layer size'],
                 parameter_dict['classifier output layer size'],
                 parameter_dict['classifier dropout'])
    
    #
    model = utils_.init_parameters(model)
    model = model.to(parameter_dict['device'])   
    
    # count and print total number of parameters
    _ = utils_.count_params(model)
    
    # print model structure
    utils_.print_model_structure(model)
        
    return model

def get_objective_function(parameter_dict, task):
    
    if task == 'unsupervised learning':
        objective_f = UnsupLoss(parameter_dict['adjacency matrix'],
                                parameter_dict['device'],
                                parameter_dict['GSage unsupervised random walk length'],
                                parameter_dict['GSage unsupervised positive sample size'],
                                parameter_dict['GSage unsupervised negative sample size'])
    elif task == 'supervised learning':
        objective_f = nn.CrossEntropyLoss()
    
    #
    objective_f = objective_f.to(parameter_dict['device'])
    
    return objective_f

def get_solution_method(parameter_dict, model_, task):
    
    if task == 'unsupervised learning':
        l_rate = parameter_dict['GSage learning rate']
        l_rate_red = parameter_dict['GSage learning rate reduction']
        l_rate_red_step = parameter_dict['GSage learning rate reduction step']
    elif task == 'supervised learning':
        l_rate = parameter_dict['classifier learning rate']
        l_rate_red = parameter_dict['classifier learning rate reduction']
        l_rate_red_step = parameter_dict['classifier learning rate reduction step']
        
    
    
    # solution method
    solution_m = optim.Adam(model_.parameters(), lr = l_rate)
    
    # gradual reduction in the learning rate
    lr_scheduler = optim.lr_scheduler.StepLR(solution_m, 
                                             step_size = l_rate_red_step, 
                                             gamma = l_rate_red)
    
    return model_, solution_m, lr_scheduler
    
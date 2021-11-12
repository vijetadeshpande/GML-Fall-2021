#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  3 08:24:39 2021

@author: vijetadeshpande
"""

import torch.nn
import torch
import sys
import pandas as pd
import numpy as np
import os
from copy import deepcopy
from numpy import random
from EvalMetrics import EvalMetrics
import train_utils as t_utils_

#%%
def train(model, data, optimizer, criterion, clip, device):
    
    # initialize
    model.train()
    torch.autograd.set_detect_anomaly(True)
    epoch_loss, epoch_acc = 0, 0
    update_nodes = update_neis = {}
    
    idx = -1
    for batch in data:
        idx += 1
        
        # extract source and target
        nodes_ = list(map(int, batch[0].tolist()))#
        trgs_ = batch[1].long()#.detach()#
        
        # reset gradient
        optimizer.zero_grad()
        
        # forward pass
        prediction = model(nodes_)
        
        # error calculation
        loss = criterion(prediction, trgs_)
        
        # backprop
        loss.backward(retain_graph = True)

        # gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        
        # update weights
        optimizer.step()
        
        # update loss
        epoch_loss += loss.item()
        
        # calculate other evaluation metics
        
        # update the hidden states 
        #model = update_variables(model, update_nodes, update_neis)
        
        # calculate other evaluation metics
        epoch_acc += EvalMetrics().__accuracy__(torch.argmax(prediction.clone(), dim = -1).detach().numpy(), trgs_.detach().numpy())
        
    # take average of the loss
    epoch_loss = epoch_loss / len(data)
    epoch_acc = epoch_acc / len(data)
    
    return epoch_loss, epoch_acc


#%%
def evaluate(model, data, criterion, device, teacher_forcing_ratio = 0, ignore_error:bool = False):
    
    
    # initialize
    model.eval()
    epoch_loss, epoch_acc = 0, 0       
    with torch.no_grad():
        for batch in data:

            # access the source and target sequence
            nodes_ = list(map(int, batch[0].tolist()))#
            trgs_ = batch[1].long().detach()#
            
            # forward pass
            prediction = model(nodes_)
            
            if not ignore_error:
                # error calculation
                loss = criterion(prediction, trgs_)            
            
                # update error
                epoch_loss += loss.item()
            
            # TODO: do we need to update variables during testing or not?
            
            # calculate other evaluation metics
            epoch_acc += EvalMetrics().__accuracy__(torch.argmax(prediction.clone(), dim = -1).detach().numpy(), trgs_.detach().numpy())
            
    # return a dictionary
    epoch_loss = epoch_loss/len(data)
    epoch_acc = epoch_acc / len(data)
    all_metrics = {'average epoch loss': epoch_loss}
            
    return epoch_loss, epoch_acc, all_metrics

#%%
    
def inference(model, data, device, path_ref = r'/Users/vijetadeshpande/Documents/GitHub/GraphML-Fall-2021/Assignments/Assignment 4/data'):
    
    # initialize
    model.eval()
    node_list, pred_class_list = [], []    
    
    # load reference files for converting node_idx to title and class_idx to name
    ref_ntot = pd.read_csv(os.path.join(path_ref, 'titles.txt'), sep = 'delimiter', header = None, engine = 'python').loc[:, 0].str.split(" ", 1, expand = True)
    ref_cton = pd.read_csv(os.path.join(path_ref, 'categories.txt'), sep = ' ', header = None)
    
    with torch.no_grad():
        for batch in data:

            # access the source and target sequence
            nodes_ = list(map(int, batch[0].tolist()))
            
            # forward pass
            predictions = model(nodes_)
            
            # append
            node_list += nodes_
            pred_class_list += torch.argmax(predictions.clone(), dim = -1).detach().numpy().tolist()
    
    #
    node_pred = pd.DataFrame(0, index = np.arange(len(node_list)), columns = ['node', 'node title', 'predicted class', 'predicted class name'])
    node_pred.loc[:, 'node'] = node_list
    node_pred.loc[:, 'node title'] = t_utils_.node_to_title(ref_ntot, node_list)
    node_pred.loc[:, 'predicted class'] = pred_class_list
    node_pred.loc[:, 'predicted class name'] = t_utils_.class_to_name(ref_cton, pred_class_list)
    
    return node_pred


#%%
def update_variables(model, update_nodes, update_neis):
    
    # unroll all hidden vectors (these are model variabes)
    new_tensors = {}
    for layer in model.hidden:
        new_tensors[layer] = model.hidden[layer].clone()
    
    # hidden states of the nodes
    if not update_nodes == {}:
        for layer in update_nodes:
            batch = update_nodes[layer][0]
            #new_tensor = model.hidden[layer].clone()
            new_tensors[layer][batch, :] = update_nodes[layer][1]
            #model.hidden[layer] = new_tensor
    
    # hidden states of the neighbors of the nodes, explored in different layers
    if not update_neis == {}:
        for layer in update_neis:
            for node in update_neis[layer]:
                if not node[0] == []:
                    nei_set = node[0]
                    #new_tensor = model.hidden[layer].clone()
                    new_tensors[layer][nei_set, :] = node[1]
                    #model.hidden[layer] = new_tensor
    
    # normalize the hidden vectors (layer 1 to n)
    
    
    # point hidden to new tensors
    for layer in model.hidden:
        model.hidden[layer].data = new_tensors[layer].data
    #model.hidden = new_tensors
    
    return model
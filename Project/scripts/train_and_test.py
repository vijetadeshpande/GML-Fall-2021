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
def train(model, data, optimizer, criterion, clip, device, task):
    
    if task == 'unsupervised_learning':
        loss, acc = train_unsup(model, data, optimizer, criterion, clip, device)
    elif task == 'supervised_learning':
        loss, acc = train_sup(model, data, optimizer, criterion, clip, device)
    
    return loss, acc


def train_unsup(model, data, optimizer, criterion, clip, device):
    
    # initialize
    model.train()
    torch.autograd.set_detect_anomaly(True)
    epoch_loss, epoch_acc = 0, 0
    
    idx = -1
    for batch in data:
        idx += 1
        
        # extract source and target
        nodes_ = list(map(int, batch[0].tolist()))#
        trgs_ = batch[1].long()#.detach()#
        
        # reset gradient
        optimizer.zero_grad()
        
        # forward pass
        batch_emb, all_emb = model(nodes_)
        
        # error calculation
        loss = criterion(nodes_, batch_emb, all_emb)
        
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
        #epoch_acc += EvalMetrics().__accuracy__(torch.argmax(prediction.clone(), dim = -1).detach().numpy(), trgs_.detach().numpy())
        
    # take average of the loss
    epoch_loss = epoch_loss / len(data)
    epoch_acc = epoch_acc / len(data)
    
    return epoch_loss, epoch_acc
    
def train_sup(model, data, optimizer, criterion, clip, device):
    
    # initialize
    model.train()
    torch.autograd.set_detect_anomaly(True)
    epoch_loss, epoch_acc = 0, 0
    
    idx = -1
    for batch in data:
        idx += 1
        
        # extract source and target
        nodes_ = batch[0]#
        trgs_ = batch[1].float().unsqueeze(1).to(device)#.detach()#
        
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
        prediction = (prediction > 0.5).float().squeeze(1).detach().numpy()
        _acc_, _ = EvalMetrics().__accuracy__(prediction, trgs_.squeeze(1).detach().numpy())
        epoch_acc += _acc_
        
    # take average of the loss
    epoch_loss = epoch_loss / len(data)
    epoch_acc = epoch_acc / len(data)
    
    return epoch_loss, epoch_acc


#%%

def evaluate(model, data, criterion, device, task, teacher_forcing_ratio = 0, ignore_error:bool = False):
    
    if task == 'unsupervised_learning':
        loss, acc, all_met = evaluate_unsup(model, data, criterion, device)
    elif task == 'supervised_learning':
        loss, acc, all_met = evaluate_sup(model, data, criterion, device, ignore_error)
    
    return loss, acc, all_met


def evaluate_unsup(model, data, criterion, device):
    
    
    # initialize
    model.eval()
    epoch_loss, epoch_acc = 0, 0       
    with torch.no_grad():
        for batch in data:

            # access the source and target sequence
            nodes_ = list(map(int, batch[0].tolist()))#
            trgs_ = batch[1].long().detach()#
            
            # forward pass
            batch_emb, all_emb = model(nodes_)
            
            # error calculation
            loss = criterion(nodes_, batch_emb, all_emb)         
            
            # update error
            epoch_loss += loss.item()
            
            # TODO: do we need to update variables during testing or not?
            
            # calculate other evaluation metics
            #epoch_acc += EvalMetrics().__accuracy__(torch.argmax(prediction.clone(), dim = -1).detach().numpy(), trgs_.detach().numpy())
            
    # return a dictionary
    epoch_loss = epoch_loss/len(data)
    epoch_acc = epoch_acc / len(data)
    all_metrics = {'average epoch loss': epoch_loss}
            
    return epoch_loss, epoch_acc, all_metrics    


def evaluate_sup(model, data, criterion, device, ignore_error = False):
    
    
    # initialize
    model.eval()
    epoch_loss, epoch_acc = 0, 0
    epoch_others = {'false positive': 0, 'false negative': 0, 'true positive': 0, 'true negative': 0}      
    with torch.no_grad():
        for batch in data:

            # access the source and target sequence
            nodes_ = batch[0]#
            trgs_ = batch[1].float().unsqueeze(1).to(device)
            
            # forward pass
            prediction = model(nodes_)
            
            #if not ignore_error:
            # error calculation
            loss = criterion(prediction, trgs_)            
        
            # update error
            epoch_loss += loss.item()
            
            # calculate other evaluation metics
            prediction = (prediction > 0.5).float().squeeze(1).detach().numpy()
            _acc_, _others_ = EvalMetrics().__accuracy__(prediction, trgs_.squeeze(1).detach().numpy())
            epoch_acc += _acc_
            for met in epoch_others:
                epoch_others[met] += _others_[met]
            
    # return a dictionary
    epoch_loss = epoch_loss/len(data)
    epoch_acc = epoch_acc / len(data)
    all_metrics = {'average epoch loss': epoch_loss,
                   'average epoch accuracy': epoch_acc}
    for met in epoch_others:
        all_metrics[met] = epoch_others[met]/len(data)
            
    return epoch_loss, epoch_acc, all_metrics

#%%
    
def inference(model, data, device, path_ref = r'/Users/vijetadeshpande/Documents/GitHub/GraphML-Fall-2021/Assignments/Assignment 4/data'):
    
    # initialize
    model.eval()
    ni, nj, link = [], [] , []   
    
    # load reference files for converting node_idx to title and class_idx to name
    #ref_ntot = pd.read_csv(os.path.join(path_ref, 'titles.txt'), sep = 'delimiter', header = None, engine = 'python').loc[:, 0].str.split(" ", 1, expand = True)
    #ref_cton = pd.read_csv(os.path.join(path_ref, 'categories.txt'), sep = ' ', header = None)
    
    with torch.no_grad():
        for batch in data:

            # access the source and target sequence
            nodes_ = batch[0]
            
            # forward pass
            predictions = model(nodes_)
            
            # append
            ni += nodes_.detach().numpy()[:, 0].tolist()
            nj += nodes_.detach().numpy()[:, 1].tolist()
            #link += torch.argmax(predictions.clone(), dim = -1).detach().numpy().tolist()
            link += predictions.squeeze(1).detach().numpy().tolist()
    
    #
    node_pred = pd.DataFrame(0, index = np.arange(len(ni)), columns = ['ni', 'nj', 'link'])
    node_pred.loc[:, 'ni'] = ni
    node_pred.loc[:, 'nj'] = nj
    node_pred.loc[:, 'link'] = link


    return node_pred

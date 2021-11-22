#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 19 00:22:57 2021

@author: vijetadeshpande
"""
import os
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import networkx as nx
from RandomWalk import RandWalk

class GSUnsupLoss(nn.Module):
    
    def __init__(self, 
                 adjacency_mat: np.array,
                 path_data,
                 device: torch.device,
                 random_walk_length: int = 3,
                 positive_sample_size: int = 10,
                 negative_sample_size: int = 10):
        super(GSUnsupLoss, self).__init__()
        
        # create sample set for positive and negative examples
        if os.path.exists(os.path.join(path_data, 'sample_set_negative.csv')) and os.path.exists(os.path.join(path_data, 'sample_set_positive.csv')):
            self.sample_set = {}
            self.sample_set['negative'] = pd.read_csv(os.path.join(path_data, 'sample_set_negative.csv')).iloc[:, 1:].values.astype(int)
            self.sample_set['positive'] = pd.read_csv(os.path.join(path_data, 'sample_set_positive.csv')).iloc[:, 1:].values.astype(int)
        else:
            self.sample_set = RandWalk(random_walk_length).create_sample_set(adjacency_mat)
        self.max_size_pos = self.sample_set['positive'].shape[1]
        self.max_size_neg = self.sample_set['negative'].shape[1]
        
        #
        self.sample_size_pos = positive_sample_size
        self.sample_size_neg = negative_sample_size
        self.random_walk_length = random_walk_length
        self.device = device
        
        # functions used
        self.log_sig = nn.LogSigmoid()
        
        
        return
        
    def forward(self, 
                node_batch,
                batch_emb,
                all_emb):
        
        #
        batch_emb = batch_emb.unsqueeze(1)
        
        #
        pos_emb = torch.zeros((all_emb.shape[0], self.sample_size_pos, all_emb.shape[1])).float().to(self.device)
        neg_emb = torch.zeros((all_emb.shape[0], self.sample_size_neg, all_emb.shape[1])).float().to(self.device)
        for node in node_batch:
            
            #
            rand_ = torch.randint(0, self.max_size_pos, (self.sample_size_pos,))
            pos_emb[node, :, :] = all_emb[self.sample_set['positive'][node, rand_], :]
            
            #
            rand_ = torch.randint(0, self.max_size_neg, (self.sample_size_neg,))
            neg_emb[node, :, :] = all_emb[self.sample_set['negative'][node, rand_], :]
        
        #
        pos_emb = pos_emb[node_batch, :, :]
        neg_emb = neg_emb[node_batch, :, :]
        
        # how much similar nodes numerically far?
        pos_loss = -1 * self.log_sig(torch.matmul(batch_emb, torch.transpose(pos_emb, -1, 1)))
        pos_loss = torch.sum(pos_loss, dim = -1)/(len(node_batch))
        
        # how much dissimilar nodes are numerically close?
        neg_loss = -1 * self.log_sig(-1 * torch.matmul(batch_emb, torch.transpose(neg_emb, -1, 1))) #self.sample_size_neg * self.log_sig(-1 * torch.matmul(batch_emb, torch.transpose(neg_emb, -1, 1)))
        neg_loss = torch.sum(neg_loss, dim = -1)/(len(node_batch))
        
        #
        loss = torch.sum(pos_loss + neg_loss, dim = 0)
        
        return loss


        
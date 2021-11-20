#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 19 00:22:57 2021

@author: vijetadeshpande
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import networkx as nx
from RandomWalk import RandWalk

class GSUnsupLoss(nn.Module):
    
    def __init__(self, 
                 adjacency_mat: np.array,
                 device: torch.device,
                 random_walk_length: int = 3,
                 positive_sample_size: int = 10,
                 negative_sample_size: int = 10):
        super(GSUnsupLoss, self).__init__()
        
        # create sample set for positive and negative examples
        self.sample_set = RandWalk().create_sample_set(adjacency_mat)
        self.max_size_pos = self.sample_set['positive'].shape[1]
        self.max_size_neg = self.sample_set['negative'].shape[1]
        
        #
        self.sample_size_pos = positive_sample_size
        self.sample_size_neg = negative_sample_size
        self.random_walk_length = random_walk_length
        self.device = device
        
    def forward(self, 
                node_batch,
                embeddings):
        
        #
        node_emb = embeddings[node_batch, :].unsqueeze(1).float().to(self.device)
        
        #
        rand_ = torch.randint(0, self.max_size_pos, self.sample_size_pos)
        pos_ = self.sample_set['positive'][node_batch, rand_]
        rand_ = torch.randint(0, self.max_size_neg, self.sample_size_neg)
        neg_ = self.sample_set['negative'][node_batch, rand_]
        
        #
        pos_emb = torch.zeros((node_batch, self.random_walk_length, embeddings.shape[1])).float().to(self.device)
        neg_emb = torch.zeros((node_batch, self.random_walk_length, embeddings.shape[1])).float().to(self.device)
        for node in node_batch:
            pos_emb[node, :, :] = embeddings[pos_[node, :], :]
            neg_emb[node, :, :] = embeddings[neg_[node, :], :]
            
        pos_loss = -1 * torch.nn.LogSigmoid(torch.matmul(node_emb, torch.transpose(pos_emb, -1, 1)))
        pos_loss = torch.sum(pos_loss)/(len(node_batch))
        neg_loss = -1 * self.negative_sample_size * torch.nn.LogSigmoid(-1 * torch.matmul(node_emb, torch.transpose(neg_emb, -1, 1)))
        neg_loss = torch.sum(neg_loss)/(len(node_batch))
        
        #
        loss = pos_loss + neg_loss
        
        return loss


        
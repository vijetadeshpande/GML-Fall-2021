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

class RandWalk():
    
    def __init__(self, walk_length):
        #self.adj_mat = adj_mat
        self.walk_len = walk_length
        return
    
    def bfs_(self, 
             start_node: set, 
             adj_mat: np.array,
             hops: int):
        
        #
        nodes_k_hops = start_node
        
        # if node is not present in the graph
        for node in nodes_k_hops:
            if node >= adj_mat.shape[0]:
                return set()
        
        hops -= 1
        nodes_on_walk = set()
        while hops >= 0:
            
            trg = set()
            for node in nodes_k_hops:
                try:
                    trg = trg.union(set(np.where(adj_mat[node, :] == 1)[0]))
                except:
                    continue
            
            #
            nodes_k_hops = trg
            nodes_on_walk = nodes_on_walk.union(nodes_k_hops)
            hops -= 1
        
        # upper bound on number of neighbours 
        #if self.sample_size:
        #    if len(nodes) > self.sample_size:
        #        nodes = set(random.sample(nodes, self.sample_size))
        
        
        return nodes_on_walk, nodes_k_hops
    
    def get_positive_samples(self, nodes, adj_mat):
        
        #
        sample_set = -1 * np.ones((len(nodes), 10000))
        max_c = 0
        for node in nodes:
            neigs, _ = self.bfs_(set([node]), adj_mat, self.walk_len)
            sample_set[node, 0:len(neigs)] = list(neigs)
            sample_set[node, len(neigs):] = node
            max_c = max(max_c, len(neigs))
        
        #
        sample_set = sample_set[:, 0:max_c]
        
        return sample_set
    
    def get_negative_samples(self, nodes, adj_mat):
        
        # create netx graph from adj_mat
        G = nx.from_numpy_matrix(adj_mat)
        
        #
        sample_set = -1 * np.ones((len(nodes), 10000))
        max_c = 0
        min_c = 100000
        for node in nodes:
            reachable = nx.algorithms.dag.descendants(G, node)
            not_reachable = set(nodes) - set(reachable)
            sample_set[node, 0:len(not_reachable)] = list(not_reachable)
            max_c = max(max_c, len(not_reachable))
            min_c = min(min_c, len(not_reachable))
            
            #if len(not_reachable) == 0 or len(not_reachable) < max_c:
            _, nodes_far_away = self.bfs_(set([node]), adj_mat, self.walk_len + 2)
            start_, end_ = len(not_reachable), len(not_reachable) + len(nodes_far_away)
            sample_set[node, start_:end_] = list(nodes_far_away)
            
            #
            _, nodes_far_away = self.bfs_(set([node]), adj_mat, self.walk_len + 1)
            start_ = end_
            end_ = start_ + len(nodes_far_away)
            sample_set[node, start_:end_] = list(nodes_far_away)
        
        #
        sample_set = sample_set[:, max_c]
        
        return sample_set
    
    def create_sample_set(self, adj_mat):
        
        # TODO: need to create a generalizable list for node tags
        nodes = np.arange(adj_mat.shape[0])
        
        #
        sample_set = {}
        
        #
        sample_set['positive'] = self.get_positive_samples(nodes, adj_mat)
        sample_set['negative'] = self.get_negative_samples(nodes, adj_mat)
        
        
        return sample_set
        
        
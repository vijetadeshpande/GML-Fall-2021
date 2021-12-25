#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 20 10:56:00 2021

@author: vijetadeshpande
"""

import numpy as np
import networkx as nx
import random
from tqdm import tqdm 

class RandWalk():
    
    def __init__(self, walk_length):
        #self.adj_mat = adj_mat
        self.walk_len = walk_length
        return
    
    def bfs_(self, 
             start_node: set, 
             adj_mat: np.array,
             hops: int,
             node2idx: dict,
             neighbor_map = None):
        
        #
        nodes_k_hops = start_node
        #hops -= 1
        level = 0
        nodes_on_walk = set()
        while level < hops:#self.walk_len:
            level += 1
            
            trg = set()
            for node in nodes_k_hops:
                try:
                    trg = trg.union(set(np.where(adj_mat[node, :] == 1)[0]))
                except:
                    continue
            
            #
            nodes_k_hops = trg
            nodes_on_walk = nodes_on_walk.union(nodes_k_hops)
            #hops -= 1
            
            # update neighbor map
            layer = self.walk_len - level
            layer_name = 'depth %d'%(level)
            neighbor_map[layer_name].append(list(nodes_k_hops))
        
        # upper bound on number of neighbours 
        #if self.sample_size:
        #    if len(nodes) > self.sample_size:
        #        nodes = set(random.sample(nodes, self.sample_size))
        
        
        return nodes_on_walk, (nodes_k_hops, neighbor_map)
    
    def get_positive_samples(self, adj_mat, nodes, node2idx, neighbor_map):
        
        #
        sample_set = -1 * np.ones((len(nodes), len(nodes)))
        max_c = 0
        min_c = 10000
        for node in tqdm(nodes):
            idx = node2idx[node]
            neigs, (_, neighbor_map) = self.bfs_(set([idx]), adj_mat, self.walk_len, node2idx, neighbor_map)
            sample_set[idx, 0:len(neigs)] = list(neigs)
            sample_set[idx, len(neigs):] = idx
            max_c = max(max_c, len(neigs))
            min_c = min(min_c, len(neigs))
        
        #
        sample_set = sample_set[:, 0:max_c]
        
        return sample_set.astype(int), neighbor_map
    
    def get_negative_samples(self, adj_mat, nodes, node2idx, nodes_idx, set_pos):
        
        #
        sample_set = -1 * np.ones((len(nodes), len(nodes)))
        max_c = 0
        min_c = 100000
        for _, node in tqdm(enumerate(nodes)):
            idx = node2idx[node]
            reachable_idx = set_pos[idx, :]
            not_reachable_idx = set(nodes_idx) - set(reachable_idx)
            
            if len(list(not_reachable_idx)) == 0:
                not_reachable_idx = set(random.choices(nodes_idx, k = 1))
            
            sample_set[idx, 0:len(not_reachable_idx)] = list(not_reachable_idx)
            sample_set[idx, len(not_reachable_idx):] = random.choices(list(not_reachable_idx), k = (len(nodes) - len(list(not_reachable_idx))))
            
            
            #
            max_c = max(max_c, len(not_reachable_idx))
            min_c = min(min_c, len(not_reachable_idx))
            
        #
        sample_set = sample_set[:, 0:min_c]
        
        return sample_set.astype(int)
    
    def OPTIONAL_get_negative_samples(self, adj_mat, nodes, node2idx):
        
        
        
        # create netx graph from adj_mat
        G = net #nx.from_numpy_matrix(adj_mat, create_using = nx.DiGraph)
        nodes = list(G.nodes)
        
        #
        sample_set = -1 * np.ones((len(nodes), len(nodes)))
        max_c = 0
        min_c = 100000
        for _, node in enumerate(nodes):
            idx = node2idx[node]
            reachable = nx.algorithms.dag.descendants(G, node)
            not_reachable = set(nodes) - set(reachable)
            sample_set[idx, 0:len(not_reachable)] = list(not_reachable)
            sample_set[idx, len(not_reachable):] = random.choices(list(not_reachable), k = (len(nodes) - len(list(not_reachable))))
            
            
            #
            max_c = max(max_c, len(not_reachable))
            min_c = min(min_c, len(not_reachable))
            
        #
        #sample_set = sample_set[:, max_c]
        
        return sample_set
    
    def create_sample_set(self, adj_mat, net, node2idx = None):
        
        # TODO: need to create a generalizable list for node tags
        nodes = list(net.nodes)#np.arange(np.array(adj_mat).shape[0])
        nodes_idx = [node2idx[i] for i in nodes]
        
        #
        sample_set = {}
        neighbor_map = {}
        for layer in range(1, self.walk_len+1, 1):
            layer_name = 'depth %d'%(layer)
            neighbor_map[layer_name] = []
        
        
        #
        sample_set['positive'], neighbor_map = self.get_positive_samples(adj_mat, nodes, node2idx, neighbor_map)
        sample_set['negative'] = self.get_negative_samples(adj_mat, nodes, node2idx, nodes_idx, sample_set['positive'])
        
        return sample_set, neighbor_map
        
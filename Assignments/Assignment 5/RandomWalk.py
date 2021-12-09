#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 20 10:56:00 2021

@author: vijetadeshpande
"""

import numpy as np
import networkx as nx
import random

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
        sample_set = -1 * np.ones((len(nodes), len(nodes)))
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
        G = nx.from_numpy_matrix(adj_mat, create_using = nx.DiGraph)
        
        #
        sample_set = -1 * np.ones((len(nodes), len(nodes)))
        max_c = 0
        min_c = 100000
        for node in nodes:
            reachable = nx.algorithms.dag.descendants(G, node)
            not_reachable = set(nodes) - set(reachable)
            sample_set[node, 0:len(not_reachable)] = list(not_reachable)
            sample_set[node, len(not_reachable):] = random.choices(list(not_reachable), k = (len(nodes) - len(list(not_reachable))))
            
            
            #
            max_c = max(max_c, len(not_reachable))
            min_c = min(min_c, len(not_reachable))
            
        #
        #sample_set = sample_set[:, max_c]
        
        return sample_set
    
    def create_sample_set(self, adj_mat):
        
        # TODO: need to create a generalizable list for node tags
        nodes = np.arange(np.array(adj_mat).shape[0])
        
        #
        sample_set = {}
        
        #
        sample_set['positive'] = self.get_positive_samples(nodes, adj_mat)
        sample_set['negative'] = self.get_negative_samples(nodes, adj_mat)
        
        
        return sample_set
        
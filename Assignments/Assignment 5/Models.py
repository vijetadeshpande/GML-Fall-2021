#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  1 10:23:58 2021

@author: vijetadeshpande
"""


from collections import OrderedDict
import torch
import torch.nn as nn
import random
from collections import defaultdict
import pickle
import os
import numpy as np
import pandas as pd
import tensorflow as tf
from copy import deepcopy

class GSageUnsup(nn.Module):
    
    def __init__(self, 
                 graph: list,
                 nodes_feat: pd.DataFrame, 
                 depth: int,
                 dim_feature: int,                  # say 300
                 dim_hidden: list,                  # say [256, 128, 64]
                 dim_output: int,                   # this is 26
                 dropout: float,
                 device: torch.device,
                 aggregator: str = 'mean',
                 sample_size: list = [5, 7, 10, 10, 10, 10, 10]):
        super(GSageUnsup, self).__init__()
        
        #
        assert len(dim_hidden) == depth
        assert len(sample_size) == depth
        
        #
        self.nodes = len(graph)
        self.depth = min(7, depth)
        self.aggregator_type = aggregator
        self.sample_size = sample_size
        self.device = device
        dim_hidden = [dim_feature] + dim_hidden
        
        # create neural network
        blocks = []
        self.hidden = {}
        self.neighbor_map = {}
        for layer in range(1, self.depth + 1, 1):
            
            # layer properties
            name_layer, name_linear, name_act, name_norm, name_drp = 'layer %d'%(layer), 'linear %d'%(layer), 'activation %d'%(layer), 'norm %d'%(layer), 'dropout %d'%(layer)
            dim_in, dim_out = dim_hidden[layer-1] + dim_hidden[layer-1], dim_hidden[layer]
            
            # One block of convolution
            linear_ = (name_linear, nn.Linear(dim_in, dim_out))
            activ_ = (name_act, nn.ReLU())
            norm_ = (name_norm, nn.LayerNorm(dim_out))
            drop_ = (name_drp, nn.Dropout(dropout))
            block = (name_layer, nn.Sequential(OrderedDict([linear_, activ_, norm_, drop_])))
            blocks.append(block)
            
            # create space to store embeddings
            if layer == 1:
                self.hidden['layer %d'%(layer-1)] = torch.tensor(nodes_feat.values).float().to(device)
            self.hidden[name_layer] = torch.rand(self.nodes, dim_hidden[layer]).float().to(device)
            #self.hidden[name_layer] = torch.zeros(self.nodes, dim_hidden[layer]).float().to(device)
            
        
        #
        self.aggregator = Aggregator(self, graph, self.depth, self.aggregator_type, sample_size)
        self.graph_network = nn.ModuleDict(OrderedDict(blocks))
        #self.classifier = nn.Sequential(OrderedDict([('layer out', nn.Linear(dim_hidden[-1], dim_output)), 
        #                                             ('activation out', nn.Softmax(dim = -1))]))
        
        return
    
    def robbins_monro_approximation(self, estimate, sample, alpha = 0.9):
        
        estimate = estimate - alpha * (estimate - sample)
        
        return estimate
    
    
    def forward(self, node_batch: list):
        
        # unroll all hidden vectors (these are model variabes)
        new_tensors = {}
        for layer in self.hidden:
            new_tensors[layer] = self.hidden[layer].clone()
            
        # iterate over layers
        for layer in range(1, self.depth + 1, 1):
            layer_cur = 'layer %d'%(layer)
            layer_pre = 'layer %d'%(layer-1)
            
            
            # aggregate the information of neighbors at layer-hop/s
            aggregated = self.aggregator.aggregate(node_batch, self.depth - (layer-1), layer_pre, self.sample_size[-layer])
            #update_nei['layer %d'%(layer)] = hidden_new
            
            # concatenate and pass through the network
            hidden_at_k = self.graph_network[layer_cur](torch.cat((aggregated, self.hidden[layer_pre][node_batch]), -1).float().to(self.device))
            
            # save the value of hidden for current node batch
            #hidden_at_k = self.robbins_monro_approximation(new_tensors[layer_cur][node_batch, :].float().to(self.device), hidden_at_k)
            new_tensors[layer_cur][node_batch, :] = hidden_at_k
            
        
        # after the for loop msg_pass should have shape: (N, dim_out) Note N is same as input        
        #prediction = self.classifier(hidden_at_k)
        
        # update embeddings for every layer
        for layer in new_tensors:
            self.hidden[layer].data = new_tensors[layer].data
        
        
        return hidden_at_k, self.hidden[layer]
    

class Aggregator():
    
    def __init__(self, 
                 GNN: GSageUnsup,
                 adj_mat:list, 
                 depth:int,
                 aggregator_type:str = 'mean',
                 sample_size:int = None):
        
        # we need to access the hidden states for aggregation
        self.GNN = GNN
        
        # attributes
        self.sampler = NeighSampler(adj_mat, sample_size)
        self.aggregator_type = aggregator_type
        self.depth = depth
        self.alpha_rb = 0.9
        
        # create neighbor map for the current depth
        self.neighbor_map = {}
        for layer in range(1, self.GNN.depth + 1, 1):
            # TODO: np.arange(self.nodes) this may not map to unique nodes
            name_layer = 'layer %d'%(layer)
            self.neighbor_map[name_layer] = self.sampler.get_neig(np.arange(self.GNN.nodes), layer)
        
        
        return
    
    def sample_set(self, node_batch: list, depth: int, sample_size: int):
        
        nodes, neighs, no_neigh, fill_idx = self.sampler.sample_neig(node_batch, self.neighbor_map, depth, sample_size)
        
        return nodes, neighs, no_neigh, fill_idx
    
    
    def get_seg_id(self, nodes, no_neigh):
        
        
        seg_id = []
        nodes_unique = []
        idx = -1
        for node in nodes:
            if not node in nodes_unique:
                idx += 1
                nodes_unique.append(node)
                seg_id.append(idx)
            else:
                if not node in no_neigh:
                    seg_id.append(idx)
                
            
        return seg_id, nodes_unique
    
    
    def aggregate(self, node_batch: list, 
                  depth: int, 
                  layer: int,
                  sample_size: int):
        
        # take a sample set
        nodes, neighs, no_neigh, fill_idx = self.sample_set(node_batch, depth, sample_size)
        seg_id, nodes_unique = self.get_seg_id(nodes, no_neigh)
        seg_id = torch.tensor(seg_id).long().to(self.GNN.device)
        neighs = self.GNN.hidden[layer][neighs, :].clone().float().to(self.GNN.device)
        
        # aggreagte
        agg_ = torch.tensor(tf.math.unsorted_segment_mean(neighs, seg_id, len(nodes_unique)).numpy()).float().to(self.GNN.device)
        
        #
        agg = torch.zeros(len(node_batch), agg_.shape[1]).float().to(self.GNN.device)
        agg[fill_idx, :] = agg_
        
        return agg
    
    
class NeighSampler():
    #super().__init__()
    
    def __init__(self, adj_mat: list, sample_size: int = None):
        
        self.adj_mat = np.array(adj_mat)
        self.sample_size = sample_size
        
    def bfs_(self, nodes: set, hops: int):
        
        """
        This function returns neighbors of a node at 'hops' distance away.
        
        Initially the nodes list has only one element in it, i.e. at the end
        we get neighbors of a specific node at a specified distance
        
        Type:
            inputs:
                nodes: set
                hops: inetger
            outputs:
                nodes: set
        
        Shape:
            inputs:
                nodes.shape = [1, 0]
                hops.shape = integer-type
            output:
                nodes.shape = [K, 0]
        
        """
        # if node is not present in the graph
        for node in nodes:
            if node >= self.adj_mat.shape[0]:
                return set()
        
        hops -= 1
        while hops >= 0:
            
            trg = set()
            for node in nodes:
                try:
                    trg = trg.union(set(np.where(self.adj_mat[:, node] == 1)[0]))
                except:
                    continue
            
            #
            nodes = trg
            hops -= 1
        
        # upper bound on number of neighbours 
        #if self.sample_size:
        #    if len(nodes) > self.sample_size:
        #        nodes = set(random.sample(nodes, self.sample_size))
        
        
        return nodes
    
    def get_neig(self, nodes_batch: list, hops: int):
        
        neigh_ = defaultdict(set)
        for node in nodes_batch:
            neigh_[node] = self.bfs_(set([node]), hops)
        
        return neigh_
    
    def sample_neig(self, node_batch: list, neigh_map: dict, depth: int, sample_size: int):
        
        neighs, nodes, no_neigh = [], [], []
        fill_idx = []
        depth_ = 'layer %d'%(depth)
        idx = -1
        for node in node_batch:
            idx += 1
            neigh = neigh_map[depth_][node]
            if len(neigh) > sample_size:
                neigh = list(random.sample(neigh, sample_size))

            neighs += neigh
            nodes += [node] * len(neigh)
            if not len(neigh) > 0:
                no_neigh += [node]
            else:
                fill_idx += [idx]

        
        return nodes, neighs, no_neigh, fill_idx


class NNClassifier(nn.Module):
    
    def __init__(self,
                 num_layers_lan: int,
                 dim_emb_lan: int,
                 dim_input_lan: int,
                 dim_hidden_lan: list,
                 num_layers_graph: int,
                 dim_emb_graph: int,
                 dim_input_graph: int,
                 dim_hidden_graph: list,
                 dim_output: int,
                 dropout: float,
                 node_embeddings_lan: torch.tensor,
                 node_embeddings_graph: torch.tensor,
                 device: str,
                 use_language_embeddings = False):
        super(NNClassifier, self).__init__()
        
        # save the attributes
        self.node_embeddings = {'language': torch.tensor(node_embeddings_lan.values).float().to(device),
                                'graph': torch.tensor(node_embeddings_graph.values).float().to(device)}
        self.use_language_embeddings = use_language_embeddings
        self.device = device
        
        #
        dim_hidden_lan = [dim_input_lan*2] + dim_hidden_lan #+ [dim_output]
        dim_hidden_graph = [dim_input_graph*2] + dim_hidden_graph
        
        #
        blocks = {'language': [], 'graph': []}
        
        if use_language_embeddings:
            for layer in range(1, num_layers_lan+1, 1):
                
                # Layer for processing doc embeddings
                layer_name = 'language layer %d'%(layer)
                linear_ = ('linear', nn.Linear(dim_hidden_lan[layer-1], dim_hidden_lan[layer]))
                activ_ = ('activation', nn.ReLU())
                norm_ = ('normalization', nn.LayerNorm(dim_hidden_lan[layer]))
                drop_ = ('dropout', nn.Dropout(dropout))
                block_lan = (layer_name, nn.Sequential(OrderedDict([linear_, activ_, norm_, drop_])))
                
                #
                blocks['language'].append(block_lan)
        
        for layer in range(1, num_layers_graph+1, 1):
            # Layer for processing node embeddings
            layer_name = 'graph layer %d'%(layer)
            linear_ = ('linear', nn.Linear(dim_hidden_graph[layer-1], dim_hidden_graph[layer]))
            activ_ = ('activation', nn.ReLU())
            norm_ = ('normalization', nn.LayerNorm(dim_hidden_graph[layer]))
            drop_ = ('dropout', nn.Dropout(dropout))
            block_graph = (layer_name, nn.Sequential(OrderedDict([linear_, activ_, norm_, drop_])))
            
            #
            blocks['graph'].append(block_graph)
        
        
        # define the full classifier
        self.lin_transform_lan = nn.Sequential(OrderedDict(blocks['language']))
        self.lin_transform_graph = nn.Sequential(OrderedDict(blocks['graph']))
        
        # define last layer for classification
        dim_input_end = (dim_hidden_lan[-1] + dim_hidden_graph[-1]) if use_language_embeddings else dim_hidden_graph[-1]
        linear_ = ('linear', nn.Linear(dim_input_end, dim_output))
        activ_ = ('activation', nn.Softmax(dim = -1))
        self.classifier = nn.Sequential(OrderedDict([linear_, activ_]))
            
        
        return
    
    def get_signal(self, node_pairs, emb_type):
        
        ni_ = self.node_embeddings[emb_type][node_pairs[:, 0].detach().numpy().tolist(), :].float().to(self.device)
        nj_ = self.node_embeddings[emb_type][node_pairs[:, 1].detach().numpy().tolist(), :].float().to(self.device)
        signal_ = torch.cat((ni_, nj_), dim = -1).float().to(self.device)
        
        return signal_
    
    def forward(self, node_pairs):
        
        # linear tranformation of graph embeddings
        sig_graph = self.get_signal(node_pairs, 'graph')
        sig_graph = self.lin_transform_graph(sig_graph)
        
        # linear tranformation of language embeddings
        if self.use_language_embeddings:
            sig_lan = self.get_signal(node_pairs, 'language')
            sig_lan = self.lin_transform_lan(sig_lan)
            
            #
            sig_node_pair = torch.cat((sig_graph, sig_lan), dim = -1).float().to(self.device)
        
        else:
            sig_node_pair = sig_graph
        
        #
        prediction = self.classifier(sig_node_pair)
        
        return prediction

##
#with open(os.path.join('data', 'adj_mat.p'), 'rb') as fp:
#    graph_ = pickle.load(fp)

#feat_ = pd.read_csv(os.path.join('data', 'bert-base-uncased_node_to_features.csv'))

#sampler_n = NeighSampler(graph)
#nei_ = sampler_n.get_neig([1165, 66, 76, 888, 3245, 128, 10000], 3)
#for src in nei_:
#    print(nei_[src])

# 
#device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
#model_ = GSage(graph_, feat_.iloc[:, 2:], 2, 768, [512, 128, 64], 25, device)
#pred_x = model_([1165, 34, 100, 567])
    
    
#for param in model_.parameters():
#    print(param)
    
#for layer in model_.network:
#    model_.network[layer].train()

#for name, param in model_.named_parameters():
#    if param.requires_grad:
#        print(name, param.data.shape)
    
    
    
    
    
    
    
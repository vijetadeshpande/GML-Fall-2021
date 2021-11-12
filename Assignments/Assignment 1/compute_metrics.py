#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 18 16:43:07 2021

@author: vijetadeshpande
"""

import numpy as np
import pandas as pd
import get_graph as get_graph
import networkx as nx
from collections import defaultdict
from copy import deepcopy
from tqdm import tqdm
from collections import OrderedDict
import seaborn as sns

def get_density(adj_):
    
    # calculate size of the graph
    m_ = adj_.shape[0]
    
    # calculate number of 1s in the matrix
    edges_ = adj_.sum().sum()
    
    # calculate density
    dens_ = np.divide(edges_, (np.power(m_, 2) - m_))
    
    return dens_

def get_node_centrality(adj_, node_ = None):
    
    if not node_ == None:
        node_deg = adj_[node_][:].sum()
    else:
        node_deg = adj_.sum(0)
    
    return node_deg

def get_neighbours(node_, adj_):
    
    row_ = adj_[node_]
    neig_ = np.where(row_ == 1)[0]
    
    return neig_

def bfs_traversal(node_, adj_, visits = {}):
    
    # initialize queue of nodes that needs to be visited and explored
    queue_ = []
    queue_.append(node_)
    
    # now visit and explore every node 
    cons = []
    while queue_:
        cur_node = queue_.pop(0)
        
        # get neighbours of the current node and append to the queue
        if not cur_node in visits:
            
            # get neighbour nodes only if the node is not visited
            neighs_ = get_neighbours(cur_node, adj_)
            for neigh_ in neighs_:
                queue_.append(neigh_)
        
            # update visit
            visits[cur_node] = 1
            cons.append(cur_node)
    
    return visits, cons

def find_connected_components(adj_):
    
    # get list of nodes to iterate through 
    nodes = np.arange(adj_.shape[0])
    
    # initialize dictionary for visits and connected comp
    con_coms = {} # key = # of con_com and value = list of nodes
    visits = {} # key = node # and value = # of con_com
    con_com_n = -1
    
    # iterate through all the nodes
    for node in nodes:
        if not node in visits:
            con_com_n += 1
            
            # travel the graph in bfs manner from the node = node
            visits, node_cons = bfs_traversal(node, adj_, visits)
            
            # update the connected components
            con_coms[con_com_n] = node_cons
    

    return con_coms

def get_number_of_connected_components(adj_):
    
    #
    con_comps = find_connected_components(adj_)
    
    return len(con_comps)


def get_node_to_component_map(con_comps):
    
    map_out = {}
    for key_ in con_comps:
        nodes = con_comps[key_]
        for node in nodes:
            map_out[node] = key_
    
    return map_out

def convert_adj_to_dist(adj_):
    
    dist_ = deepcopy(adj_)
    np.place(dist_, dist_ == 0, 1000)
    np.fill_diagonal(dist_, 0)
    
    return dist_

def compute_cost(start_, end_, cost_, dist_):
    
    #
    cost_end = cost_[start_] + dist_[start_][end_]
    
    return cost_end
    

def dijkstra(start_, dist_, visits = {}, node_to_comp = {}):
    
    #
    nodes_n = dist_.shape[0]
    cost_ = 1000 * np.ones(nodes_n)
    cost_[start_] = 0
    
    #
    queue_ = [(start_, start_)] # this is initial query
    
    while queue_:
        
        #
        s_, e_ = queue_.pop(0)
        if not (s_, e_) in visits:
            
            # find neighbours and append
            neigh_ = get_neighbours(e_, dist_)
            for n_ in neigh_:
                queue_.append((e_, n_))
        
            # check if both nodes are in the same component of graph
            if (node_to_comp[s_] == node_to_comp[e_]):
                cost_e_ = compute_cost(s_, e_, cost_, dist_)
            else: 
                if node_to_comp[s_] != node_to_comp[e_]:
                    cost_e_ = 1000
                elif s_ == e_:
                    cost_e_ = 0
        
            #
            visits[(s_, e_)] = visits[(e_, s_)] = 1
            if cost_[e_] > cost_e_:
                cost_[e_] = cost_e_
    
    
    return cost_, visits    

def get_shortest_path_matrix(adj_):
    
    
    # get connected components (will be needed later to skip some queries)
    conn_comps = find_connected_components(adj_)
    node_to_comp = get_node_to_component_map(conn_comps)
    
    # convert adj_ to matrix of node-to-node distance
    dist_ = convert_adj_to_dist(adj_)
    
    # total nodes
    nodes_n = dist_.shape[0]
    
    # it is reciprocal of the average of SP to every other node
    short_paths = pd.DataFrame(np.multiply(-1, np.ones((nodes_n, nodes_n))))
    visits = {}
    
    #
    for src_ in range(nodes_n):
        
        #
        visits = {}
        cost_from_src_, visits = dijkstra(src_, dist_, visits, node_to_comp)
        short_paths.iloc[src_, :] = cost_from_src_
        #short_paths.iloc[src_:, src_] = cost_from_src_[src_:]

    #
    short_paths = short_paths.to_numpy()
    
    return short_paths

def calculate_closeness_centrality(sp_):
    
    #
    node_degree = len(np.where(sp_ != 1000)[0])
    total_nodes = len(sp_)
    
    #
    sp_sum = sp_.sum()
    
    #
    #cls_centrality = np.multiply(np.divide((node_degree - 1), (total_nodes - 1)),  np.divide(node_degree - 1, sp_sum))
    cls_centrality = np.multiply(1,  np.divide(total_nodes - 1, sp_sum))
    
    return cls_centrality

def get_closeness_centrality(adj_, node_ = None):
    
    # 
    short_paths = get_shortest_path_matrix(adj_)
    nodes_n = adj_.shape[0]
    
    #
    if node_ != None:
        #
        avg_sp = np.mean(short_paths[node_][:], 0)
    else:
        avg_sp = np.mean(short_paths, 0)

    # 
    cls_cen = []
    for node in range(short_paths.shape[0]):
        cls_cen.append(calculate_closeness_centrality(short_paths[node][:]))

    
    return cls_cen


"""
path = r'/Users/vijetadeshpande/Documents/GitHub/GraphML-Fall-2021/Assignments/Assignment 1/net-sample.txt'


df_con = get_graph.read_txt(path)
adj_mat = get_graph.get_adjacency(df_con)
get_graph.plot_graph(adj_mat)

# calculate diameter
rows, cols = np.where(adj_mat == 1)
edges = zip(rows.tolist(), cols.tolist())
gr = nx.Graph()
gr.add_edges_from(edges)

#
aaa = nx.algorithms.centrality.closeness_centrality(gr)
aaa = OrderedDict(sorted(aaa.items()))
aaa_nx = []
for i in aaa:
    aaa_nx.append(aaa[i])

#
aaa_ = list(get_closeness_centrality(adj_mat))

# plot two closeness centralities
plot_df = pd.DataFrame(0, index = np.arange(2 * len(aaa_)), columns = ['Closeness centrality', 'Type', 'Node'])
plot_df.loc[:len(aaa_)-1, 'Closeness centrality'] = aaa_nx
plot_df.loc[:len(aaa_)-1, 'Type'] = 'NetworkX'
plot_df.loc[len(aaa_):, 'Closeness centrality'] = aaa_
plot_df.loc[len(aaa_):, 'Type'] = 'Calculated'
plot_df.loc[:, 'Node'] = 2 * list(np.arange(len(aaa_)))

sns.lineplot(data = plot_df, x = 'Node', y = 'Closeness centrality', hue = 'Type')
"""







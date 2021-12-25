#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  9 01:51:28 2021

@author: vijetadeshpande
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import sys
import os
import networkx as nx
import networkx.classes.function as F
import networkx.drawing.nx_pylab as D
from tqdm import tqdm
import json
import random
from sklearn.model_selection import train_test_split

# locations to import code from
path_import = []
path_file = os.path.dirname(os.path.abspath(__file__))
path_project = os.path.abspath(os.path.join(path_file, os.pardir))
path_data = os.path.join(path_project, 'Data')
path_import.append(os.path.join(path_project, 'decagon', 'polypharmacy'))
for i in path_import:
    sys.path.insert(1, i)

# import other scripts
from utility import *

if False:
    # read the data
    combo2stitch, combo2se, se2name = load_combo_se(path_data)
    net, node2idx = load_ppi(path_data)
    stitch2se, se2name_mono = load_mono_se(path_data)
    stitch2proteins = load_targets(path_data, fname='bio-decagon-targets-all.csv')
    se2class, se2name_class = load_categories(path_data)
    se2name.update(se2name_mono)
    se2name.update(se2name_class)


# =============================================================================
# PROTEIN-PROTEIN GRAPH
# =============================================================================
if False:
    F.info(net)
    ppi_n = F.number_of_nodes(net)
    ppi_graph = pd.DataFrame(0, index = np.arange(ppi_n), columns = ['ni', 'Degree of node'])
    ppi_graph.loc[:, ['ni', 'Degree of node']] = list(F.degree(net))#list(F.nodes(net))
    ppi_graph['Degree of node'].describe()
    plt.figure()
    sns.histplot(data = ppi_graph, x = 'Degree of node')
    plt.savefig('PPI_degree_distribution.png')
    
    # more stats
    comp_n = nx.algorithms.components.number_connected_components(net)
    comp_l = max(nx.algorithms.components.connected_component_subgraphs(net), key=len)
    #comp_l_dia = nx.algorithms.distance_measures.diameter(comp_l)
    comp_l_dens = F.density(comp_l)

#
#D.draw(comp_l)

# =============================================================================
# DRUG-DRUG GRAPH
# =============================================================================
if False:
    count = -1
    ddi_type = pd.DataFrame(0, index = np.arange(len(combo2se)), columns = ['ci', 'cj', 'n_se'])
    for comb in tqdm(combo2stitch):
        count += 1    
        #
        ddi_type.loc[count, ['ci', 'cj']] = combo2stitch[comb]
        ddi_type.loc[count, 'n_se'] = len(combo2se[comb])

    unique_d = set(ddi_type.iloc[:, 0].unique().tolist()).union(set(ddi_type.iloc[:, 1].unique().tolist()))
    unique_d = sorted(list(unique_d))
    
    #
    d2idx = {}
    for idx, val in enumerate(unique_d):
        d2idx[val] = idx

    # adjacency matrix
    adj_mat = np.zeros((len(unique_d), len(unique_d)))
    for idx in tqdm(ddi_type.index):
        ci, cj = ddi_type.loc[idx, 'ci'], ddi_type.loc[idx, 'cj']
        i, j = d2idx[ci], d2idx[cj]
        adj_mat[i, j] = 1
        adj_mat[j, i] = 1
    adj_mat = adj_mat.astype(int).tolist()
    adj_mat = np.array(adj_mat)
    
    #
    filename_= os.path.join(path_data, 'D_CC_adj_mat.json')
    with open(filename_, 'w') as f:
        json.dump(adj_mat, f)
    
    filename_ = os.path.join(path_data, 'CtoIndex.json')
    with open(filename_, 'w') as f:
        json.dump(d2idx, f)

#ddi_type['Number of side-effects associated with a pair of drugs'] = ddi_type.loc[:, 'n_se'].values
#plt.figure()
#sns.histplot(data = ddi_type, x = 'Number of side-effects associated with a pair of drugs')
#plt.savefig('Side-effect_distribution.png')

    for i in stitch2se:
        print(i)
        print(stitch2se[i])
        print('\n')

# =============================================================================
# DRUG-PROTEIN GRAPH
# =============================================================================
if True:
    
    # read 
    embeddings_pp = pd.read_csv(os.path.join(path_data, 'ProteinNodeEmbeddings_GSage.csv')).iloc[:, 1:]
    embeddings_cc = pd.read_csv(os.path.join(path_data, 'DrugNodeEmbeddings_se_GSage.csv')).iloc[:, 1:]
    median_trg_protein = 5
    neg_sample_size = 2
    
    # read mapper
    with open(os.path.join(path_data, 'CtoIndex.json'), 'rb') as f:
        node2idx_c = json.load(f)
    all_c = sorted(list(stitch2proteins.keys()))
    node2idx_c_full = {}
    for idx, val in enumerate(all_c):
        node2idx_c_full[val] = idx
    
    # chage embedding shape
    z_pp = embeddings_pp
    z_cc = embeddings_cc
    z_cp = pd.DataFrame(0, index = np.arange(len(node2idx_c)), columns = np.arange(256))
    
    
    # read pos and negative sample set
    pos_set = pd.read_csv(os.path.join(path_data, 'CC_sample_set_positive.csv')).iloc[:, 1:]
    neg_set = pd.read_csv(os.path.join(path_data, 'CC_sample_set_negative.csv')).iloc[:, 1:]
    
    #
    dataset = pd.DataFrame(0, index = np.arange(500000), columns = ['ci', 'cj', 'edge'])
    count = 0
    for chem in node2idx_c:
        idx = node2idx_c[chem]
        
        #
        pos_samples = list(set(pos_set.iloc[idx, :].values.tolist()))
        s_, e_ = count, count+len(pos_samples)
        dataset.loc[s_:e_-1, 'ci'] = idx
        dataset.loc[s_:e_-1, 'cj'] = pos_samples
        dataset.loc[s_:e_-1, 'edge'] = 1
        
        #
        count = e_
        neg_samples = list(set(neg_set.iloc[idx, :].values.tolist()))
        if len(neg_samples) > neg_sample_size:
            neg_samples = random.choices(neg_samples, k = neg_sample_size)
        s_, e_ = count, count+len(neg_samples)
        dataset.loc[s_:e_-1, 'ci'] = idx
        dataset.loc[s_:e_-1, 'cj'] = neg_samples
        dataset.loc[s_:e_-1, 'edge'] = 0
        count = e_
        
        # z_cp
        proteins = list(stitch2proteins[chem])
        sample_number = min(median_trg_protein, len(proteins))
        protein_samples = random.choices(proteins, k = sample_number)
        if protein_samples != []:
            z = np.zeros((256))
            for protein in protein_samples:
                if protein in node2idx:
                    idx_p = node2idx[protein]
                    z += embeddings_pp.iloc[idx_p, :].values
            
            #
            z = np.divide(z, sample_number)
        
            #
            idx_c = node2idx_c[chem]
            z_cp.iloc[idx_c, :] = z
    
    # save
    pd.DataFrame(z_cp).to_csv(os.path.join(path_data, 'z_cp.csv'))
    pd.DataFrame(z_cc).to_csv(os.path.join(path_data, 'z_cc.csv'))
    pd.DataFrame(z_pp).to_csv(os.path.join(path_data, 'z_pp.csv'))
    
    # create dataset for training
    dataset = dataset.iloc[0:count, :]
    dataset = dataset.sample(frac=1).reset_index(drop=True)
    
    # train, val, test sets
    X_train, X_val, y_train, y_val = train_test_split(dataset.loc[:, ['ci', 'cj']], dataset.loc[:, ['edge']], test_size = 0.2)
    X_train['target'] = y_train.values
    X_val['target'] = y_val.values
    
    # save
    X_train.to_csv(os.path.join(path_data, 'train_GSage_supervised.csv'))
    X_val.to_csv(os.path.join(path_data, 'val_GSage_supervised.csv'))
    
    
    
    
    dpi_graph = pd.DataFrame(0, index = np.arange(len(stitch2proteins)), columns = ['ci' ,'Number of target proteins'])
    count = -1
    for i in stitch2proteins:
        count += 1
        dpi_graph.loc[count, 'ci'] = i
        dpi_graph.loc[count, 'Number of target proteins'] = len(set(stitch2proteins[i]))
    dpi_graph.describe()
    plt.figure()
    sns.histplot(data = dpi_graph, x = 'Number of target proteins')
    plt.savefig('DPI_taget_proteins_per_drug.png')
    
    # measure the overlap of the target proteins
    overlaps = []
    for i in ddi_type.index:
        c1 = ddi_type.loc[i, 'ci']
        c2 = ddi_type.loc[i, 'cj']
        if (c1 in stitch2proteins) and (c2 in stitch2proteins):
            t1 = stitch2proteins[c1]
            t2 = stitch2proteins[c2]
            uni_ = set(t1).union(set(t2))
            int_ = set(t1).intersection(set(t2))
            overlap_percentage = len(int_)/len(uni_)
            overlaps.append(overlap_percentage)
    overlap_df = pd.DataFrame(0, index = np.arange(len(overlaps)), columns = ['x', 'Target protein overlap percentage for pair of drugs'])
    overlap_df['Target protein overlap percentage for pair of drugs'] = overlaps
    overlap_df['Target protein overlap percentage for pair of drugs'].describe()
    plt.figure()
    sns.histplot(data = overlap_df, x = 'Target protein overlap percentage for pair of drugs', kde = True)
    plt.savefig('target_protein_overlap.png')


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

count = -1
ddi_type = pd.DataFrame(0, index = np.arange(len(combo2se)), columns = ['ci', 'cj', 'n_se'])
for comb in combo2stitch:
    count += 1    
    #
    ddi_type.loc[count, ['ci', 'cj']] = combo2stitch[comb]
    ddi_type.loc[count, 'n_se'] = len(combo2se[comb])
ddi_type['Number of side-effects associated with a pair of drugs'] = ddi_type.loc[:, 'n_se'].values
plt.figure()
sns.histplot(data = ddi_type, x = 'Number of side-effects associated with a pair of drugs')
plt.savefig('Side-effect_distribution.png')

for i in stitch2se:
    print(i)
    print(stitch2se[i])
    print('\n')

# =============================================================================
# DRUG-PROTEIN GRAPH
# =============================================================================
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


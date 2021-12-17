#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 15 15:26:09 2021

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
from TextEncoder import EncBERT, EncMedSpacy

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
# NODE EMBEDDINGS FOR DRUGS
# =============================================================================

# STEPS
# 1. Get the representative class name (SE name) for every drug
# 2. Pass it through encoder (either BioBERT or MedSpacy-embeddings)

# list of side effects for every drug


# create encoder instance
encoder = EncBERT()
embeddings = {}
for chem in tqdm(stitch2se):
    vector = np.zeros((768,))
    for se in stitch2se[chem]:
        try:
            se_category = se2name[se]
            vector += encoder(se_category)
        except:
            continue

    
    # take average
    vector /= len(stitch2se[chem])
    
    # save
    embeddings[chem] = vector
 
#
for chem in embeddings:
    embeddings[chem] = embeddings[chem].tolist()
       
#
with open('/Users/vijetadeshpande/Documents/GitHub/GML-Fall-2021/Project/Data/DrugNodeEmbedding.json', 'w') as f:
    json.dump(embeddings, f, indent = 4, sort_keys = True)


# =============================================================================
# NODE EMBEDDINGS FOR PROTEIN
# =============================================================================


    
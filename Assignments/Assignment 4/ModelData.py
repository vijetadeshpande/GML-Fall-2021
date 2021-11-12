#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  3 14:53:56 2021

@author: vijetadeshpande
"""
import pandas as pd
import torch
import os
import train_utils as t_utils


class ModelData():
    
    def __init__(self, data_dir:str, 
                 batch_size:int = 32,
                 data_train = None,
                 data_val = None,
                 data_test = None,
                 bert_feature_dim = 768):
        
        # step by step data preprocessing
        # 1. Read the training and testing data
        # 2. Convert data into required shape and convert data into FloatTensor
        # 3. Create tuples of source and target, i.e. (source, target)
        # 4. Create batches of examples
        
        # STEP - 1: read data
        try:
            data_train = pd.read_csv(data_train, sep = ' ', header = None)
            data_test = pd.read_csv(data_test, sep = ' ', header = None)
            data_val = pd.read_csv(data_val, sep = ' ', header = None)
        except:
            try:
                data_train = pd.read_csv(os.path.join(data_dir, 'train.txt'), sep = ' ', header = None)
                data_test = pd.read_csv(os.path.join(data_dir, 'test.txt'), sep = ' ', header = None)
                data_val = pd.read_csv(os.path.join(data_dir, 'val.txt'), sep = ' ', header = None)
            except:
                print('\nNO DATA FILE FOUND')
                return
    
        #
        assert type(data_train) == pd.DataFrame
        assert type(data_val) == pd.DataFrame
        assert type(data_test) == pd.DataFrame
        
        
        # STEP - 2: create numpy array of required shape and convert them into torch tensors
        X_train, Y_train = torch.FloatTensor(data_train.iloc[:, 0].values), torch.FloatTensor(data_train.iloc[:, -1].values)
        X_test, Y_test = torch.FloatTensor(data_test.iloc[:, 0].values), torch.FloatTensor(data_test.iloc[:, -1].values)
        X_val, Y_val = torch.FloatTensor(data_val.iloc[:, 0].values), torch.FloatTensor(data_val.iloc[:, -1].values)
        
        # 
        INPUT_DIM, OUTPUT_DIM = bert_feature_dim, 26#len(set(data_train.loc[:, 1].unique().values.tolist()).union(set(data_val.loc[:, 1].unique().values.tolist())))
        
        # STEP - 3: create tuples
        train, test, val = [], [], []
        for example in range(0, X_train.shape[0], batch_size):
            X, Y = X_train[example:example + batch_size], Y_train[example:example + batch_size]
            X, Y = X, Y#.unsqueeze(2)
            train.append((X, Y))
        for example in range(0, X_test.shape[0], batch_size):
            X, Y = X_test[example:example + batch_size], Y_test[example:example + batch_size]
            X, Y = X, Y#.unsqueeze(2)
            test.append((X, Y))
        for example in range(0, X_val.shape[0], batch_size):
            X, Y = X_val[example:example + batch_size], Y_val[example:example + batch_size]
            X, Y = X, Y#.unsqueeze(2)
            val.append((X, Y))
        
        # set attributes for the ModelData class
        self.train, self.val, self.test = train, val, test
        #self.SRC, self.TRG = SRC, TRG
        
        # dimension attributes
        self.input_features, self.output_features = INPUT_DIM, OUTPUT_DIM
        self.batch_size = batch_size
    
        return
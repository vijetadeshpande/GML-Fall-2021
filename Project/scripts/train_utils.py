#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  3 10:07:13 2021

@author: vijetadeshpande
"""

import torch
import torch.nn as nn
import math
import numpy as np
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import pandas as pd
import pickle
import os
import seaborn as sns
import matplotlib.pyplot as plt
import json

def transform_data(df_, to_ = 64):
    
    #
    nodes, x = df_.iloc[:, 1], df_.iloc[:, 2:].values#, df_.loc[:, -1].values
    #x_red = TSNE(n_components = to_, learning_rate='auto', init='random').fit_transform(x)
    x_red = PCA(n_components = to_).fit_transform(x)
    
    #
    df_out = pd.DataFrame(0, index = np.arange(df_.shape[0]), columns = np.arange(to_+1))
    df_out.loc[:, 1:] = x_red
    df_out.loc[:, 0] = nodes
    
    return df_out

def init_parameters(model):
    
    for name, param in model.named_parameters():
        if param.requires_grad:
            nn.init.uniform_(param.data, -0.08, 0.08)
    
    return model

def print_model_structure(model):
    
    print('\n\nGraphSage model structure is as follows:')
    print(model.graph_network)
    
    #print('\n\nClassifier structure is as follows:')
    #print(model.classifier)
    
    return 

def count_params(model, model_name = 'GraphSage'):
    
    count_ = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f'\nThe {model_name} model has {count_:,} trainable parameters')
    
    return count_

def split_time(time_):
    
    mins_ = int(time_ / 60)
    secs_ = int((time_ - (mins_*60)))
    
    return mins_, secs_

def epoch_time(start_, end_):
    
    time_ = end_ - start_
    mins_, secs_ = split_time(time_)
    
    return mins_, secs_

def print_train_progress(epoch, epoch_n, loss_train, acc_train, loss_val, acc_val, start_time, end_time):
    
    epoch_mins, epoch_secs = epoch_time(start_time, end_time)
    
    if True:#epoch == (epoch_n - 1):
        try:
            
            print(f'\nEpoch: {epoch+1:02} | Time: {epoch_mins}m {epoch_secs}s')
            print(f'Train loss: {loss_train:.4f} | Train accuracy: {acc_train:.4f}')
            print(f'Val. loss: {loss_val:.4f} | Val. accuracy: {acc_val:.4f}')
            
        except:
            print('Error in print the progress. Some error might be too large.')        
    
    return

def print_test_report(loss_test, total_time):
    
    epoch_mins, epoch_secs = split_time(total_time)
    
    #
    try:
        
        print(f'\nTime taken for training: {epoch_mins}m {epoch_secs}s')
        print(f'Test loss: {loss_test:.4f} | Train PPL: {math.exp(loss_test):7.4f}')
        #print(f'Val. loss: {loss_val:.4f} | val. PPL: {math.exp(loss_val):7.4f}')
        
    except:
        print('Error in print the progress. Some error might be too large.')        
    
    return 

def node_to_title(ref_: pd.DataFrame, nodes_: list):
    
    titles_= ref_.loc[nodes_, 1].values.tolist()
    
    return titles_

def class_to_name(ref_: pd.DataFrame, classes_: list):
    
    class_names_ = ref_.loc[classes_, 1].values.tolist()
    
    return class_names_

def load_model(model, model_filename):
    
    try:
        model.load_state_dict(torch.load(model_filename))
    except:
        print('\nFailed to load the specified model. Proceeding with randomly initiaized model')
    
    return model

def save_model(model_state_dict, model_filename):
    
    torch.save(model_state_dict, model_filename)
    
    return 

def update_model(valid_loss, best_valid_loss, model, model_state_dict):
    
    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        model_state_dict = model.state_dict()
        #torch.save(model.state_dict(), filename_)
    
    return model_state_dict, best_valid_loss

def load_pickle(filename_):
    
    with open(filename_, 'rb') as fp:
        data_ = pickle.load(fp)
        
    return data_

def load_results(path_):
    

    results_top_3 = load_pickle(os.path.join(path_, 'top_3_results.p'))
    results_best = load_pickle(os.path.join(path_, 'best_result.p'))

    
    return results_top_3, results_best

def save_tuning_results(results, path_, 
                        save_all_:bool = False,
                        save_top3_:bool = False):
    
    # from the set of all results extract top 3 results
    results_top = {}
    results_best = {}
    all_val = []
    for hp_set in results:
        all_val.append(results[hp_set]['results']['best validation loss'])
    
    best_val = sorted(all_val)
    best_val = best_val[:3] if save_top3_ else [best_val[0]]
    for hp_set in results:
        if results[hp_set]['results']['best validation loss'] in best_val:
            # collect everything in a dict
            results_top[hp_set] = results[hp_set]
            
            # save best model
            if results_best == {}:
                results_best[hp_set] = results[hp_set]
            
    if False:
        # save all top results
        if save_all_:
            filename_hp = os.path.join(path_, 'results.p')
            with open(filename_hp, 'wb') as fp:
                pickle.dump(results, fp, protocol=pickle.HIGHEST_PROTOCOL)
        
        # save all top results
        if save_top3_:
            filename_hp = os.path.join(path_, 'top_3_results.p')
            with open(filename_hp, 'wb') as fp:
                pickle.dump(results_top, fp, protocol=pickle.HIGHEST_PROTOCOL)
        
        # save best model
        filename_hp = os.path.join(path_, 'best_result.p')
        with open(filename_hp, 'wb') as fp:
            pickle.dump(results_best, fp, protocol=pickle.HIGHEST_PROTOCOL)
    
    for hp_set in results_best:
        results_out = results_best[hp_set]
    
    # save node embeddings
    try:
        if isinstance(results_out['results']['node embeddings'], (list)):
            pd.DataFrame(results_out['results']['node embeddings']).to_csv(os.path.join(path_, 'ProteinNodeEmbedding.csv'))
        else:
            pd.DataFrame(results_out['results']['node embeddings'].numpy().tolist()).to_csv(os.path.join(path_, 'ProteinNodeEmbedding.csv'))
    except:
        pass
    """
    filename = os.path.join(path_, 'ProteinNodeEmbeddings_.json')
    with open(filename, 'w') as f:
        json.dump(results_out['results']['node embeddings'].numpy().tolist(), f, indent = 4)
    """
    
    return results_out

def write_predictions(results, filename_):
    
    try:
        os.remove(filename_)
    except:
        pass
    
    results['results']['inference'].loc[:, ['ni', 'nj', 'link']].to_csv(filename_, header=None, index=None, sep=' ', mode='a')
    
    return

def plot_train_val(best_result: dict, save_path):
    
    #
    losses_trn = best_result['results']['train losses']
    losses_val = best_result['results']['validation losses']
    losses = losses_trn + losses_val
    
    #
    accuracies_trn = best_result['results']['train accuracies']
    accuracies_val = best_result['results']['validation accuracies']
    accuracies = accuracies_trn + accuracies_val
    
    #
    df_ = pd.DataFrame(0, index = np.arange(len(losses)), columns = ['Epoch', 'Data', 'Cross entropy loss', 'Accuracy'])
    df_.loc[:, 'Epoch'] = np.arange(len(losses)/2).tolist() + np.arange(len(losses)/2).tolist()
    df_.loc[:, 'Data'] = ['Train'] * int(len(losses)/2) + ['Validation'] * int(len(losses)/2)
    df_.loc[:, 'Cross entropy loss'] = losses
    df_.loc[:, 'Accuracy'] = accuracies
    
    # create footnote with all hyperparameter values
    footnote = 'Hyperparameter values:'
    for hpar in best_result['hyperparameters']:
        if not hpar in ['device', 'learning rate reduction', 'learning rate reduction step', 'model filename', 'number of classes']:
            val_ = best_result['hyperparameters'][hpar]
            footnote += f'\n{hpar}: {val_}'
    
    # plot
    filename_ = os.path.join(save_path, 'CrossEntropyLoss.png')
    plt.figure()
    plot_loss = sns.lineplot(data = df_,
                             x = 'Epoch',
                             y = 'Cross entropy loss',
                             hue = 'Data')
    plt.gcf().text(1, 0.25, footnote, fontsize=14)
    #plt.figtext(0.5, 0.01, footnote, ha="center", fontsize=18)#, bbox={"facecolor":"orange", "alpha":0.5, "pad":5})
    plt.savefig(filename_, bbox_inches="tight")
    
    #
    filename_ = os.path.join(save_path, 'Accuracies.png')
    plt.figure()
    plot_acc = sns.lineplot(data = df_,
                            x = 'Epoch',
                            y = 'Accuracy',
                            hue = 'Data')
    plt.gcf().text(1, 0.25, footnote, fontsize=14)
    plt.savefig(filename_, bbox_inches="tight")
    
    
    return

def embedding_visualization(embeddings: dict, 
                            train:str,
                            val:str,
                            save_path:str):
    
    
    #
    try:
        data_train = pd.read_csv(train, sep = ' ', header = None)
        data_val = pd.read_csv(val, sep = ' ', header = None)
    except:
        try:
            data_train = pd.read_csv(os.path.join(save_path, '..', 'train.txt'), sep = ' ', header = None)
            data_val = pd.read_csv(os.path.join(save_path, '..', 'val.txt'), sep = ' ', header = None)
        except:
            print('\nNO DATA FILE FOUND FOR embedding_visualization()')
            return
    
    # collect all the labeled examples
    labeled = pd.concat([data_train, data_val])
    labeled = pd.DataFrame(labeled.values, index = labeled.iloc[:, 0].values, columns = ['node', 'true_label'])
    similar_labels = {'stock_market_related': [0, 2, 10, 19, 25], 
                      'emotions': [1], 
                      'disease_related': [3, 4, 5, 6, 7, 11, 12, 15, 17, 21, 22, 23, 24], 
                      'marketing_related': [8, 13, 18, 20],  
                      'security_related': [9, 14, 16]}
    for row_ in labeled.index:
        y_ = labeled.loc[row_, 'true_label']
        for cat_ in similar_labels:
            if y_ in similar_labels[cat_]:
                labeled.loc[row_, 'category'] = cat_
                break
    labeled_x = labeled.iloc[:, 0].values.tolist()
    
    #
    for layer in embeddings:
        
        #
        layer_name = 'GNN_' + layer
        
        # slice out vectors of the labeled data and transform the data to 2-d
        vec_ = pd.DataFrame(PCA(n_components = 2).fit_transform(embeddings[layer][labeled_x, :]), index = labeled_x, columns = ['x1', 'x2'])
        vec_['Category'] = labeled.loc[labeled_x, 'category']
        
        #
        filename_ = os.path.join(save_path, 'PCA for %s embeddings.png'%(layer_name))
        plt.figure()
        plot_ = sns.scatterplot(data = vec_,
                                x = 'x1',
                                y = 'x2',
                                hue = 'Category')
        plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
        plt.title(layer_name)
        plt.savefig(filename_, bbox_inches="tight")
        
    
    return 



#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  4 14:18:29 2021

@author: vijetadeshpande
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd

class EvalMetrics():
    
    def __init__(self):
        
        
        return
    
    
    def __accuracy__(self, pred, trg):
        
        acc = sum(pred == trg)/(len(pred))
        
        # check
        other_met = {}
        other_met['false negative'] = sum(pred[pred == 0] != trg[pred == 0])/(len(pred))
        other_met['true negative'] = sum(pred[pred == 0] == trg[pred == 0])/(len(pred))
        other_met['false positive'] = sum(pred[pred == 1] != trg[pred == 1])/(len(pred))
        other_met['true positive'] = sum(pred[pred == 1] == trg[pred == 1])/(len(pred))
        
        #true_labels_for_correct_pred = trg[pred == trg]
        #true_labels_for_incorect_pred = trg[pred != trg]
        #percent_link_correct_pred = sum(true_labels_for_correct_pred)/len(true_labels_for_correct_pred)
        #if len(true_labels_for_incorect_pred) > 0:
        #    percent_link_incorrect_pred = sum(true_labels_for_incorect_pred)/len(true_labels_for_incorect_pred)
        #else:
        #    percent_link_incorrect_pred = 0
            

        return acc, other_met
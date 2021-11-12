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
        
        return acc
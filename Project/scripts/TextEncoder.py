#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 20 12:22:17 2021

@author: vijetadeshpande
"""

import torch
import torch.nn as nn
from transformers import BertConfig
from transformers import BertModel
from transformers import BertTokenizer
from transformers import AutoModel
import medspacy

class EncBERT(nn.Module):
    
    def __init__(self, model_name = 'allenai/scibert_scivocab_uncased', device = None):
        super(EncBERT, self).__init__()
        
        # models to try
        # allenai/scibert_scivocab_uncased
        # anindabitm/sagemaker-BioclinicalBERT-ADR
        # jsylee/scibert_scivocab_uncased-finetuned-ner
        # abhibisht89/spanbert-large-cased-finetuned-ade_corpus_v2
        
        # model blocks
        model_config = BertConfig.from_pretrained(model_name, output_hidden_states = True)
        self.enc = BertModel.from_pretrained(model_name, config = model_config)
        self.tokenizer = BertTokenizer.from_pretrained(model_name)

        
        #
        if device:
            self.device = device
        else:
            
            self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        
        return
    
    def forward(self, seq_in):
        
        # tokenize
        seq_in = self.tokenizer.encode_plus(seq_in, 
                                            add_special_tokens = True,
                                            max_length = 512,
                                            return_token_type_ids = True,
                                            pad_to_max_length = False,
                                            return_attention_mask = True,
                                            return_tensors = 'pt')
        
        ip_ids, attn_mask = seq_in['input_ids'].flatten().unsqueeze(0).to(self.device), seq_in['attention_mask'].flatten().unsqueeze(0).to(self.device)
        
        # forward pass
        OUT = self.enc(input_ids = ip_ids,
                       attention_mask = attn_mask)#,
                       #output_hidden_states = True)
        hidden = OUT[0].to(self.device)#OUT['hidden_states'][-1].to(self.device)
        
        # average pooling
        hidden = torch.mean(hidden, dim = -2).detach().numpy()[0]#.tolist()
        
        
        return hidden

class EncMedSpacy(nn.Module):
    
    def __init__(self, model_name = 'en_ner_bionlp13cg_md', device = None):
        super(EncMedSpacy, self).__init__()
        
        #
        self.enc = medspacy.load(model_name)
        
        
        return
    
    
    def forward(self, seq_in):
        
        seq_in = self.enc(seq_in)
        embedding = seq_in.vector
        
        
        return embedding
    
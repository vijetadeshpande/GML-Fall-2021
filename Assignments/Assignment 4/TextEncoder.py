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

class EmbModel(nn.Module):
    
    def __init__(self, model_name, device = None):
        super().__init__()
        
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
        hidden = torch.mean(hidden, dim = -2).detach().numpy()[0].tolist()
        
        
        return hidden
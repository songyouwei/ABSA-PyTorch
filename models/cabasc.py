# -*- coding: utf-8 -*-
# file: cabasc.py
# author: albertopaz <aj.paz167@gmail.com>
# Copyright (C) 2018. All Rights Reserved.

from layers.attention import Attention
import torch
import torch.nn as nn
import torch.nn.functional as F

from layers.squeeze_embedding import SqueezeEmbedding
from layers.dynamic_rnn import DynamicLSTM         

class Cabasc(nn.Module):
    def __init__(self, embedding_matrix, opt, type = 'cabasc'):
        super(Cabasc, self).__init__()
        self.opt = opt
        self.embed = nn.Embedding.from_pretrained(torch.tensor(embedding_matrix, dtype=torch.float))
        self.squeeze_embedding = SqueezeEmbedding(batch_first=True)
        self.attention = Attention(opt.embed_dim, score_function='mlp')              # content attention
        self.m_linear = nn.Linear(opt.embed_dim, opt.embed_dim, bias = False)        
        self.mlp = nn.Linear(opt.embed_dim, opt.embed_dim)                           # W4
        self.dense = nn.Linear(opt.embed_dim, opt.polarities_dim)                    # W5
        # context attention layer
        self.rnn_l = DynamicLSTM(opt.embed_dim, opt.hidden_dim, num_layers=1, batch_first=True, rnn_type = 'GRU') 
        self.rnn_r = DynamicLSTM(opt.embed_dim, opt.hidden_dim, num_layers=1, batch_first=True, rnn_type = 'GRU')
        self.mlp_l = nn.Linear(opt.hidden_dim, 1)
        self.mlp_r = nn.Linear(opt.hidden_dim, 1)
    
    def context_attention(self, x_l, x_r, memory, memory_len, aspect_len):
        
        # Context representation
        left_len, right_len = torch.sum(x_l != 0, dim=-1), torch.sum(x_r != 0, dim=-1)
        x_l, x_r = self.embed(x_l), self.embed(x_r) 
        
        context_l, (_, _) =  self.rnn_l(x_l, left_len)        # left, right context : (batch size, max_len, embedds)
        context_r, (_, _) =  self.rnn_r(x_r, right_len)           
        
        # Attention weights : (batch_size, max_batch_len, 1) 
        attn_l = F.sigmoid(self.mlp_l(context_l)) + 0.5
        attn_r = F.sigmoid(self.mlp_r(context_r)) + 0.5
        
        # apply weights one sample at a time
        for i in range(memory.size(0)): 
            aspect_start = (left_len[i] - aspect_len[i]).item()
            aspect_end = left_len[i] 
            # attention weights for each element in the sentence
            for idx in range(memory_len[i]):
                if idx < aspect_start: memory[i][idx] *= attn_l[i][idx]             
                elif idx < aspect_end: memory[i][idx] *= attn_l[i][idx] + attn_r[i][idx - aspect_start]
                else: memory[i][idx] *= attn_r[i][idx - aspect_start]       
          
        return memory
    
    def locationed_memory(self, memory, memory_len, left_len, aspect_len):
        # based on the absolute distance to the first border word of the aspect
        for i in range(memory.size(0)):
            for idx in range(memory_len[i]):
                aspect_start = left_len[i] - aspect_len[i]
                aspect_end = left_len[i] 
                if idx < aspect_start: l = aspect_start.item() - idx                   
                elif idx <= aspect_end: l = 0
                else: l = idx - aspect_end.item()
                memory[i][idx] *= (1-float(l)/int(memory_len[i]))
               
        return memory
    
    def forward(self, inputs):
        
        # inputs
        text_raw_indices, aspect_indices, x_l, x_r = inputs[0], inputs[1], inputs[2], inputs[3]
        memory_len = torch.sum(text_raw_indices != 0, dim = -1)
        aspect_len = torch.sum(aspect_indices != 0, dim = -1)
        
        # aspect representation
        nonzeros_aspect = torch.tensor(aspect_len, dtype=torch.float).to(self.opt.device)
        aspect = self.embed(aspect_indices)
        aspect = torch.sum(aspect, dim=1)
        aspect = torch.div(aspect, nonzeros_aspect.view(nonzeros_aspect.size(0), 1))
        x = aspect.unsqueeze(dim=1)
        
        # memory module
        memory = self.embed(text_raw_indices)
        memory = self.squeeze_embedding(memory, memory_len)
        
        # sentence representation 
        nonzeros_memory = torch.tensor(memory_len, dtype=torch.float).to(self.opt.device)
        v_s = torch.sum(memory, dim = 1)
        v_s = torch.div(v_s, nonzeros_memory.view(nonzeros_memory.size(0),1))  
        v_s = v_s.unsqueeze(dim=1)
        
        # position attention module
        if type == 'c': memory = self.locationed_memory(memory, memory_len, left_len, aspect_len)
        elif type == 'cabasc': 
            # context attention
            memory = self.context_attention(x_l, x_r, memory, memory_len, aspect_len)
            # recalculate sentence rep with new memory
            v_s = torch.sum(memory, dim = 1)                                             
            v_s = torch.div(v_s, nonzeros_memory.view(nonzeros_memory.size(0),1))  
            v_s = v_s.unsqueeze(dim=1)
        
        # content attention module
        for _ in range(self.opt.hops):  
            #x = self.x_linear(x)
            v_ts = self.attention(memory, x)                 
       
        # classifier
        v_ns = v_ts + v_s                                 # embedd the sentence
        v_ns = v_ns.view(v_ns.size(0), -1)
        v_ms = F.tanh(self.mlp(v_ns))
        out = self.dense(v_ms)
        out = F.softmax(out, dim=-1)   
        
        return out

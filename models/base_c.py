# -*- coding: utf-8 -*-
# file: base_c.py
# author: albertopaz <aj.paz167@gmail.com>
# Copyright (C) 2018. All Rights Reserved.

from layers.attention import Attention
import torch
import torch.nn as nn
import torch.nn.functional as F

from layers.squeeze_embedding import SqueezeEmbedding

class BaseC(nn.Module):
    '''
    Position attention based memory module
    '''
    def __init__(self, embedding_matrix, opt):
        super(BaseC, self).__init__()
        self.opt = opt
        self.embed = nn.Embedding.from_pretrained(torch.tensor(embedding_matrix, dtype=torch.float))
        self.squeeze_embedding = SqueezeEmbedding(batch_first=True)
        self.attention = Attention(opt.embed_dim, score_function='mlp')
        self.m_linear = nn.Linear(opt.embed_dim, opt.embed_dim, bias = False)        
        self.x_linear = nn.Linear(opt.embed_dim, opt.embed_dim, bias = False)        
        self.s_linear = nn.Linear(opt.embed_dim, opt.embed_dim, bias = False)
        self.mlp = nn.Linear(opt.embed_dim, opt.embed_dim)                           # W4
        self.dense = nn.Linear(opt.embed_dim, opt.polarities_dim)                    # W5
    
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
        text_raw_indices, aspect_indices, left_with_aspect_indices = inputs[0], inputs[1], inputs[2]
        left_len = torch.sum(left_with_aspect_indices != 0, dim = -1)
        memory_len = torch.sum(text_raw_indices != 0, dim = -1)
        aspect_len = torch.sum(aspect_indices != 0, dim = -1)
        
        # aspect representation
        nonzeros_aspect = torch.tensor(aspect_len, dtype=torch.float).to(self.opt.device)
        aspect = self.embed(aspect_indices)
        aspect = torch.sum(aspect, dim=1)
        aspect = torch.div(aspect, nonzeros_aspect.view(nonzeros_aspect.size(0), 1))
        x = aspect.unsqueeze(dim=1)
        
        # sentence representation 
        nonzeros_memory = torch.tensor(memory_len, dtype=torch.float).to(self.opt.device)
        memory = self.embed(text_raw_indices)
        v_s = torch.sum(memory, dim = 1)
        v_s = torch.div(v_s, nonzeros_memory.view(nonzeros_memory.size(0),1))  
        v_s = v_s.unsqueeze(dim=1)
        
        # memory module
        memory = self.squeeze_embedding(memory, memory_len)
        memory = self.locationed_memory(memory, memory_len, left_len, aspect_len)

        # content attention module
        for _ in range(self.opt.hops):  
            x = self.x_linear(x)
            s = self.s_linear(v_s)
            v_ts = self.attention(memory, x)              #### TO DO : IMPLEMENT THE RIGHT ATTENTION MECHANISM FwNN3
       
        # classifier
        v_ns = v_ts + v_s                                 # embedd the sentence 
        v_ns = v_ns.view(v_ns.size(0), -1)
        v_ms = F.tanh(self.mlp(v_ns))
        out = self.dense(v_ms)
        out = F.softmax(out, dim=-1)   
        
        return out

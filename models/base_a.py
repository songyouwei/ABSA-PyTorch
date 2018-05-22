# -*- coding: utf-8 -*-
# file: base_a.py
# author: albertopaz <aj.paz167@gmail.com>
# Copyright (C) 2018. All Rights Reserved.

from layers.attention import Attention
import torch
import torch.nn as nn
import torch.nn.functional as F

from layers.squeeze_embedding import SqueezeEmbedding

class BaseA(nn.Module):
    def __init__(self, embedding_matrix, opt):
        super(BaseA, self).__init__()
        self.opt = opt
        self.embed = nn.Embedding.from_pretrained(torch.tensor(embedding_matrix, dtype=torch.float))
        self.squeeze_embedding = SqueezeEmbedding(batch_first=True)
        self.attention = Attention(opt.embed_dim, score_function='mlp')
        self.m_linear = nn.Linear(opt.embed_dim, opt.embed_dim)                                 # W2
        self.x_linear = nn.Linear(opt.embed_dim, opt.embed_dim)                                 # W3
        self.mlp = nn.Linear(opt.embed_dim, opt.embed_dim)                                      # W4
        self.dense = nn.Linear(opt.embed_dim, opt.polarities_dim)                               # W5

    def forward(self, inputs):
        
        # inputs
        text_raw_indices, aspect_indices = inputs[0], inputs[1]
        memory_len = torch.sum(text_raw_indices != 0, dim=-1)
        aspect_len = torch.sum(aspect_indices != 0, dim=-1)
        nonzeros_aspect = torch.tensor(aspect_len, dtype=torch.float).to(self.opt.device)
        
        # aspect representation
        aspect = self.embed(aspect_indices)
        aspect = torch.sum(aspect, dim=1)
        aspect = torch.div(aspect, nonzeros_aspect.view(nonzeros_aspect.size(0), 1))
        x = aspect.unsqueeze(dim=1)
        
        # memory module
        memory = self.embed(text_raw_indices)
        memory = self.squeeze_embedding(memory, memory_len)
        
        # content attention module
        for _ in range(self.opt.hops): 
            x = self.x_linear(x)
            #memory = self.m_linear(memory)
            v_ns = self.attention(memory, x)              # aspect-specific sentence representation
        
        # classifier
        v_ns = v_ns.view(v_ns.size(0), -1)
        v_ms = F.tanh(self.mlp(v_ns))
        out = self.dense(v_ms)
        out = F.softmax(out, dim=-1)   
        return out

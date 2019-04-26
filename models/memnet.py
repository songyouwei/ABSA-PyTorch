# -*- coding: utf-8 -*-
# file: memnet.py
# author: songyouwei <youwei0314@gmail.com>
# Copyright (C) 2018. All Rights Reserved.

from layers.attention import Attention
import torch
import torch.nn as nn

from layers.squeeze_embedding import SqueezeEmbedding


class MemNet(nn.Module):
    
    def locationed_memory(self, memory, memory_len):
        # here we just simply calculate the location vector in Model2's manner
        batch_size = memory.shape[0]
        seq_len = memory.shape[1]
        memory_len = memory_len.cpu().numpy()
        weight = [[] for i in range(batch_size)]
        for i in range(batch_size):
            for idx in range(memory_len[i]):
                weight[i].append(1-float(idx+1)/memory_len[i])
            for idx in range(memory_len[i], seq_len):
                weight[i].append(1)
        weight = torch.tensor(weight).to(self.opt.device)
        memory = weight.unsqueeze(2)*memory
        return memory

    def __init__(self, embedding_matrix, opt):
        super(MemNet, self).__init__()
        self.opt = opt
        self.embed = nn.Embedding.from_pretrained(torch.tensor(embedding_matrix, dtype=torch.float))
        self.squeeze_embedding = SqueezeEmbedding(batch_first=True)
        self.attention = Attention(opt.embed_dim, score_function='mlp')
        self.x_linear = nn.Linear(opt.embed_dim, opt.embed_dim)
        self.dense = nn.Linear(opt.embed_dim, opt.polarities_dim)

    def forward(self, inputs):
        text_raw_without_aspect_indices, aspect_indices = inputs[0], inputs[1]
        memory_len = torch.sum(text_raw_without_aspect_indices != 0, dim=-1)
        aspect_len = torch.sum(aspect_indices != 0, dim=-1)
        nonzeros_aspect = torch.tensor(aspect_len, dtype=torch.float).to(self.opt.device)

        memory = self.embed(text_raw_without_aspect_indices)
        memory = self.squeeze_embedding(memory, memory_len)
        # memory = self.locationed_memory(memory, memory_len)
        aspect = self.embed(aspect_indices)
        aspect = torch.sum(aspect, dim=1)
        aspect = torch.div(aspect, nonzeros_aspect.view(nonzeros_aspect.size(0), 1))
        x = aspect.unsqueeze(dim=1)
        for _ in range(self.opt.hops):
            x = self.x_linear(x)
            out_at, _ = self.attention(memory, x)
            x = out_at + x
        x = x.view(x.size(0), -1)
        out = self.dense(x)
        return out

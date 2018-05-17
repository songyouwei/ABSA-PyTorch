# -*- coding: utf-8 -*-
# file: memnet.py
# author: songyouwei <youwei0314@gmail.com>
# Copyright (C) 2018. All Rights Reserved.

from layers.attention import Attention
import torch
import torch.nn as nn


class MemNet(nn.Module):
    def locationed_memory(self, memory, text_raw_without_aspect_indices):
        # here we just simply calculate the location vector in Model2's manner
        lens_memory = torch.tensor(torch.sum(text_raw_without_aspect_indices != 0, dim=-1), dtype=torch.int).to(self.opt.device)
        for i in range(memory.size(0)):
            start = self.opt.max_seq_len-int(lens_memory[i])
            for j in range(lens_memory[i]):
                idx = start+j
                memory[i][idx] *= (1-float(j)/int(lens_memory[i]))
        return memory

    def __init__(self, embedding_matrix, opt):
        super(MemNet, self).__init__()
        self.opt = opt
        self.embed = nn.Embedding.from_pretrained(torch.tensor(embedding_matrix, dtype=torch.float))
        self.attention = Attention(opt.embed_dim, score_function='mlp')
        self.x_linear = nn.Linear(opt.embed_dim, opt.embed_dim)
        self.dense = nn.Linear(opt.embed_dim, opt.polarities_dim)

    def forward(self, inputs):
        text_raw_without_aspect_indices, aspect_indices = inputs[0], inputs[1]
        nonzeros_aspect = torch.tensor(torch.sum(aspect_indices != 0, dim=-1), dtype=torch.float).to(self.opt.device)
        memory = self.embed(text_raw_without_aspect_indices)
        memory = self.locationed_memory(memory, text_raw_without_aspect_indices)
        aspect = self.embed(aspect_indices)
        aspect = torch.sum(aspect, dim=1)
        aspect = torch.div(aspect, nonzeros_aspect.view(nonzeros_aspect.size(0), 1))
        x = aspect.unsqueeze(dim=1)
        for _ in range(self.opt.hops):
            x = self.x_linear(x)
            out_at = self.attention(memory, x)
            x = out_at + x
        x = x.view(x.size(0), -1)
        out = self.dense(x)
        return out

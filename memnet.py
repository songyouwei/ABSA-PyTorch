# -*- coding: utf-8 -*-
# file: memnet.py
# author: songyouwei <youwei0314@gmail.com>
# Copyright (C) 2018. All Rights Reserved.

from helpers import instructor
from attention import Attention
import torch
import torch.nn as nn

# Hyper Parameters
hops = 7
embed_dim = 100
max_seq_len = 80
polarities_dim = 3
num_epochs = 50
batch_size = 128
learning_rate = 0.001

model_name = 'memnet'
dataset = 'twitter'  # twitter / restaurant / laptop
inputs_cols = ['text_raw_without_aspect_indices', 'aspect_indices']
log_step = 10


class MemNet(nn.Module):
    # @staticmethod
    # def locationed_memory(memory):
    #     # here we just simply calculate the location vector in Model2's manner
    #     n = memory.shape[1]
    #     v = torch.ones(n)
    #     for i in range(n):
    #         v[i] -= i / int(n)
    #     v = v.unsqueeze(dim=-1)
    #     locationed_mem = torch.mul(memory, v)
    #     return locationed_mem

    def __init__(self, embedding_matrix):
        super(MemNet, self).__init__()
        self.embed = nn.Embedding.from_pretrained(torch.tensor(embedding_matrix, dtype=torch.float))
        self.attention = Attention(embed_dim, max_seq_len, 1, score_function='mlp')
        self.x_linear = nn.Linear(embed_dim, embed_dim)
        self.dense = nn.Linear(embed_dim, polarities_dim)

    def forward(self, inputs):
        text_raw_without_aspect_indices, aspect_indices = inputs[0], inputs[1]
        nonzeros_aspect = torch.tensor(torch.sum(aspect_indices != 0, dim=-1), dtype=torch.float)
        memory = self.embed(text_raw_without_aspect_indices)
        # memory = MemNet.locationed_memory(memory)
        aspect = self.embed(aspect_indices)
        aspect = torch.sum(aspect, dim=1)
        aspect = torch.div(aspect, nonzeros_aspect.view(nonzeros_aspect.size(0), 1))
        x = aspect.unsqueeze(dim=1)
        for _ in range(hops):
            x = self.x_linear(x)
            out_at = self.attention((memory, x))
            x = out_at + x
        x = x.view(x.size(0), -1)
        out = self.dense(x)
        return out


if __name__ == '__main__':
    ins = instructor(module_class=MemNet, model_name=model_name,
                     dataset=dataset, embed_dim=embed_dim, max_seq_len=max_seq_len,
                     batch_size=batch_size)
    ins.run(inputs_cols=inputs_cols,
            learning_rate=learning_rate,
            num_epochs=num_epochs,
            log_step=log_step)

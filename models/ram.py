# -*- coding: utf-8 -*-
# file: ram.py
# author: songyouwei <youwei0314@gmail.com>
# Copyright (C) 2018. All Rights Reserved.

from layers.dynamic_rnn import DynamicLSTM
from layers.attention import Attention
import torch
import torch.nn as nn


class RAM(nn.Module):
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
        super(RAM, self).__init__()
        self.opt = opt
        self.embed = nn.Embedding.from_pretrained(torch.tensor(embedding_matrix, dtype=torch.float))
        self.bi_lstm_context = DynamicLSTM(opt.embed_dim, opt.hidden_dim, num_layers=1, batch_first=True, bidirectional=True)
        self.bi_lstm_aspect = DynamicLSTM(opt.embed_dim, opt.hidden_dim, num_layers=1, batch_first=True, bidirectional=True)
        self.attention = Attention(opt.hidden_dim*2, score_function='mlp')
        self.gru_cell = nn.GRUCell(opt.hidden_dim*2, opt.hidden_dim*2)
        self.dense = nn.Linear(opt.hidden_dim*2, opt.polarities_dim)

    def forward(self, inputs):
        text_raw_without_aspect_indices, aspect_indices = inputs[0], inputs[1]
        context_len = torch.sum(text_raw_without_aspect_indices != 0, dim=-1)
        aspect_len = torch.sum(aspect_indices != 0, dim=-1)
        nonzeros_aspect = torch.tensor(aspect_len, dtype=torch.float).to(self.opt.device)

        memory = self.embed(text_raw_without_aspect_indices)
        memory, (_, _) = self.bi_lstm_context(memory, context_len)
        memory = self.locationed_memory(memory, text_raw_without_aspect_indices)
        aspect = self.embed(aspect_indices)
        aspect, (_, _) = self.bi_lstm_aspect(aspect, aspect_len)
        aspect = torch.sum(aspect, dim=1)
        aspect = torch.div(aspect, nonzeros_aspect.view(nonzeros_aspect.size(0), 1))

        et = aspect
        for _ in range(self.opt.hops):
            it_al = self.attention(memory, et).squeeze(dim=1)
            et = self.gru_cell(it_al, et)
        out = self.dense(et)
        return out

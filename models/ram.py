# -*- coding: utf-8 -*-
# file: ram.py
# author: songyouwei <youwei0314@gmail.com>
# Copyright (C) 2018. All Rights Reserved.

from layers.dynamic_rnn import DynamicLSTM
import torch
import torch.nn as nn
import torch.nn.functional as F

class RAM(nn.Module):
    def locationed_memory(self, memory, memory_len, left_len, aspect_len):
        u = torch.zeros(memory.size(0), memory.size(1), 1).to(self.opt.device)
        for i in range(memory.size(0)):
            for idx in range(memory_len[i]):
                aspect_start = left_len[i]
                if idx < aspect_start:
                    l = aspect_start - idx                   # l = absolute distance to the aspect
                    u[i][idx][0] = idx - aspect_start
                elif idx < aspect_start + aspect_len[i]:
                    l = 0
                else: 
                    l = idx - aspect_start - aspect_len[i] + 1
                    u[i][idx][0] = idx - aspect_start - aspect_len[i] + 1
                memory[i][idx] *= (1-float(l)/int(memory_len[i]))
        memory = torch.cat([memory, u], dim=2)       
        return memory

    def __init__(self, embedding_matrix, opt):
        super(RAM, self).__init__()
        self.opt = opt
        self.embed = nn.Embedding.from_pretrained(torch.tensor(embedding_matrix, dtype=torch.float))
        self.bi_lstm_context = DynamicLSTM(opt.embed_dim, opt.hidden_dim, num_layers=1, batch_first=True, bidirectional=True)
        self.att_linear = nn.Linear(opt.hidden_dim*2 + 1 + opt.embed_dim*2, 1)
        self.gru_cell = nn.GRUCell(opt.hidden_dim*2 + 1, opt.embed_dim)
        self.dense = nn.Linear(opt.embed_dim, opt.polarities_dim)

    def forward(self, inputs):
        text_raw_indices, aspect_indices, text_left_indices = inputs[0], inputs[1], inputs[2]
        left_len = torch.sum(text_left_indices != 0, dim=-1)
        memory_len = torch.sum(text_raw_indices != 0, dim=-1)
        aspect_len = torch.sum(aspect_indices != 0, dim=-1)
        nonzeros_aspect = aspect_len.float()

        memory = self.embed(text_raw_indices)
        memory, (_, _) = self.bi_lstm_context(memory, memory_len)
        memory = self.locationed_memory(memory, memory_len, left_len, aspect_len)
        
        aspect = self.embed(aspect_indices)
        aspect = torch.sum(aspect, dim=1)
        aspect = torch.div(aspect, nonzeros_aspect.unsqueeze(-1))
        et = torch.zeros_like(aspect).to(self.opt.device)

        batch_size = memory.size(0)
        seq_len = memory.size(1)
        for _ in range(self.opt.hops):
            g = self.att_linear(torch.cat([memory, 
                torch.zeros(batch_size, seq_len, self.opt.embed_dim).to(self.opt.device) + et.unsqueeze(1), 
                torch.zeros(batch_size, seq_len, self.opt.embed_dim).to(self.opt.device) + aspect.unsqueeze(1)], 
                dim=-1))
            alpha = F.softmax(g, dim=1)
            i = torch.bmm(alpha.transpose(1, 2), memory).squeeze(1)  
            et = self.gru_cell(i, et)
        out = self.dense(et)
        return out

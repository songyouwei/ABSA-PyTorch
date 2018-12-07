# -*- coding: utf-8 -*-
# file: aoa.py
# author: gene_zc <gene_zhangchen@163.com>
# Copyright (C) 2018. All Rights Reserved.

from layers.dynamic_rnn import DynamicLSTM
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class AOA(nn.Module):
    def __init__(self, embedding_matrix, opt):
        super(AOA, self).__init__()
        self.opt = opt
        self.embed = nn.Embedding.from_pretrained(torch.tensor(embedding_matrix, dtype=torch.float))
        self.ctx_lstm = DynamicLSTM(opt.embed_dim, opt.hidden_dim, num_layers=1, batch_first=True, bidirectional=True)
        self.asp_lstm = DynamicLSTM(opt.embed_dim, opt.hidden_dim, num_layers=1, batch_first=True, bidirectional=True)
        self.dense = nn.Linear(2 * opt.hidden_dim, opt.polarities_dim)

    def forward(self, inputs):
        text_raw_indices = inputs[0] # batch_size x seq_len
        aspect_indices = inputs[1] # batch_size x seq_len
        ctx_len = torch.sum(text_raw_indices != 0, dim=1)
        asp_len = torch.sum(aspect_indices != 0, dim=1)
        ctx = self.embed(text_raw_indices) # batch_size x seq_len x embed_dim
        asp = self.embed(aspect_indices) # batch_size x seq_len x embed_dim
        ctx_out, (_, _) = self.ctx_lstm(ctx, ctx_len) #  batch_size x (ctx) seq_len x 2*hidden_dim
        asp_out, (_, _) = self.asp_lstm(asp, asp_len) # batch_size x (asp) seq_len x 2*hidden_dim
        interaction_mat = torch.matmul(ctx_out, torch.transpose(asp_out, 1, 2)) # batch_size x (ctx) seq_len x (asp) seq_len
        alpha = F.softmax(interaction_mat, dim=1) # col-wise, batch_size x (ctx) seq_len x (asp) seq_len
        beta = F.softmax(interaction_mat, dim=2) # row-wise, batch_size x (ctx) seq_len x (asp) seq_len
        beta_avg = beta.mean(dim=1, keepdim=True) # batch_size x 1 x (asp) seq_len
        gamma = torch.matmul(alpha, beta_avg.transpose(1, 2)) # batch_size x (ctx) seq_len x 1
        weighted_sum = torch.matmul(torch.transpose(ctx_out, 1, 2), gamma).squeeze(-1) # batch_size x 2*hidden_dim
        out = self.dense(weighted_sum) # batch_size x polarity_dim

        return out
# -*- coding: utf-8 -*-
# file: attention.py
# author: songyouwei <youwei0314@gmail.com>
# Copyright (C) 2018. All Rights Reserved.

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class Attention(nn.Module):
    def __init__(self, embed_dim, score_function='scaled_dot_product'):
        # score_function: scaled_dot_product / mlp (concat) / bi_linear (general dot)
        super(Attention, self).__init__()
        self.embed_dim = embed_dim
        self.score_function = score_function
        self.linear_k = nn.Linear(embed_dim, embed_dim)
        self.linear_q = nn.Linear(embed_dim, embed_dim)
        if score_function == 'mlp':
            self.weight = nn.Parameter(torch.Tensor(embed_dim*2, 1))
        elif self.score_function == 'bi_linear':
            self.weight = nn.Parameter(torch.Tensor(embed_dim, embed_dim))
        else:
            self.register_parameter('weight', None)
        self.reset_parameters()

    def reset_parameters(self):
        for weight in self.parameters():
            nn.init.xavier_uniform_(weight)

    def forward(self, k, q):
        if len(q.shape) == 2:  # q_len missing
            q = torch.unsqueeze(q, dim=1)
        if len(k.shape) == 2:  # k_len missing
            k = torch.unsqueeze(k, dim=1)
        # k: (?, k_len, embed_dim,)
        # q: (?, q_len, embed_dim,)
        # score: (?, q_len, k_len,)
        # output: (?, q_len, embed_dim,)
        k = self.linear_k(k)
        q = self.linear_q(q)
        if self.score_function == 'scaled_dot_product':
            kt = k.permute(0, 2, 1)
            qkt = torch.bmm(q, kt)
            score = torch.div(qkt, math.sqrt(self.embed_dim))
        elif self.score_function == 'mlp':
            kx = torch.unsqueeze(k, dim=1).expand(-1, q.shape[1], -1, -1)
            qx = torch.unsqueeze(q, dim=2).expand(-1, -1, k.shape[1], -1)
            kq = torch.cat((kx, qx), dim=-1)  # (?, q_len, k_len, embed_dim*2)
            score = F.tanh(torch.matmul(kq, self.weight).squeeze(dim=-1))
        elif self.score_function == 'bi_linear':
            qw = torch.matmul(q, self.weight)
            kt = k.permute(0, 2, 1)
            score = torch.bmm(qw, kt)
        else:
            raise RuntimeError('invalid score_function')
        score = F.softmax(score, dim=-1)
        output = torch.bmm(score, k)
        return output
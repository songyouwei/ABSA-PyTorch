# -*- coding: utf-8 -*-
# file: attention.py
# author: songyouwei <youwei0314@gmail.com>
# Copyright (C) 2018. All Rights Reserved.

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class Attention(nn.Module):
    def __init__(self, embed_dim, hidden_dim=None, n_head=1, score_function='scaled_dot_product'):
        # score_function: scaled_dot_product / mlp (concat) / bi_linear (general dot)
        super(Attention, self).__init__()
        if hidden_dim is None:
            hidden_dim = embed_dim
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.n_head = n_head
        self.score_function = score_function
        self.linear_k = nn.Linear(embed_dim, hidden_dim)
        self.linear_q = nn.Linear(embed_dim, hidden_dim)
        if score_function == 'mlp':
            self.weight = nn.Parameter(torch.Tensor(hidden_dim*2, 1))
        elif self.score_function == 'bi_linear':
            self.weight = nn.Parameter(torch.Tensor(hidden_dim, hidden_dim))
        else:
            self.register_parameter('weight', None)
        self.reset_parameters()

    def reset_parameters(self):
        for weight in self.parameters():
            if len(weight.shape) > 1:
                nn.init.xavier_uniform_(weight)

    def forward(self, k, q):
        if len(q.shape) == 2:  # q_len missing
            q = torch.unsqueeze(q, dim=1)
        if len(k.shape) == 2:  # k_len missing
            k = torch.unsqueeze(k, dim=1)
        # k: (?, k_len, embed_dim,)
        # q: (?, q_len, embed_dim,)
        # kx: (n_head*?, k_len, hidden_dim)
        # qx: (n_head*?, q_len, hidden_dim)
        # score: (n_head*?, q_len, k_len,)
        # output: (?, q_len, n_head*embed_dim,)
        kx = self.linear_k(k).repeat(self.n_head, 1, 1)
        qx = self.linear_q(q).repeat(self.n_head, 1, 1)
        if self.score_function == 'scaled_dot_product':
            kt = kx.permute(0, 2, 1)
            qkt = torch.bmm(qx, kt)
            score = torch.div(qkt, math.sqrt(self.embed_dim))
        elif self.score_function == 'mlp':
            kx = torch.unsqueeze(kx, dim=1).expand(-1, q.shape[1], -1, -1)
            qx = torch.unsqueeze(qx, dim=2).expand(-1, -1, k.shape[1], -1)
            kq = torch.cat((kx, qx), dim=-1)  # (n_head*?, q_len, k_len, hidden_dim*2)
            score = F.tanh(torch.matmul(kq, self.weight).squeeze(dim=-1))
        elif self.score_function == 'bi_linear':
            qw = torch.matmul(q, self.weight)
            kt = k.permute(0, 2, 1)
            score = torch.bmm(qw, kt)
        else:
            raise RuntimeError('invalid score_function')
        score = F.softmax(score, dim=-1)
        k = k.repeat(self.n_head, 1, 1)
        output = torch.bmm(score, k)
        output = output.view(-1, q.shape[1], self.n_head*self.embed_dim)
        return output
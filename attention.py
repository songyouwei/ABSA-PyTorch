# -*- coding: utf-8 -*-
# file: attention
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

    def forward(self, inputs):
        # output = softmax(score)
        k, q = inputs
        if len(q.shape) == 2:
            q = torch.unsqueeze(q, dim=1)
        # k: (?, K_LEN, EMBED_DIM,)
        # q: (?, Q_LEN, EMBED_DIM,)
        # score: (?, Q_LEN, K_LEN,)
        if self.score_function == 'scaled_dot_product':
            kt = k.permute(0, 2, 1)
            qkt = torch.bmm(q, kt)
            score = torch.div(qkt, math.sqrt(self.embed_dim))
        elif self.score_function == 'mlp':
            kx = torch.unsqueeze(k, dim=1).repeat(1, q.shape[1], 1, 1)
            qx = torch.unsqueeze(q, dim=2).repeat(1, 1, k.shape[1], 1)
            kq = torch.cat((kx, qx), dim=-1)  # (?, Q_LEN, K_LEN, EMBED_DIM*2)
            score = F.tanh(torch.matmul(kq, self.weight).squeeze(dim=-1))
        elif self.score_function == 'bi_linear':
            qw = torch.matmul(q, self.weight)
            kt = k.permute(0, 2, 1)
            score = torch.bmm(qw, kt)
        else:
            raise RuntimeError('invalid score_function')
        score = F.softmax(score, dim=-1)
        # output: (?, Q_LEN, EMBED_DIM,)
        output = torch.bmm(score, k)
        return output
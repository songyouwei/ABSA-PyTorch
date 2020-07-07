# -*- coding: utf-8 -*-
# file: cnn.py
# author: anthng <thienan99dt@gmail.com>
# Copyright (C) 2020. All Rights Reserved.

import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.squeeze_embedding import SqueezeEmbedding

"""
https://www.aclweb.org/anthology/D14-1181.pdf

## CNN-multichannel
| data           |      Acc      |   F1    |
|:--------------:|:-------------:|:-------:|
| twitter        | 0.6763        | 0.6592  |
| restaurant     | 0.7518        | 0.5730  |
| laptop         | 0.6536        | 0.5741  |
"""

class CNN(nn.Module):
    def __init__(self, embedding_matrix, opt):
        super(CNN, self).__init__()
        """
        CNN-multichannel
            - activation: relu (for binay classes), tanh (aspect sentiment)
            - filter: 100
            - windows: 3,4,5
            - dropout rate: 0.5
        """
        self.embed = nn.Embedding.from_pretrained(torch.tensor(embedding_matrix, dtype=torch.float), freeze = True)
        self.opt = opt

        self.squeeze_embedding = SqueezeEmbedding()

        filters = 100
        #kernel size
        kSize = [3,4,5]

        self.conv0 = nn.Conv1d(in_channels=1, out_channels=filters, kernel_size=(kSize[0], self.opt.embed_dim), bias=True)
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=filters, kernel_size=(kSize[1], self.opt.embed_dim), bias=True)
        self.conv2 = nn.Conv1d(in_channels=1, out_channels=filters, kernel_size=(kSize[2], self.opt.embed_dim), bias=True)


        feature = len(kSize) * filters

        self.dropout = nn.Dropout(self.opt.dropout)
        self.softmax = nn.Softmax()
        self.dense = nn.Linear(feature, self.opt.polarities_dim, bias=True)

    def forward(self, inputs):
        text_raw_indices, aspect_indices = inputs[0], inputs[1]
        
        x = self.embed(text_raw_indices)
        x_len = torch.sum(text_raw_indices != 0, dim=-1)

        embed = self.squeeze_embedding(x, x_len)
        embed = self.dropout(embed)
        embed = embed.unsqueeze(1)

        out0 = F.tanh(self.conv0(embed)).squeeze(3)
        out1 = F.tanh(self.conv1(embed)).squeeze(3)
        out2 = F.tanh(self.conv2(embed)).squeeze(3)

        max_pool0 = F.max_pool1d(out0, out0.size(2)).squeeze(2)
        max_pool1 = F.max_pool1d(out1, out1.size(2)).squeeze(2)
        max_pool2 = F.max_pool1d(out2, out2.size(2)).squeeze(2)

        cat = torch.cat([max_pool0,max_pool1,max_pool2], dim = 1)
        cat = self.dropout(cat)

        output = self.dense(cat)

        if self.opt.softmax:
            output = self.softmax(output)
        return output

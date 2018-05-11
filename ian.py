# -*- coding: utf-8 -*-
# file: ian.py
# author: songyouwei <youwei0314@gmail.com>
# Copyright (C) 2018. All Rights Reserved.

from train_utils import Instructor
from attention import Attention
import torch
import torch.nn as nn

# Hyper Parameters
embed_dim = 100
hidden_dim = 200
lstm_layers = 1
max_seq_len = 80
polarities_dim = 3
num_epochs = 20
batch_size = 128
learning_rate = 0.001

model_name = 'ian'
dataset = 'twitter'  # twitter / restaurant / laptop
inputs_cols = ['text_raw_indices', 'aspect_indices']
log_step = 10
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class IAN(nn.Module):
    def __init__(self, embedding_matrix):
        super(IAN, self).__init__()
        self.embed = nn.Embedding.from_pretrained(torch.tensor(embedding_matrix, dtype=torch.float))
        self.lstm = nn.LSTM(embed_dim, hidden_dim, lstm_layers, batch_first=True)
        self.attention_aspect = Attention(hidden_dim, score_function='bi_linear')
        self.attention_context = Attention(hidden_dim, score_function='bi_linear')
        self.dense = nn.Linear(hidden_dim*2, polarities_dim)

    def forward(self, inputs):
        text_raw_indices, aspect_indices = inputs[0], inputs[1]
        context = self.embed(text_raw_indices)
        context, (_, _) = self.lstm(context)
        aspect = self.embed(aspect_indices)
        aspect, (_, _) = self.lstm(aspect)

        nonzeros_aspect = torch.tensor(torch.sum(aspect_indices != 0, dim=-1), dtype=torch.float).to(device)
        aspect = torch.sum(aspect, dim=1)
        aspect = torch.div(aspect, nonzeros_aspect.view(nonzeros_aspect.size(0), 1))

        nonzeros_context = torch.tensor(torch.sum(text_raw_indices != 0, dim=-1), dtype=torch.float).to(device)
        context = torch.sum(context, dim=1)
        context = torch.div(context, nonzeros_context.view(nonzeros_context.size(0), 1))

        aspect_final = self.attention_aspect((aspect, context)).squeeze(dim=1)
        context_final = self.attention_context((context, aspect)).squeeze(dim=1)

        x = torch.cat((aspect_final, context_final), dim=-1)
        out = self.dense(x)
        return out


if __name__ == '__main__':
    ins = Instructor(module_class=IAN, model_name=model_name,
                     dataset=dataset, embed_dim=embed_dim, max_seq_len=max_seq_len,
                     batch_size=batch_size)
    ins.run(inputs_cols=inputs_cols,
            learning_rate=learning_rate,
            num_epochs=num_epochs,
            log_step=log_step)

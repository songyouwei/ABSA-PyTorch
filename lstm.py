# -*- coding: utf-8 -*-
# file: lstm.py
# author: songyouwei <youwei0314@gmail.com>
# Copyright (C) 2018. All Rights Reserved.

from train_utils import Instructor
from dynamic_rnn import DynamicLSTM
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

model_name = 'lstm'
dataset = 'twitter'  # twitter / restaurant / laptop
inputs_cols = ['text_raw_indices']
log_step = 10


class LSTM(nn.Module):
    def __init__(self, embedding_matrix):
        super(LSTM, self).__init__()
        self.embed = nn.Embedding.from_pretrained(torch.tensor(embedding_matrix, dtype=torch.float))
        self.lstm = DynamicLSTM(embed_dim, hidden_dim, lstm_layers, batch_first=True)
        self.dense = nn.Linear(hidden_dim, polarities_dim)

    def forward(self, inputs):
        text_raw_indices = inputs[0]
        x = self.embed(text_raw_indices)
        x_len = torch.sum(text_raw_indices != 0, dim=-1)
        _, (h_n, _) = self.lstm(x, x_len)
        out = self.dense(h_n[0])
        return out


if __name__ == '__main__':
    ins = Instructor(module_class=LSTM, model_name=model_name,
                     dataset=dataset, embed_dim=embed_dim, max_seq_len=max_seq_len,
                     batch_size=batch_size)
    ins.run(inputs_cols=inputs_cols,
            learning_rate=learning_rate,
            num_epochs=num_epochs,
            log_step=log_step)

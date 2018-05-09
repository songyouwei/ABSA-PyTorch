# -*- coding: utf-8 -*-
# file: td_lstm
# author: songyouwei <youwei0314@gmail.com>
# Copyright (C) 2018. All Rights Reserved.

from helpers import instructor
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

model_name = 'td_lstm'
dataset = 'restaurant'
inputs_cols = ['text_left_with_aspect_indices', 'text_right_with_aspect_indices']
log_step = 10


class td_lstm(nn.Module):
    def __init__(self, embedding_matrix):
        super(td_lstm, self).__init__()
        self.embed = nn.Embedding.from_pretrained(torch.tensor(embedding_matrix, dtype=torch.float))
        self.lstm_l = nn.LSTM(embed_dim, hidden_dim, lstm_layers, batch_first=True)
        self.lstm_r = nn.LSTM(embed_dim, hidden_dim, lstm_layers, batch_first=True)
        self.dense = nn.Linear(hidden_dim*2, polarities_dim)

    def forward(self, inputs):
        x_l, x_r = inputs[0], inputs[1]
        x_l, x_r = self.embed(x_l), self.embed(x_r)
        _, (h_n_l, c_n_l) = self.lstm_l(x_l)
        _, (h_n_r, c_n_r) = self.lstm_r(x_r)
        h_n = torch.cat((h_n_l, h_n_r), dim=-1)
        out = self.dense(h_n)
        return out


if __name__ == '__main__':
    ins = instructor(module_class=td_lstm, model_name=model_name,
                     dataset=dataset, embed_dim=embed_dim, max_seq_len=max_seq_len,
                     batch_size=batch_size)
    ins.run(inputs_cols=inputs_cols,
            learning_rate=learning_rate,
            num_epochs=num_epochs,
            log_step=log_step)

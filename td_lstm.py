# -*- coding: utf-8 -*-
# file: td_lstm
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

model_name = 'td_lstm'
dataset = 'twitter'  # twitter / restaurant / laptop
inputs_cols = ['text_left_with_aspect_indices', 'text_right_with_aspect_indices']
log_step = 10


class TD_LSTM(nn.Module):
    def __init__(self, embedding_matrix):
        super(TD_LSTM, self).__init__()
        self.embed = nn.Embedding.from_pretrained(torch.tensor(embedding_matrix, dtype=torch.float))
        self.lstm_l = DynamicLSTM(embed_dim, hidden_dim, lstm_layers, batch_first=True)
        self.lstm_r = DynamicLSTM(embed_dim, hidden_dim, lstm_layers, batch_first=True)
        self.dense = nn.Linear(hidden_dim*2, polarities_dim)

    def forward(self, inputs):
        x_l, x_r = inputs[0], inputs[1]
        x_l_len, x_r_len = torch.sum(x_l != 0, dim=-1), torch.sum(x_r != 0, dim=-1)
        x_l, x_r = self.embed(x_l), self.embed(x_r)
        _, (h_n_l, _) = self.lstm_l(x_l, x_l_len)
        _, (h_n_r, _) = self.lstm_r(x_r, x_r_len)
        h_n = torch.cat((h_n_l[0], h_n_r[0]), dim=-1)
        out = self.dense(h_n)
        return out


if __name__ == '__main__':
    ins = Instructor(module_class=TD_LSTM, model_name=model_name,
                     dataset=dataset, embed_dim=embed_dim, max_seq_len=max_seq_len,
                     batch_size=batch_size)
    ins.run(inputs_cols=inputs_cols,
            learning_rate=learning_rate,
            num_epochs=num_epochs,
            log_step=log_step)

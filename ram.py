# -*- coding: utf-8 -*-
# file: ram
# author: songyouwei <youwei0314@gmail.com>
# Copyright (C) 2018. All Rights Reserved.

from train_utils import Instructor
from dynamic_rnn import DynamicLSTM
from attention import Attention
import torch
import torch.nn as nn

# Hyper Parameters
hops = 3
embed_dim = 300
hidden_dim = 600
lstm_layers = 1
max_seq_len = 80
polarities_dim = 3
num_epochs = 50
batch_size = 128
learning_rate = 0.001

model_name = 'ram'
dataset = 'restaurant'  # twitter / restaurant / laptop
inputs_cols = ['text_raw_indices', 'aspect_indices']
log_step = 10
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class RAM(nn.Module):
    @staticmethod
    def locationed_memory(memory, text_raw_without_aspect_indices):
        # here we just simply calculate the location vector in Model2's manner
        lens_memory = torch.tensor(torch.sum(text_raw_without_aspect_indices != 0, dim=-1), dtype=torch.int).to(device)
        for i in range(memory.size(0)):
            start = max_seq_len-int(lens_memory[i])
            for j in range(lens_memory[i]):
                idx = start+j
                memory[i][idx] *= (1-float(j)/int(lens_memory[i]))
        return memory

    def __init__(self, embedding_matrix):
        super(RAM, self).__init__()
        self.embed = nn.Embedding.from_pretrained(torch.tensor(embedding_matrix, dtype=torch.float))
        self.bi_lstm_context = DynamicLSTM(embed_dim, hidden_dim, lstm_layers, batch_first=True, bidirectional=True)
        self.bi_lstm_aspect = DynamicLSTM(embed_dim, hidden_dim, lstm_layers, batch_first=True, bidirectional=True)
        self.attention = Attention(hidden_dim*2, score_function='mlp')
        self.gru_cell = nn.GRUCell(hidden_dim*2, hidden_dim*2)
        self.dense = nn.Linear(hidden_dim*2, polarities_dim)

    def forward(self, inputs):
        text_raw_without_aspect_indices, aspect_indices = inputs[0], inputs[1]
        context_len = torch.sum(text_raw_without_aspect_indices != 0, dim=-1)
        aspect_len = torch.sum(aspect_indices != 0, dim=-1)
        nonzeros_aspect = torch.tensor(aspect_len, dtype=torch.float).to(device)

        memory = self.embed(text_raw_without_aspect_indices)
        memory, (_, _) = self.bi_lstm_context(memory, context_len)
        # memory = RAM.locationed_memory(memory, text_raw_without_aspect_indices)
        aspect = self.embed(aspect_indices)
        aspect, (_, _) = self.bi_lstm_aspect(aspect, aspect_len)
        aspect = torch.sum(aspect, dim=1)
        aspect = torch.div(aspect, nonzeros_aspect.view(nonzeros_aspect.size(0), 1))

        et = aspect
        for _ in range(hops):
            it_al = self.attention(memory, et).squeeze(dim=1)
            et = self.gru_cell(it_al, et)
        out = self.dense(et)
        return out


if __name__ == '__main__':
    ins = Instructor(module_class=RAM, model_name=model_name,
                     dataset=dataset, embed_dim=embed_dim, max_seq_len=max_seq_len,
                     batch_size=batch_size)
    ins.run(inputs_cols=inputs_cols,
            learning_rate=learning_rate,
            num_epochs=num_epochs,
            log_step=log_step)

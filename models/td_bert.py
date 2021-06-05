# -*- coding: utf-8 -*-
# file: td_bert.py
# author: xiangpan <xiangpan@Nyu.edu>
# Copyright (C) 2020. All Rights Reserved.
import torch
import torch.nn as nn
from layers.attention import Attention


class TD_BERT(nn.Module):
    def __init__(self, bert, opt):
        super(TD_BERT, self).__init__()
        self.bert = bert
        self.dropout = nn.Dropout(opt.dropout)
        self.opt = opt
        self.dense = nn.Linear(opt.bert_dim, opt.polarities_dim)

    def forward(self, inputs):
        text_bert_indices, bert_segments_ids, left_context_len, aspect_len = (
            inputs[0],
            inputs[1],
            inputs[2],
            inputs[3],
        )
        

        # 
        bert_outs = self.bert(text_bert_indices, bert_segments_ids)
        # print(bert_outs)
        encoded_layers = bert_outs["last_hidden_state"]
        cls_output =  bert_outs["pooler_output"]

        pooled_list = []
        for i in range(0, encoded_layers.shape[0]):  # batch_size i th batch
            encoded_layers_i = encoded_layers[i]
            left_context_len_i = left_context_len[i]
            aspect_len_i = aspect_len[i]
            e_list = []
            if (left_context_len_i + 1) == (left_context_len_i + 1 + aspect_len_i):
                e_list.append(encoded_layers_i[0])
            else:
                for j in range(left_context_len_i + 1, left_context_len_i + 1 + aspect_len_i):
                    e_list.append(encoded_layers_i[j])
            e = torch.stack(e_list, 0)
            embed = torch.stack([e], 0)
            pooled = nn.functional.max_pool2d(embed, (embed.size(1), 1)).squeeze(1)
            pooled_list.append(pooled)

        pooled_output = torch.cat(pooled_list)
        pooled_output = self.dropout(pooled_output)

        logits = self.dense(pooled_output)

        return logits
    
    # def get_adversarial_outputs(self, input, p_adv):


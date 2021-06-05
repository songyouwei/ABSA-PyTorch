# -*- coding: utf-8 -*-
# file: BERT_SPC.py
# author: songyouwei <youwei0314@gmail.com>
# Copyright (C) 2019. All Rights Reserved.
import torch
import torch.nn as nn
# from .modeling_bert import BertEmbeddings

class BERT_SPC(nn.Module):
    def __init__(self, bert, opt):
        super(BERT_SPC, self).__init__()
        self.bert = bert
        # self.embeddings = self.bert.embeddings
        # self.embeddings.requires_grad=True
        self.dropout = nn.Dropout(opt.dropout)
        self.dense = nn.Linear(opt.bert_dim, opt.polarities_dim)

    def forward(self, inputs):
        text_bert_indices, bert_segments_ids = inputs[0], inputs[1]
        # _, pooled_output = self.bert(text_bert_indices, token_type_ids=bert_segments_ids)
        bert_outs = self.bert(text_bert_indices, token_type_ids=bert_segments_ids)
        last_hidden_state = bert_outs['last_hidden_state']
        pooled_output = bert_outs['pooler_output']
        pooled_output = self.dropout(pooled_output)
        logits = self.dense(pooled_output)
        return logits

    def adv_forward(self, inputs, p_adv = None):
        text_bert_indices, bert_segments_ids = inputs[0], inputs[1]
        word_eb = self.bert.embeddings.word_embeddings(text_bert_indices)
        # print(word_eb.shape)
        if p_adv is not None:
            word_eb += p_adv
        eb = self.bert.embeddings(inputs_embeds = word_eb)
        # eb = self.bert.embeddings(text_bert_indices)
        # bert_outs = self.bert(text_bert_indices, token_type_ids=bert_segments_ids)
        # print(eb == word_eb)
        bert_outs = self.bert(inputs_embeds = eb, token_type_ids = bert_segments_ids)
        eb = None
        last_hidden_state = bert_outs['last_hidden_state']
        pooled_output = bert_outs['pooler_output']
        pooled_output = self.dropout(pooled_output)
        logits = self.dense(pooled_output)
        return logits, eb

    # def forward_with
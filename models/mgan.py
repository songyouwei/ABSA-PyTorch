# -*- coding: utf-8 -*-
# file: mgan.py
# author: gene_zc <gene_zhangchen@163.com>
# Copyright (C) 2018. All Rights Reserved.

from layers.dynamic_rnn import DynamicLSTM
import torch
import torch.nn as nn
import torch.nn.functional as F

class LocationEncoding(nn.Module):
    def __init__(self, opt):
        super(LocationEncoding, self).__init__()
        self.opt = opt

    def forward(self, x, pos_inx):
        batch_size, seq_len = x.size()[0], x.size()[1]
        weight = self.weight_matrix(pos_inx, batch_size, seq_len).to(self.opt.device)
        x = weight.unsqueeze(2) * x
        return x

    def weight_matrix(self, pos_inx, batch_size, seq_len):
        pos_inx = pos_inx.cpu().numpy()
        weight = [[] for i in range(batch_size)]
        for i in range(batch_size):
            for j in range(pos_inx[i][0]):
                relative_pos = pos_inx[i][0] - j
                aspect_len = pos_inx[i][1] - pos_inx[i][0] + 1
                sentence_len = seq_len - aspect_len
                weight[i].append(1 - relative_pos / sentence_len)
            for j in range(pos_inx[i][0], pos_inx[i][1] + 1):
                weight[i].append(0)
            for j in range(pos_inx[i][1] + 1, seq_len):
                relative_pos = j - pos_inx[i][1]
                aspect_len = pos_inx[i][1] - pos_inx[i][0] + 1
                sentence_len = seq_len - aspect_len
                weight[i].append(1 - relative_pos / sentence_len)
        weight = torch.tensor(weight)
        return weight

class AlignmentMatrix(nn.Module):
    def __init__(self, opt):
        super(AlignmentMatrix, self).__init__()
        self.opt = opt
        self.w_u = nn.Parameter(torch.Tensor(6*opt.hidden_dim, 1))

    def forward(self, batch_size, ctx, asp):
        ctx_len = ctx.size(1)
        asp_len = asp.size(1)
        alignment_mat = torch.zeros(batch_size, ctx_len, asp_len).to(self.opt.device)
        ctx_chunks = ctx.chunk(ctx_len, dim=1)
        asp_chunks = asp.chunk(asp_len, dim=1)
        for i, ctx_chunk in enumerate(ctx_chunks):
            for j, asp_chunk in enumerate(asp_chunks):
                feat = torch.cat([ctx_chunk, asp_chunk, ctx_chunk*asp_chunk], dim=2) # batch_size x 1 x 6*hidden_dim 
                alignment_mat[:, i, j] = feat.matmul(self.w_u.expand(batch_size, -1, -1)).squeeze(-1).squeeze(-1) 
        return alignment_mat

class MGAN(nn.Module):
    def __init__(self, embedding_matrix, opt):
        super(MGAN, self).__init__()
        self.opt = opt
        self.embed = nn.Embedding.from_pretrained(torch.tensor(embedding_matrix, dtype=torch.float))
        self.ctx_lstm = DynamicLSTM(opt.embed_dim, opt.hidden_dim, num_layers=1, batch_first=True, bidirectional=True)
        self.asp_lstm = DynamicLSTM(opt.embed_dim, opt.hidden_dim, num_layers=1, batch_first=True, bidirectional=True)
        self.location = LocationEncoding(opt)
        self.w_a2c = nn.Parameter(torch.Tensor(2*opt.hidden_dim, 2*opt.hidden_dim))
        self.w_c2a = nn.Parameter(torch.Tensor(2*opt.hidden_dim, 2*opt.hidden_dim))
        self.alignment = AlignmentMatrix(opt)
        self.dense = nn.Linear(8*opt.hidden_dim, opt.polarities_dim)

    def forward(self, inputs):
        text_raw_indices = inputs[0] # batch_size x seq_len
        aspect_indices = inputs[1] 
        text_left_indices= inputs[2]
        batch_size = text_raw_indices.size(0)
        ctx_len = torch.sum(text_raw_indices != 0, dim=1)
        asp_len = torch.sum(aspect_indices != 0, dim=1)
        left_len = torch.sum(text_left_indices != 0, dim=-1)
        aspect_in_text = torch.cat([left_len.unsqueeze(-1), (left_len+asp_len-1).unsqueeze(-1)], dim=-1)

        ctx = self.embed(text_raw_indices) # batch_size x seq_len x embed_dim
        asp = self.embed(aspect_indices) # batch_size x seq_len x embed_dim

        ctx_out, (_, _) = self.ctx_lstm(ctx, ctx_len) 
        ctx_out = self.location(ctx_out, aspect_in_text) # batch_size x (ctx)seq_len x 2*hidden_dim
        ctx_pool = torch.sum(ctx_out, dim=1)
        ctx_pool = torch.div(ctx_pool, ctx_len.float().unsqueeze(-1)).unsqueeze(-1) # batch_size x 2*hidden_dim x 1

        asp_out, (_, _) = self.asp_lstm(asp, asp_len) # batch_size x (asp)seq_len x 2*hidden_dim
        asp_pool = torch.sum(asp_out, dim=1)
        asp_pool = torch.div(asp_pool, asp_len.float().unsqueeze(-1)).unsqueeze(-1) # batch_size x 2*hidden_dim x 1

        alignment_mat = self.alignment(batch_size, ctx_out, asp_out) # batch_size x (ctx)seq_len x (asp)seq_len
        # batch_size x 2*hidden_dim
        f_asp2ctx = torch.matmul(ctx_out.transpose(1, 2), F.softmax(alignment_mat.max(2, keepdim=True)[0], dim=1)).squeeze(-1)
        f_ctx2asp = torch.matmul(F.softmax(alignment_mat.max(1, keepdim=True)[0], dim=2), asp_out).transpose(1, 2).squeeze(-1) 

        c_asp2ctx_alpha = F.softmax(ctx_out.matmul(self.w_a2c.expand(batch_size, -1, -1)).matmul(asp_pool), dim=1)
        c_asp2ctx = torch.matmul(ctx_out.transpose(1, 2), c_asp2ctx_alpha).squeeze(-1)
        c_ctx2asp_alpha = F.softmax(asp_out.matmul(self.w_c2a.expand(batch_size, -1, -1)).matmul(ctx_pool), dim=1)
        c_ctx2asp = torch.matmul(asp_out.transpose(1, 2), c_ctx2asp_alpha).squeeze(-1)

        feat = torch.cat([c_asp2ctx, f_asp2ctx, f_ctx2asp, c_ctx2asp], dim=1)
        out = self.dense(feat) # bathc_size x polarity_dim

        return out
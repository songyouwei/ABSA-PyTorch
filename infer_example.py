# -*- coding: utf-8 -*-
# file: infer.py
# author: songyouwei <youwei0314@gmail.com>
# Copyright (C) 2019. All Rights Reserved.

from data_utils import ABSADatesetReader
import torch
import torch.nn.functional as F
import argparse

from models import IAN, MemNet, TD_LSTM, ATAE_LSTM, AOA


class Inferer:
    """A simple inference example"""
    def __init__(self, opt):
        self.opt = opt
        absa_dataset = ABSADatesetReader(dataset=opt.dataset, embed_dim=opt.embed_dim, max_seq_len=opt.max_seq_len)
        self.tokenizer = absa_dataset.tokenizer

        self.model = opt.model_class(absa_dataset.embedding_matrix, opt)
        self.model.load_state_dict(torch.load(opt.state_dict_path))
        self.model = self.model.to(opt.device)

    def evaluate(self, raw_texts):
        context_seqs = [self.tokenizer.text_to_sequence(raw_text.lower().strip()) for raw_text in raw_texts]
        aspect_seqs = [self.tokenizer.text_to_sequence('null')] * len(raw_texts)
        context_indices = torch.tensor(context_seqs, dtype=torch.int64).to(self.opt.device)
        aspect_indices = torch.tensor(aspect_seqs, dtype=torch.int64).to(self.opt.device)
        # switch model to evaluation mode
        self.model.eval()
        with torch.no_grad():
            t_inputs = [context_indices, aspect_indices]
            t_outputs = self.model(t_inputs)

        t_probs = F.softmax(t_outputs, dim=-1).cpu().numpy()
        return t_probs


if __name__ == '__main__':
    # Hyper Parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', default='ian', type=str)
    parser.add_argument('--state_dict_path', default='state_dict/ian_restaurant_acc0.7911', type=str)
    parser.add_argument('--dataset', default='restaurant', type=str, help='twitter, restaurant, laptop')
    parser.add_argument('--embed_dim', default=300, type=int)
    parser.add_argument('--hidden_dim', default=300, type=int)
    parser.add_argument('--max_seq_len', default=80, type=int)
    parser.add_argument('--polarities_dim', default=3, type=int)
    parser.add_argument('--device', default=None, type=str)
    opt = parser.parse_args()

    model_classes = {
        'td_lstm': TD_LSTM,
        'atae_lstm': ATAE_LSTM,
        'ian': IAN,
        'memnet': MemNet,
        'aoa': AOA,
    }
    opt.model_class = model_classes[opt.model_name]
    opt.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') \
        if opt.device is None else torch.device(opt.device)

    inf = Inferer(opt)
    t_probs = inf.evaluate(['happy memory', 'the service is terrible', 'just normal food'])
    print(t_probs.argmax(axis=-1) - 1)

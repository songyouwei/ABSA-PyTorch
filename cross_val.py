# -*- coding: utf-8 -*-
# file: cross_val.py
# author: gene_zc <gene_zhangchen@163.com>
# Copyright (C) 2018. All Rights Reserved.
from pytorch_pretrained_bert import BertModel
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
import itertools
import argparse
import math
import os
import random
import numpy as np

from data_utils import build_tokenizer, build_embedding_matrix, Tokenizer4Bert, ABSADataset

from models import LSTM, IAN, MemNet, RAM, TD_LSTM, Cabasc, ATAE_LSTM, TNet_LF, AOA, MGAN
from models.aen import CrossEntropyLoss_LSR, AEN, AEN_BERT
from models.bert_spc import BERT_SPC


class CrossVal:
    def __init__(self, opt):
        self.opt = opt

        if 'bert' in opt.model_name:
            tokenizer = Tokenizer4Bert(opt.max_seq_len, opt.pretrained_bert_name)
            bert = BertModel.from_pretrained(opt.pretrained_bert_name)
            # freeze pretrained bert params
            # for param in bert.parameters():
            #     param.requires_grad = False
            self.model = opt.model_class(bert, opt).to(opt.device)
        else:
            tokenizer = build_tokenizer(
                fnames=[opt.dataset_file['train'], opt.dataset_file['test']],
                max_seq_len=opt.max_seq_len,
                dat_fname='{0}_tokenizer.dat'.format(opt.dataset))
            embedding_matrix = build_embedding_matrix(
                word2idx=tokenizer.word2idx,
                embed_dim=opt.embed_dim,
                dat_fname='{0}_{1}_embedding_matrix.dat'.format(str(opt.embed_dim), opt.dataset))
            self.model = opt.model_class(embedding_matrix, opt).to(opt.device)

        dataset = ABSADataset(opt.dataset_file['train'], tokenizer)
        datalen = len(dataset)
        dataset = [dataset[i*datalen//opt.fold:(i+1)*datalen//opt.fold] for i in range(opt.fold)]
        
        self.dataset = [(list(itertools.chain.from_iterable(dataset[:i]+dataset[i+1:])), dataset[i]) for i in range(opt.fold)]
        if opt.shuffle:
            random.shuffle(self.dataset)

        if opt.device.type == 'cuda':
            print("cuda memory allocated:", torch.cuda.memory_allocated(device=opt.device.index))
        self._print_args()

    def _print_args(self):
        n_trainable_params, n_nontrainable_params = 0, 0
        for p in self.model.parameters():
            n_params = torch.prod(torch.tensor(p.shape))
            if p.requires_grad:
                n_trainable_params += n_params
            else:
                n_nontrainable_params += n_params
        print('n_trainable_params: {0}, n_nontrainable_params: {1}'.format(n_trainable_params, n_nontrainable_params))
        print('> training arguments:')
        for arg in vars(self.opt):
            print('>>> {0}: {1}'.format(arg, getattr(self.opt, arg)))

    def _reset_params(self):
        for child in self.model.children():
            if type(child) != BertModel:  # skip bert params (with unfreezed bert)
                for p in child.parameters():
                    if p.requires_grad:
                        if len(p.shape) > 1:
                            self.opt.initializer(p)
                        else:
                            stdv = 1. / math.sqrt(p.shape[0])
                            torch.nn.init.uniform_(p, a=-stdv, b=stdv)

    def _train(self, criterion, optimizer, fold_idx):
        writer = SummaryWriter(log_dir=self.opt.logdir)
        val_loss_list = []
        global_step = 0
        train_data, val_data = self.dataset[fold_idx]
        train_data_loader = DataLoader(dataset=train_data, batch_size=self.opt.batch_size, shuffle=True)
        val_data_loader = DataLoader(dataset=val_data, batch_size=self.opt.batch_size, shuffle=False)
        for epoch in range(self.opt.num_epoch):
            print('>' * 100)
            print('epoch: ', epoch)
            for i_batch, sample_batched in enumerate(train_data_loader):
                global_step += 1

                # switch model to training mode, clear gradient accumulators
                self.model.train()
                optimizer.zero_grad()

                inputs = [sample_batched[col].to(self.opt.device) for col in self.opt.inputs_cols]
                outputs = self.model(inputs)
                targets = sample_batched['polarity'].to(self.opt.device)

                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()

                if global_step % self.opt.log_step == 0:

                    # switch model to evaluation mode
                    self.model.eval()
                    val_loss = self._evaluate_loss(criterion, val_data_loader)
                    val_loss_list.append(val_loss)

                    writer.add_scalar('train_loss', loss, global_step)
                    writer.add_scalar('val_loss', val_loss, global_step)
                    print('train_loss: {:.4f}, val_loss: {:.4f}'.format(loss.item(), val_loss))

        writer.close()
        return np.mean(val_loss_list)

    def _evaluate_loss(self, criterion, data_loader):
        v_loss_total = 0
        n_val_total = 0
        with torch.no_grad():
            for v_batch, v_sample_batched in enumerate(data_loader):
                v_inputs = [v_sample_batched[col].to(opt.device) for col in self.opt.inputs_cols]
                v_targets = v_sample_batched['polarity'].to(opt.device)
                v_outputs = self.model(v_inputs)

                v_loss = criterion(v_outputs, v_targets)
                v_loss_total += v_loss.item()
                n_val_total += len(v_outputs)

        val_loss = v_loss_total / n_val_total
        return val_loss

    def run(self, repeats=5):
        # Loss and Optimizer
        criterion = nn.CrossEntropyLoss()
        _params = filter(lambda p: p.requires_grad, self.model.parameters())
        optimizer = self.opt.optimizer(_params, lr=self.opt.learning_rate, weight_decay=self.opt.l2reg)

        best_val_loss = float('inf')
        for r in range(repeats):
            print('repeat: '+str(r+1))
            fold_avg_val_losses = []
            for i in range(self.opt.fold):
                print('fold: '+str(i+1))
                self._reset_params()
                fold_avg_val_loss = self._train(criterion, optimizer, i)
                fold_avg_val_losses.append(fold_avg_val_loss)
                print('avg_val_loss: {0}'.format(fold_avg_val_loss))
            val_loss = np.mean(fold_avg_val_losses)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                if not os.path.exists('state_dict'):
                    os.mkdir('state_dict')
                path = 'state_dict/{0}_{1}_acc{2}'.format(self.opt.model_name, self.opt.dataset, round(val_loss, 4))
                torch.save(self.model.state_dict(), path)
                print('>> saved: ' + path)

if __name__ == '__main__':
    # Hyper Parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', default='lstm', type=str)
    parser.add_argument('--dataset', default='twitter', type=str, help='twitter, restaurant, laptop')
    parser.add_argument('--optimizer', default='adam', type=str)
    parser.add_argument('--initializer', default='xavier_uniform_', type=str)
    parser.add_argument('--learning_rate', default=2e-5, type=float)  # try 5e-5, 3e-5, 2e-5 for BERT models (sensitive)
    parser.add_argument('--dropout', default=0.1, type=float)
    parser.add_argument('--l2reg', default=0.01, type=float)
    parser.add_argument('--num_epoch', default=20, type=int)
    parser.add_argument('--batch_size', default=64, type=int)  # try 16, 32, 64 for BERT models
    parser.add_argument('--log_step', default=5, type=int)
    parser.add_argument('--logdir', default='log', type=str)
    parser.add_argument('--embed_dim', default=300, type=int)
    parser.add_argument('--hidden_dim', default=300, type=int)
    parser.add_argument('--bert_dim', default=768, type=int)
    parser.add_argument('--pretrained_bert_name', default='bert-base-uncased', type=str)
    parser.add_argument('--max_seq_len', default=80, type=int)
    parser.add_argument('--polarities_dim', default=3, type=int)
    parser.add_argument('--hops', default=3, type=int)
    parser.add_argument('--fold', default=10, type=int)
    parser.add_argument('--shuffle', default=True, type=bool)
    parser.add_argument('--device', default=None, type=str)
    opt = parser.parse_args()

    model_classes = {
        'lstm': LSTM,
        'td_lstm': TD_LSTM,
        'atae_lstm': ATAE_LSTM,
        'ian': IAN,
        'memnet': MemNet,
        'ram': RAM,
        'cabasc': Cabasc,
        'tnet_lf': TNet_LF,
        'aoa': AOA,
        'mgan': MGAN,
        'bert_spc': BERT_SPC,
        'aen': AEN,
        'aen_bert': AEN_BERT,
    }
    dataset_files = {
        'twitter': {
            'train': './datasets/acl-14-short-data/train.raw',
            'test': './datasets/acl-14-short-data/test.raw'
        },
        'restaurant': {
            'train': './datasets/semeval14/Restaurants_Train.xml.seg',
            'test': './datasets/semeval14/Restaurants_Test_Gold.xml.seg'
        },
        'laptop': {
            'train': './datasets/semeval14/Laptops_Train.xml.seg',
            'test': './datasets/semeval14/Laptops_Test_Gold.xml.seg'
        }
    }
    input_colses = {
        'lstm': ['text_raw_indices'],
        'td_lstm': ['text_left_with_aspect_indices', 'text_right_with_aspect_indices'],
        'atae_lstm': ['text_raw_indices', 'aspect_indices'],
        'ian': ['text_raw_indices', 'aspect_indices'],
        'memnet': ['text_raw_without_aspect_indices', 'aspect_indices'],
        'ram': ['text_raw_indices', 'aspect_indices', 'text_left_indices'],
        'cabasc': ['text_raw_indices', 'aspect_indices', 'text_left_with_aspect_indices', 'text_right_with_aspect_indices'],
        'tnet_lf': ['text_raw_indices', 'aspect_indices', 'aspect_in_text'],
        'aoa': ['text_raw_indices', 'aspect_indices'],
        'mgan': ['text_raw_indices', 'aspect_indices', 'text_left_indices'],
        'bert_spc': ['text_bert_indices', 'bert_segments_ids'],
        'aen': ['text_raw_indices', 'aspect_indices'],
        'aen_bert' : ['text_raw_bert_indices', 'aspect_bert_indices'],
    }
    initializers = {
        'xavier_uniform_': torch.nn.init.xavier_uniform_,
        'xavier_normal_': torch.nn.init.xavier_normal,
        'orthogonal_': torch.nn.init.orthogonal_,
    }
    optimizers = {
        'adadelta': torch.optim.Adadelta,  # default lr=1.0
        'adagrad': torch.optim.Adagrad,  # default lr=0.01
        'adam': torch.optim.Adam,  # default lr=0.001
        'adamax': torch.optim.Adamax,  # default lr=0.002
        'asgd': torch.optim.ASGD,  # default lr=0.01
        'rmsprop': torch.optim.RMSprop,  # default lr=0.01
        'sgd': torch.optim.SGD,
    }
    opt.model_class = model_classes[opt.model_name]
    opt.dataset_file = dataset_files[opt.dataset]
    opt.inputs_cols = input_colses[opt.model_name]
    opt.initializer = initializers[opt.initializer]
    opt.optimizer = optimizers[opt.optimizer]
    opt.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') \
        if opt.device is None else torch.device(opt.device)

    cv = CrossVal(opt)
    cv.run()
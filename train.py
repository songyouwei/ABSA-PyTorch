# -*- coding: utf-8 -*-
# file: train.py
# author: songyouwei <youwei0314@gmail.com>
# Copyright (C) 2018. All Rights Reserved.

from data_utils import ABSADatesetReader
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
import argparse

from models.lstm import LSTM
from models.ian import IAN
from models.memnet import MemNet
from models.ram import RAM
from models.td_lstm import TD_LSTM
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Instructor:
    def __init__(self, opt):
        print('> training arguments:')
        for arg in vars(opt):
            print('>>> {0}: {1}'.format(arg, getattr(opt, arg)))

        absa_dataset = ABSADatesetReader(dataset=opt.dataset, embed_dim=opt.embed_dim, max_seq_len=opt.max_seq_len)
        self.train_data_loader = DataLoader(dataset=absa_dataset.train_data, batch_size=opt.batch_size, shuffle=True)
        self.test_data_loader = DataLoader(dataset=absa_dataset.test_data, batch_size=len(absa_dataset.test_data), shuffle=False)
        self.writer = SummaryWriter(log_dir=opt.logdir)

        self.model = opt.model_class(absa_dataset.embedding_matrix, opt).to(device)
        self.reset_parameters()

    def reset_parameters(self):
        n_trainable_params, n_nontrainable_params = 0, 0
        for p in self.model.parameters():
            n_params = torch.prod(torch.tensor(p.shape))
            if p.requires_grad:
                n_trainable_params += n_params
                if len(p.shape) > 1:
                    nn.init.xavier_uniform_(p)
            else:
                n_nontrainable_params += n_params
        print('n_trainable_params: {0}, n_nontrainable_params: {1}'.format(n_trainable_params, n_nontrainable_params))

    def run(self):
        # Loss and Optimizer
        criterion = nn.CrossEntropyLoss()
        params = filter(lambda p: p.requires_grad, self.model.parameters())
        optimizer = opt.optimizer(params, lr=opt.learning_rate)

        max_test_acc = 0
        global_step = 0
        for epoch in range(opt.num_epoch):
            print('>' * 100)
            print('epoch: ', epoch)
            n_correct, n_total = 0, 0
            for i_batch, sample_batched in enumerate(self.train_data_loader):
                global_step += 1

                # switch model to training mode, clear gradient accumulators
                self.model.train()
                optimizer.zero_grad()

                inputs = [sample_batched[col].to(device) for col in opt.inputs_cols]
                targets = sample_batched['polarity'].to(device)
                outputs = self.model(inputs)

                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()

                if global_step % opt.log_step == 0:
                    n_correct += (torch.argmax(outputs, -1) == targets).sum().item()
                    n_total += len(outputs)
                    train_acc = n_correct / n_total

                    # switch model to evaluation mode
                    self.model.eval()
                    n_test_correct, n_test_total = 0, 0
                    with torch.no_grad():
                        for t_batch, t_sample_batched in enumerate(self.test_data_loader):
                            t_inputs = [t_sample_batched[col].to(device) for col in opt.inputs_cols]
                            t_targets = t_sample_batched['polarity'].to(device)
                            t_outputs = self.model(t_inputs)

                            n_test_correct += (torch.argmax(t_outputs, -1) == t_targets).sum().item()
                            n_test_total += len(t_outputs)
                        test_acc = n_test_correct / n_test_total
                        if test_acc > max_test_acc:
                            max_test_acc = test_acc

                        print('loss: {:.4f}, acc: {:.4f}, test_acc: {:.4f}'.format(loss.item(), train_acc, test_acc))

                        # log
                        self.writer.add_scalar('loss', loss, global_step)
                        self.writer.add_scalar('acc', train_acc, global_step)
                        self.writer.add_scalar('test_acc', test_acc, global_step)

        self.writer.close()

        print('max_test_acc: {0}'.format(max_test_acc))


if __name__ == '__main__':
    # Hyper Parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', default='lstm', type=str)
    parser.add_argument('--dataset', default='twitter', type=str, help='twitter, restaurant, laptop')
    parser.add_argument('--optimizer', default='adam', type=str)
    parser.add_argument('--initializer', default='xavier_uniform_', type=str)
    parser.add_argument('--learning_rate', default=0.001, type=int)
    parser.add_argument('--num_epoch', default=20, type=int)
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--log_step', default=10, type=int)
    parser.add_argument('--logdir', default='log', type=str)
    parser.add_argument('--embed_dim', default=100, type=int)
    parser.add_argument('--hidden_dim', default=200, type=int)
    parser.add_argument('--max_seq_len', default=80, type=int)
    parser.add_argument('--polarities_dim', default=3, type=int)
    parser.add_argument('--hops', default=3, type=int)
    opt = parser.parse_args()

    model_classes = {
        'lstm': LSTM,
        'td_lstm': TD_LSTM,
        'ian': IAN,
        'memnet': MemNet,
        'ram': RAM,
    }
    input_colses = {
        'lstm': ['text_raw_indices'],
        'td_lstm': ['text_left_with_aspect_indices', 'text_right_with_aspect_indices'],
        'ian': ['text_raw_indices', 'aspect_indices'],
        'memnet': ['text_raw_without_aspect_indices', 'aspect_indices'],
        'ram': ['text_raw_indices', 'aspect_indices'],
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
    opt.inputs_cols = input_colses[opt.model_name]
    opt.initializer = initializers[opt.initializer]
    opt.optimizer = optimizers[opt.optimizer]
    opt.device = device

    ins = Instructor(opt)
    ins.run()

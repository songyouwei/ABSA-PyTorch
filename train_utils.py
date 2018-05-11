# -*- coding: utf-8 -*-
# file: train_utils.py
# author: songyouwei <youwei0314@gmail.com>
# Copyright (C) 2018. All Rights Reserved.

from data_utils import ABSADatesetReader
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Instructor:
    def __init__(self, module_class, model_name, dataset='twitter', embed_dim=100, max_seq_len=40, batch_size=128):
        absa_dataset = ABSADatesetReader(dataset=dataset, embed_dim=embed_dim, max_seq_len=max_seq_len)
        self.train_data_loader = DataLoader(dataset=absa_dataset.train_data, batch_size=batch_size, shuffle=True)
        self.test_data_loader = DataLoader(dataset=absa_dataset.test_data, batch_size=len(absa_dataset.test_data), shuffle=False)
        self.writer = SummaryWriter(log_dir='{0}_logs'.format(model_name))

        self.model = module_class(absa_dataset.embedding_matrix).to(device)

    def run(self, inputs_cols, learning_rate=0.001, num_epochs=20, log_step=5):
        # Loss and Optimizer
        criterion = nn.CrossEntropyLoss()
        params = filter(lambda p: p.requires_grad, self.model.parameters())
        optimizer = torch.optim.Adam(params, lr=learning_rate)

        max_test_acc = 0
        global_step = 0
        for epoch in range(num_epochs):
            print('>' * 100)
            print('epoch: ', epoch)
            n_correct, n_total = 0, 0
            for i_batch, sample_batched in enumerate(self.train_data_loader):
                global_step += 1

                # switch model to training mode, clear gradient accumulators
                self.model.train()
                optimizer.zero_grad()

                inputs = [sample_batched[col].to(device) for col in inputs_cols]
                targets = sample_batched['polarity'].to(device)
                outputs = self.model(inputs)

                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()

                if global_step % log_step == 0:
                    n_correct += (torch.argmax(outputs, -1) == targets).sum().item()
                    n_total += len(outputs)
                    train_acc = n_correct / n_total

                    # switch model to evaluation mode
                    self.model.eval()
                    n_test_correct, n_test_total = 0, 0
                    with torch.no_grad():
                        for t_batch, t_sample_batched in enumerate(self.test_data_loader):
                            t_inputs = [t_sample_batched[col].to(device) for col in inputs_cols]
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

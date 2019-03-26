# -*- coding: utf-8 -*-
# file: data_utils.py
# author: songyouwei <youwei0314@gmail.com>
# Copyright (C) 2018. All Rights Reserved.

import numpy as np
import torch
from torch.utils.data import Dataset
from pytorch_pretrained_bert import BertTokenizer


class Tokenizer:
    def __init__(self, max_seq_len):
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.max_seq_len = max_seq_len

    @staticmethod
    def pad_and_truncate(sequence, maxlen, dtype='int64', padding='post', truncating='post', value=0):
        x = (np.ones(maxlen) * value).astype(dtype)
        if truncating == 'pre':
            trunc = sequence[-maxlen:]
        else:
            trunc = sequence[:maxlen]
        trunc = np.asarray(trunc, dtype=dtype)
        if padding == 'post':
            x[:len(trunc)] = trunc
        else:
            x[-len(trunc):] = trunc
        return x

    def text_to_sequence(self, text, reverse=False):
        ids = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(text))
        if reverse:
            ids = ids[::-1]
        return Tokenizer.pad_and_truncate(ids, self.max_seq_len)


class ABSADataset(Dataset):
    def __init__(self, fname, tokenizer):
        fin = open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
        lines = fin.readlines()
        fin.close()

        all_data = []
        for i in range(0, len(lines), 3):
            text_left, _, text_right = [s.lower().strip() for s in lines[i].partition("$T$")]
            aspect = lines[i + 1].lower().strip()
            polarity = lines[i + 2].strip()

            text_raw_indices = tokenizer.text_to_sequence(text_left + " " + aspect + " " + text_right)
            text_raw_without_aspect_indices = tokenizer.text_to_sequence(text_left + " " + text_right)
            text_left_indices = tokenizer.text_to_sequence(text_left)
            text_left_with_aspect_indices = tokenizer.text_to_sequence(text_left + " " + aspect)
            text_right_indices = tokenizer.text_to_sequence(text_right, reverse=True)
            text_right_with_aspect_indices = tokenizer.text_to_sequence(" " + aspect + " " + text_right, reverse=True)
            aspect_indices = tokenizer.text_to_sequence(aspect)
            left_context_len = np.sum(text_left_indices != 0)
            aspect_len = np.sum(aspect_indices != 0)
            aspect_in_text = torch.tensor([left_context_len.item(), (left_context_len + aspect_len - 1).item()])
            polarity = int(polarity) + 1

            text_bert_indices = tokenizer.text_to_sequence('[CLS] ' + text_left + " " + aspect + " " + text_right + ' [SEP] ' + aspect + ' [SEP]')
            bert_segments_ids = np.asarray([0] * (np.sum(text_raw_indices != 0) + 2) + [1] * (aspect_len + 1))
            bert_segments_ids = Tokenizer.pad_and_truncate(bert_segments_ids, tokenizer.max_seq_len)

            data = {
                'text_bert_indices': text_bert_indices,
                'bert_segments_ids': bert_segments_ids,
                'text_raw_indices': text_raw_indices,
                'text_raw_without_aspect_indices': text_raw_without_aspect_indices,
                'text_left_indices': text_left_indices,
                'text_left_with_aspect_indices': text_left_with_aspect_indices,
                'text_right_indices': text_right_indices,
                'text_right_with_aspect_indices': text_right_with_aspect_indices,
                'aspect_indices': aspect_indices,
                'aspect_in_text': aspect_in_text,
                'polarity': polarity,
            }

            all_data.append(data)
        self.data = all_data

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)

# -*- coding: utf-8 -*-
# file: data_utils.py
# author: songyouwei <youwei0314@gmail.com>
# Copyright (C) 2018. All Rights Reserved.

import numpy as np
import torch
import torch.nn.functional as F
from models.lcf_bert import LCF_BERT
from models.aen import CrossEntropyLoss_LSR, AEN_BERT
from models.bert_spc import BERT_SPC
from pytorch_pretrained_bert import BertModel
from data_utils import Tokenizer4Bert



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

def prepare_data(text_left, aspect, text_right, tokenizer):
    text_left = text_left.lower().strip()
    text_right = text_right.lower().strip()
    aspect = aspect.lower().strip()
    
    text_raw_indices = tokenizer.text_to_sequence(text_left + " " + aspect + " " + text_right)            
    aspect_indices = tokenizer.text_to_sequence(aspect)
    aspect_len = np.sum(aspect_indices != 0)
    text_bert_indices = tokenizer.text_to_sequence('[CLS] ' + text_left + " " + aspect + " " + text_right + ' [SEP] ' + aspect + " [SEP]")
    text_raw_bert_indices = tokenizer.text_to_sequence(
        "[CLS] " + text_left + " " + aspect + " " + text_right + " [SEP]")
    bert_segments_ids = np.asarray([0] * (np.sum(text_raw_indices != 0) + 2) + [1] * (aspect_len + 1))
    bert_segments_ids = pad_and_truncate(bert_segments_ids, tokenizer.max_seq_len)
    aspect_bert_indices = tokenizer.text_to_sequence("[CLS] " + aspect + " [SEP]")

    return text_bert_indices, bert_segments_ids, text_raw_bert_indices, aspect_bert_indices

    
if __name__ == '__main__':

    model_classes = {
        'bert_spc': BERT_SPC,
        'aen_bert': AEN_BERT,
        'lcf_bert': LCF_BERT
    }
    # set your trained models here
    model_state_dict_paths = {
        'bert_spc': 'state_dict/bert_spc_restaurant_val_acc0.8179'
    }
    
    class Option(object): pass
    opt = Option()
    opt.model_name = 'bert_spc'
    opt.model_class = model_classes[opt.model_name]
    opt.state_dict_path = model_state_dict_paths[opt.model_name]
    opt.max_seq_len = 80
    opt.pretrained_bert_name='bert-base-uncased'
    opt.polarities_dim = 3
    opt.dropout = 0.1
    opt.bert_dim = 768
    opt.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    
    tokenizer = Tokenizer4Bert(opt.max_seq_len, opt.pretrained_bert_name)
    
    bert = BertModel.from_pretrained(opt.pretrained_bert_name)
    model = opt.model_class(bert, opt).to(opt.device)
    
    print('loading model {0} ...'.format(opt.model_name))
    model.load_state_dict(torch.load(opt.state_dict_path))
    model.eval()
    torch.autograd.set_grad_enabled(False)


    
    # input: This little place has a cute interior decor and affordable city prices.
    # text_left = This little place has a cute 
    # aspect = interior decor
    # text_right = and affordable city prices.
    
    text_bert_indices, bert_segments_ids, text_raw_bert_indices, aspect_bert_indices = \
        prepare_data('This little place has a cute', 'interior decor', 'and affordable city prices.', tokenizer)
    
    
    text_bert_indices = torch.tensor([text_bert_indices], dtype=torch.int64).to(opt.device)
    bert_segments_ids = torch.tensor([bert_segments_ids], dtype=torch.int64).to(opt.device)
    text_raw_bert_indices = torch.tensor([text_raw_bert_indices], dtype=torch.int64).to(opt.device)
    aspect_bert_indices = torch.tensor([aspect_bert_indices], dtype=torch.int64).to(opt.device)
    if 'lcf' in opt.state_dict_path:
        inputs = [text_bert_indices, bert_segments_ids, text_raw_bert_indices, aspect_bert_indices]
    elif 'aen' in opt.state_dict_path:
        inputs = [text_raw_bert_indices, aspect_bert_indices]
    elif 'spc' in opt.state_dict_path:
        inputs = [text_bert_indices, bert_segments_ids]
    outputs = model(inputs)
    t_probs = F.softmax(outputs, dim=-1).cpu().numpy()
    print('t_probs = ',t_probs)
    print('aspect sentiment = ', t_probs.argmax(axis=-1) - 1)

    
   
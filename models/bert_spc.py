import torch.nn as nn
from torch.autograd.variable import Variable

class BERT_SPC(nn.Module):
    def __init__(self, bert, opt):
        super(BERT_SPC, self).__init__()
        self.bert = bert
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
        word_eb = Variable(word_eb, requires_grad = True)
        if p_adv is not None:
            new_word_eb = p_adv + word_eb
            eb = self.bert.embeddings(inputs_embeds = new_word_eb)
        else:
            eb = self.bert.embeddings(inputs_embeds = word_eb)
        bert_outs = self.bert(inputs_embeds = eb, token_type_ids = bert_segments_ids)
        last_hidden_state = bert_outs['last_hidden_state']
        pooled_output = bert_outs['pooler_output']
        pooled_output = self.dropout(pooled_output)
        logits = self.dense(pooled_output)
        return logits, word_eb

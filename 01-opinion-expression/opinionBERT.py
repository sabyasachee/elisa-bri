import numpy as np
import torch
from torch import nn
from transformers import BertModel
from torchcrf import CRF

class opinionBERT(nn.Module):
    
    def __init__(self, bert_name:str, num_labels:int, num_layers:int, hidden_size:int, dropout_prob:float, rnn_type:str, bidirectional:bool, use_crf:bool, freeze_bert:bool):
        
        super().__init__()
        self.bert = BertModel.from_pretrained(bert_name)
        if freeze_bert:
            self.bert.requires_grad = False
        if num_layers > 0:
            if rnn_type == "gru":
                self.rnn = nn.GRU(self.bert.config.hidden_size, hidden_size, num_layers=num_layers, bidirectional=bidirectional, batch_first=True)
            else:
                self.rnn = nn.LSTM(self.bert.config.hidden_size, hidden_size, num_layers=num_layers, bidirectional=bidirectional, batch_first=True)
        else:
            self.rnn = nn.Identity()
        self.classifier = nn.Linear((1 + bidirectional) * hidden_size, num_labels)
        self.dropout = nn.Dropout(dropout_prob)
        self.use_crf = use_crf
        if self.use_crf:
            self.crf = CRF(num_labels, batch_first=True)
        
    def forward(self, input_ids, attn_mask, crf_attn_mask, tags=None, class_weights=None):
        bert_output = self.bert(input_ids, attn_mask)
        bert_output = bert_output.last_hidden_state
        bert_output = self.dropout(bert_output)
        
        rnn_output, _ = self.rnn(bert_output)
        
        logits = self.classifier(rnn_output)

        if self.use_crf:
            pred = self.crf.decode(logits, crf_attn_mask)
        else:
            detached_logits = logits.detach().cpu().numpy()
            pred = [list(sentence_pred) for sentence_pred in np.argmax(detached_logits, axis=2)]

        if tags is not None:
            if self.use_crf:
                loss = -self.crf(logits, tags, mask=crf_attn_mask, reduction="mean")
            else:
                num_labels = logits.shape[-1]
                if class_weights is not None:
                    loss_fct = nn.CrossEntropyLoss(weight=class_weights)
                else:
                    loss_fct = nn.CrossEntropyLoss()
                active_loss = attn_mask.view(-1) == 1
                active_logits = logits.view(-1, num_labels)
                active_labels = torch.where(active_loss, tags.view(-1), torch.Tensor([loss_fct.ignore_index]).type_as(tags)).long()
                loss = loss_fct(active_logits, active_labels)
            return loss, pred
        else:
            return pred
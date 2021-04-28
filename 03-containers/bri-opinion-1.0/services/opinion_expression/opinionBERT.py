import torch
from torch import nn
from transformers import BertModel, BertConfig

class opinionBERT(nn.Module):
    
    def __init__(self, bert_name, num_labels, num_layers, hidden_size, dropout_prob):
        
        super().__init__()
        self.bert = BertModel(BertConfig.from_pretrained(bert_name))
        self.gru = nn.GRU(self.bert.config.hidden_size, hidden_size, num_layers=num_layers, bidirectional=True, batch_first=True)
        self.classifier = nn.Linear(2 * hidden_size, num_labels)
        self.dropout = nn.Dropout(dropout_prob)
        
    def forward(self, input_ids, attn_mask, tags=None, class_weights=None):
        
        bert_output = self.bert(input_ids, attn_mask)
        bert_output = bert_output.last_hidden_state
        bert_output = self.dropout(bert_output)
        
        gru_output, _ = self.gru(bert_output)
        
        logits = self.classifier(gru_output)
        
        if tags is not None:
            num_labels = logits.shape[-1]
            if class_weights is not None:
                loss_fct = nn.CrossEntropyLoss(weight=class_weights)
            else:
                loss_fct = nn.CrossEntropyLoss()
            active_loss = attn_mask.view(-1) == 1
            active_logits = logits.view(-1, num_labels)
            active_labels = torch.where(active_loss, tags.view(-1), torch.Tensor([loss_fct.ignore_index]).type_as(tags)).long()
            loss = loss_fct(active_logits, active_labels)
            return (loss, logits)
        else:
            return logits
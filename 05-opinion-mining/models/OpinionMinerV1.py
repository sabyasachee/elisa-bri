# author: Sabyasachee

import numpy as np
import torch
from torch import nn
from transformers import BertModel, BertConfig
from torchcrf import CRF
import config

class OpinionRoleLabeller(nn.Module):

    def __init__(self, pretrained_model_name):
        super().__init__()
        self.encoder = BertModel.from_pretrained(pretrained_model_name)
        self.hidden_size = self.encoder.config.hidden_size
        self.holder_classifier = nn.Linear(self.hidden_size, 3)
        self.entity_classifier = nn.Linear(self.hidden_size, 3)
        self.event_classifier = nn.Linear(self.hidden_size, 3)
        

    def forward(self, input_ids, mask, holder_labels=None, entity_labels=None, event_labels=None):
        embedding = self.encoder(input_ids, mask).last_hidden_state
        holder_logits = self.holder_classifier(embedding)
        entity_logits = self.entity_classifier(embedding)
        event_logits = self.event_classifier(embedding)
        device = next(self.parameters()).device

        holder_loss, entity_loss, event_loss = None, None, None
        
        if holder_labels is not None:
            holder_class_weight = torch.zeros(3, dtype=torch.long, device=device)
            for i in range(3):
                holder_class_weight[i] = ((holder_labels == i) & (mask == 1)).sum()
            holder_class_weight = holder_class_weight.sum()/(1 + holder_class_weight)
            holder_loss_function = nn.CrossEntropyLoss(weight=holder_class_weight, ignore_index=-1000)
            holder_labels[mask == 0] = holder_loss_function.ignore_index
            holder_loss = holder_loss_function(holder_logits.view((-1, 3)), holder_labels.view(-1))
        
        if entity_labels is not None:
            entity_class_weight = torch.zeros(3, dtype=torch.long, device=device)
            for i in range(3):
                entity_class_weight[i] = ((entity_labels == i) & (mask == 1)).sum()
            entity_class_weight = entity_class_weight.sum()/(1 + entity_class_weight)
            entity_loss_function = nn.CrossEntropyLoss(weight=entity_class_weight, ignore_index=-1000)
            entity_labels[mask == 0] = entity_loss_function.ignore_index
            entity_loss = entity_loss_function(entity_logits.view((-1, 3)), entity_labels.view(-1))
        
        if event_labels is not None:
            event_class_weight = torch.zeros(3, dtype=torch.long, device=device)
            for i in range(3):
                event_class_weight[i] = ((event_labels == i) & (mask == 1)).sum()
            event_class_weight = event_class_weight.sum()/(1 + event_class_weight)
            event_loss_function = nn.CrossEntropyLoss(weight=event_class_weight, ignore_index=-1000)
            event_labels[mask == 0] = event_loss_function.ignore_index
            event_loss = event_loss_function(event_logits.view((-1, 3)), event_labels.view(-1))

        holder_pred = torch.argmax(holder_logits, dim=2)
        holder_pred = [[holder_pred[i, j] for j in range(mask[i].sum())] for i in range(len(holder_pred))]
        entity_pred = torch.argmax(entity_logits, dim=2)
        entity_pred = [[entity_pred[i, j] for j in range(mask[i].sum())] for i in range(len(entity_pred))]
        event_pred = torch.argmax(event_logits, dim=2)
        event_pred = [[event_pred[i, j] for j in range(mask[i].sum())] for i in range(len(event_pred))]

        return (holder_loss, holder_pred), (entity_loss, entity_pred), (event_loss, event_pred)

class OpinionMinerWithoutPredicateAttitudeClassified(nn.Module):

    def __init__(self, pretrained_model_name, special_token_to_indices, max_seq_len, position_embedding_size):
        super().__init__()
        self.special_token_to_indices = special_token_to_indices
        self.max_seq_len: int = max_seq_len
        self.hidden_size = BertConfig.from_pretrained(pretrained_model_name).hidden_size
        self.position_embedding_size = position_embedding_size

        self.opinion_role_labeller = OpinionRoleLabeller(pretrained_model_name)

        self.holder_relative_position_embedding = nn.Embedding(2*self.max_seq_len, self.position_embedding_size)
        self.target_relative_position_embedding = nn.Embedding(2*self.max_seq_len, self.position_embedding_size)

        self.non_span_holder_encoder = BertModel.from_pretrained(pretrained_model_name)
        self.non_span_holder_bilstm = nn.LSTM(self.hidden_size + self.position_embedding_size, self.hidden_size, batch_first=True, bidirectional=True)
        self.non_span_holder_attitude_classifier = nn.Linear(2*self.hidden_size, 7)

        self.span_holder_encoder = BertModel.from_pretrained(pretrained_model_name)
        self.span_holder_bilstm = nn.LSTM(self.hidden_size + 2*self.position_embedding_size, self.hidden_size, batch_first=True, bidirectional=True)
        self.span_holder_attitude_classifier = nn.Linear(2*self.hidden_size, 4)

    def convert_predictions_to_spans(self, pred, max_span_size):
        spans = []
        n_spans = np.zeros(len(pred), dtype=int)

        for i, p in enumerate(pred):
            j, n = 0, 0
            s = []

            while j < len(p):
                if p[j] == 1:
                    k = j + 1
                    while k < len(p) and p[k] == 2:
                        k += 1
                    if k - j <= max_span_size:
                        s.append([j, k])
                        n += 1
                    j = k
                else:
                    j += 1
            
            spans.append(s)
            n_spans[i] = n
        
        return spans, n_spans

    def create_non_span_holder_input(self, input_ids, mask, target_spans, n_target_spans, seq_len, target_max_span_size, device):
        i = 0

        non_span_holder_input = torch.full((n_target_spans, seq_len + target_max_span_size + 3), self.special_token_to_indices["PAD"], device=device)
        non_span_holder_input[:,0] = self.special_token_to_indices["CLS"]
        non_span_holder_input[:,seq_len + 1] = non_span_holder_input[:,seq_len + target_max_span_size + 2] = self.special_token_to_indices["SEP"]

        non_span_holder_mask = torch.zeros((n_target_spans, seq_len + target_max_span_size + 3), device=device)
        non_span_holder_mask[:,0] = non_span_holder_mask[:,seq_len + 1] = non_span_holder_mask[:,seq_len + target_max_span_size + 2] = 1

        target_relative_position_ids = torch.arange(0, seq_len, device=device).repeat(n_target_spans, 1)

        for j, tspans in enumerate(target_spans):
            length = mask[j].sum()

            for k, l in tspans:
                non_span_holder_input[i,1:1 + seq_len] = input_ids[j]
                non_span_holder_input[i,seq_len + 2: seq_len + 2 + l - k] = input_ids[j,k:l]
                non_span_holder_input[i,k + 1:l + 1] = self.special_token_to_indices["TARGET"]

                non_span_holder_mask[i,1:1 + seq_len] = mask[j]
                non_span_holder_mask[i,seq_len + 2: seq_len + 2 + l - k] = 1

                target_relative_position_ids[i,:k] -= k
                target_relative_position_ids[i,k:l] = 0
                target_relative_position_ids[i,l:length] -= l
                target_relative_position_ids[i,length:] = self.max_seq_len

                i += 1
        
        target_relative_position_ids += self.max_seq_len
        return non_span_holder_input, non_span_holder_mask, target_relative_position_ids

    def create_span_holder_input(self, input_ids, mask, holder_spans, target_spans, n_tuples, seq_len, holder_max_span_size, target_max_span_size, device):
        i = 0

        span_holder_input = torch.full((n_tuples, seq_len + holder_max_span_size + target_max_span_size + 4), self.special_token_to_indices["PAD"], device=device)
        span_holder_input[:,0] = self.special_token_to_indices["PAD"]
        span_holder_input[:,seq_len + 1] = span_holder_input[:,seq_len + holder_max_span_size + 2] = span_holder_input[:,seq_len + holder_max_span_size + target_max_span_size + 3] = self.special_token_to_indices["SEP"]

        span_holder_mask = torch.zeros((n_tuples, seq_len + holder_max_span_size + target_max_span_size + 4), device=device)
        span_holder_mask[:,0] = span_holder_mask[:,seq_len + 1] = span_holder_mask[:,seq_len + holder_max_span_size + 2] = span_holder_mask[:,seq_len + holder_max_span_size + target_max_span_size + 3] = 1

        holder_relative_position_ids = torch.arange(0, seq_len, device=device).repeat(n_tuples, 1)
        target_relative_position_ids = torch.arange(0, seq_len, device=device).repeat(n_tuples, 1)

        for j, (hspans, tspans) in enumerate(zip(holder_spans, target_spans)):
            length = mask[j].sum()

            for p, q in hspans:
                for r, s in tspans:
                    span_holder_input[i,1:1 + seq_len] = input_ids[j]
                    span_holder_input[i,seq_len + 2: seq_len + 2 + q - p] = input_ids[j,p:q]
                    span_holder_input[i,seq_len + holder_max_span_size + 3: seq_len + holder_max_span_size + 3 + s - r] = input_ids[j,r:s]
                    span_holder_input[i,p + 1:q + 1] = self.special_token_to_indices["HOLDER"]
                    span_holder_input[i,r + 1:s + 1] = self.special_token_to_indices["TARGET"]

                    span_holder_mask[i,1:seq_len + 1] = mask[j]
                    span_holder_mask[i,seq_len + 2:seq_len + 2 + q - p] = span_holder_mask[i,seq_len + holder_max_span_size + 3: seq_len + holder_max_span_size + 3 + s - r] = 1

                    holder_relative_position_ids[i,:p] -= p
                    holder_relative_position_ids[i,p:q] = 0
                    holder_relative_position_ids[i,q:length] -= q
                    holder_relative_position_ids[i,length:] = self.max_seq_len

                    target_relative_position_ids[i,:r] -= r
                    target_relative_position_ids[i,r:s] = 0
                    target_relative_position_ids[i,s:length] -= s
                    target_relative_position_ids[i,length:] = self.max_seq_len

                    i += 1
        
        holder_relative_position_ids += self.max_seq_len
        target_relative_position_ids += self.max_seq_len

        return span_holder_input, span_holder_mask, holder_relative_position_ids, target_relative_position_ids

    def create_labels(self, label_tuples, holder_spans, target_spans, n_target_spans, n_tuples, device):
        non_span_holder_label = torch.zeros(n_target_spans, dtype=torch.long, device=device)
        span_holder_label = torch.zeros(n_tuples, dtype=torch.long, device=device)

        i = 0
        for (tspans, ltuples) in zip(target_spans, label_tuples):
            for tspan in tspans:
                tspan = tuple(tspan)
                if tspan in ltuples["target_to_label"]:
                    non_span_holder_label[i] = ltuples["target_to_label"][tspan]
                i += 1
        
        i = 0
        for (hspans, tspans, ltuples) in zip(holder_spans, target_spans, label_tuples):
            for hspan in hspans:
                for tspan in tspans:
                    key = tuple(hspan + tspan)
                    if key in ltuples["holder_and_target_to_label"]:
                        span_holder_label[i] = ltuples["holder_and_target_to_label"][key]
                    i += 1
        
        return non_span_holder_label, span_holder_label

    def forward(self, input_ids, mask, label_tuples=None, holder_labels=None, entity_labels=None, event_labels=None, roles_only=False):
        device = next(self.parameters()).device
        batch_seq_len = input_ids.shape[1]
        (holder_loss, holder_pred), (entity_loss, entity_pred), (event_loss, event_pred) = self.opinion_role_labeller(input_ids, mask, holder_labels, entity_labels, event_labels)
        
        holder_max_span_size = config.max_holder_size
        entity_max_span_size = config.max_entity_size
        event_max_span_size = config.max_event_size

        holder_spans, holder_n_spans = self.convert_predictions_to_spans(holder_pred, holder_max_span_size)
        entity_spans, entity_n_spans = self.convert_predictions_to_spans(entity_pred, entity_max_span_size)
        event_spans, event_n_spans = self.convert_predictions_to_spans(event_pred, event_max_span_size)
        
        target_spans = [ens + evs for ens, evs in zip(entity_spans, event_spans)]
        target_max_span_size = max(entity_max_span_size, event_max_span_size)
        target_n_spans = entity_n_spans + event_n_spans
        non_span_holder_n_tuples = target_n_spans.sum()
        span_holder_n_tuples = (holder_n_spans * target_n_spans).sum()

        if roles_only:
            return holder_loss + entity_loss + event_loss, holder_spans, target_spans, non_span_holder_n_tuples, span_holder_n_tuples

        non_span_holder_loss, span_holder_loss = None, None
        non_span_holder_attitude_predictions, span_holder_attitude_predictions = [[] for _ in range(len(target_spans))], [[] for _ in range(len(target_spans))]
        non_span_holder_spans, span_holder_spans = [[] for _ in range(len(target_spans))], [[] for _ in range(len(target_spans))]

        if label_tuples is not None:
            non_span_holder_label, span_holder_label = self.create_labels(label_tuples, holder_spans, target_spans, non_span_holder_n_tuples, span_holder_n_tuples, device)
            loss = nn.CrossEntropyLoss()

        if non_span_holder_n_tuples:

            non_span_holder_input, non_span_holder_mask, non_span_holder_target_relative_position_ids = self.create_non_span_holder_input(input_ids, mask, target_spans, non_span_holder_n_tuples, batch_seq_len, target_max_span_size, device)

            non_span_holder_contextual_token_embedding = self.non_span_holder_encoder(non_span_holder_input, non_span_holder_mask).last_hidden_state[:,1:batch_seq_len + 1]
            non_span_holder_target_relative_position_embedding = self.target_relative_position_embedding(non_span_holder_target_relative_position_ids)
            non_span_holder_token_and_position_embedding = torch.cat((non_span_holder_contextual_token_embedding, non_span_holder_target_relative_position_embedding), dim=2)
            non_span_holder_bilstm_output = self.non_span_holder_bilstm(non_span_holder_token_and_position_embedding)[0]
            non_span_holder_bilstm_first_and_last_hidden_state = torch.cat((non_span_holder_bilstm_output[:, -1, :self.hidden_size], non_span_holder_bilstm_output[:, 0, self.hidden_size:]), dim=1)
            non_span_holder_attitude_logits = self.non_span_holder_attitude_classifier(non_span_holder_bilstm_first_and_last_hidden_state)
            non_span_holder_attitude_pred = torch.argmax(non_span_holder_attitude_logits, dim=1).detach().cpu().numpy()
            i = 0
            for j, tspans in enumerate(target_spans):
                for tspan in tspans:
                    p = non_span_holder_attitude_pred[i]
                    if p:
                        non_span_holder_attitude_predictions[j].append(p)
                        non_span_holder_spans[j].append(tspan)
                    i += 1

            if label_tuples is not None:
                non_span_holder_loss = loss(non_span_holder_attitude_logits, non_span_holder_label)
            
        if span_holder_n_tuples:

            span_holder_input, span_holder_mask, span_holder_holder_relative_position_ids, span_holder_target_relative_position_ids = self.create_span_holder_input(input_ids, mask, holder_spans, target_spans, span_holder_n_tuples, batch_seq_len, holder_max_span_size, target_max_span_size, device)

            span_holder_contextual_token_embedding = self.span_holder_encoder(span_holder_input, span_holder_mask).last_hidden_state[:,1:batch_seq_len + 1]
            span_holder_holder_relative_position_embedding = self.holder_relative_position_embedding(span_holder_holder_relative_position_ids)
            span_holder_target_relative_position_embedding = self.target_relative_position_embedding(span_holder_target_relative_position_ids)
            span_holder_token_and_position_embedding = torch.cat((span_holder_contextual_token_embedding, span_holder_holder_relative_position_embedding, span_holder_target_relative_position_embedding), dim=2)
            span_holder_bilstm_output = self.span_holder_bilstm(span_holder_token_and_position_embedding)[0]
            span_holder_bilstm_first_and_last_hidden_state = torch.cat((span_holder_bilstm_output[:, -1, :self.hidden_size], span_holder_bilstm_output[:, 0, self.hidden_size:]), dim=1)
            span_holder_attitude_logits = self.span_holder_attitude_classifier(span_holder_bilstm_first_and_last_hidden_state)
            span_holder_attitude_pred = torch.argmax(span_holder_attitude_logits, dim=1)
            i = 0
            for j, (hspans, tspans) in enumerate(zip(holder_spans, target_spans)):
                for hspan in hspans:
                    for tspan in tspans:
                        p = span_holder_attitude_pred[i]
                        if p:
                            span_holder_attitude_predictions[j].append(p)
                            span_holder_spans[j].append(hspan + tspan)
                        i += 1

            if label_tuples is not None:
                span_holder_loss = loss(span_holder_attitude_logits, span_holder_label)

        total_loss = 0
        if holder_labels is not None:
            total_loss += holder_loss
        if entity_labels is not None:
            total_loss += entity_loss
        if event_labels is not None:
            total_loss += event_loss
        if label_tuples is not None:
            if non_span_holder_n_tuples:
                total_loss += non_span_holder_loss
            if span_holder_n_tuples:
                total_loss += span_holder_loss

        return total_loss, non_span_holder_attitude_predictions, span_holder_attitude_predictions, non_span_holder_spans, span_holder_spans
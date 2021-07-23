import torch
from torch import nn
from transformers import BertModel
import config

class OpinionMiner(nn.Module):

    def __init__(self):
        super().__init__()
        self.encoder = BertModel.from_pretrained(config.pretrained_model_name)
        
        self.max_holder_span_size = config.max_holder_span_size
        self.max_target_span_size = config.max_target_span_size
        self.max_span_size = max(self.max_holder_span_size, self.max_target_span_size)
        self.max_n_holders_per_sentence = config.max_n_holders_per_sentence
        self.max_n_targets_per_sentence = config.max_n_targets_per_sentence

        self.hidden_size = self.encoder.config.hidden_size
        self.conv1ds = []
        for i in range(self.max_span_size - 1):
            conv1d = nn.Conv1d(self.hidden_size, self.hidden_size, i + 2, bias=False)
            conv1d.weight.data = torch.zeros((self.hidden_size, self.hidden_size, i + 2), device=config.device)
            for j in range(self.hidden_size):
                conv1d.weight.data[j, j] = torch.ones(i + 2)/(i + 2)
            conv1d.weight.requires_grad = False
            self.conv1ds.append(conv1d)
        
        self.holder_scorer = nn.Linear(self.hidden_size, 1)
        self.target_scorer = nn.Linear(self.hidden_size, 1)
        self.implicit_opinion_classifier = nn.Linear(self.hidden_size, 4)
        self.explicit_opinion_classifier = nn.Linear(2 * self.hidden_size, 4)


    def forward(self, token_ids, mask, label_tuples=None):
        device = next(self.parameters()).device
        embedding = self.encoder(token_ids, mask).last_hidden_state
        batch_size, seq_len, hidden_size = embedding.shape
        input = embedding.permute(0, 2, 1)

        span_representations_list = [embedding]
        for i in range(self.max_span_size - 1):
            output = self.conv1ds[i](input)
            output = torch.cat((output, torch.zeros((batch_size, hidden_size, i + 1), device=device)), dim=2)
            span_representations_list.append(output.permute(0, 2, 1))

        span_representations = torch.cat(span_representations_list, dim=1)
        length = torch.sum(mask, dim=1, dtype=int)
        holder_span_mask = torch.zeros((batch_size, seq_len * self.max_span_size), dtype=int, device=device)
        target_span_mask = torch.zeros((batch_size, seq_len * self.max_span_size), dtype=int, device=device)

        for i in range(batch_size):
            for j in range(self.max_holder_span_size):
                holder_span_mask[i, seq_len * j: seq_len * j + length[i] - j] = 1
            for j in range(self.max_target_span_size):
                target_span_mask[i, seq_len * j: seq_len * j + length[i] - j] = 1
        
        all_target_scores = torch.sigmoid(self.target_scorer(span_representations).squeeze())
        all_holder_scores = torch.sigmoid(self.holder_scorer(span_representations).squeeze())

        holder_scores = all_holder_scores.clone()
        target_scores = all_target_scores.clone()
        holder_scores[holder_span_mask == 0] = -1
        target_scores[target_span_mask == 0] = -1

        holder_ranks = torch.argsort(holder_scores, dim=1, descending=True).cpu().numpy()
        target_ranks = torch.argsort(target_scores, dim=1, descending=True).cpu().numpy()

        implicit_input = torch.zeros((batch_size, self.max_n_targets_per_sentence, hidden_size), device=device)
        implicit_mask = torch.zeros((batch_size, self.max_n_targets_per_sentence), dtype=int, device=device)
        explicit_input = torch.zeros((batch_size, self.max_n_holders_per_sentence * self.max_n_targets_per_sentence, 2*hidden_size), device=device)
        explicit_mask = torch.zeros((batch_size, self.max_n_holders_per_sentence * self.max_n_targets_per_sentence), dtype=int, device=device)

        for i in range(batch_size):
            
            for j in range(self.max_n_targets_per_sentence):
                r = target_ranks[i, j]
                if target_scores[i, r] > 0:
                    implicit_input[i, j] = span_representations[i, r]
                    implicit_mask[i, j] = 1
            
            for j in range(self.max_n_holders_per_sentence):
                p = holder_ranks[i, j]
                if holder_scores[i, p] > 0:
                    for k in range(self.max_n_targets_per_sentence):
                        q = target_ranks[i, k]
                        if target_scores[i, q] > 0:
                            explicit_input[i, j * self.max_n_targets_per_sentence + k] = torch.cat((span_representations[i, p], span_representations[i, q]))
                            explicit_mask[i, j * self.max_n_targets_per_sentence + k] = 1
        
        implicit_logits = self.implicit_opinion_classifier(implicit_input)
        explicit_logits = self.explicit_opinion_classifier(explicit_input)
        implicit_pred = implicit_logits.argmax(dim=2)
        explicit_pred = explicit_logits.argmax(dim=2)

        pred_tuples = []
        for i in range(batch_size):
            target_to_label = {}
            holder_and_target_to_label = {}
            
            for j in range(self.max_n_targets_per_sentence):
                r = target_ranks[i, j]
                if target_scores[i, r] > 0:
                    target_span_size, target_start = divmod(r, seq_len)
                    target_end = target_start + target_span_size + 1
                    target_key = (target_start, target_end)
                    target_to_label[target_key] = implicit_pred[i, j].item()
            
            for j in range(self.max_n_holders_per_sentence):
                p = holder_ranks[i, j]
                if holder_scores[i, p] > 0:
                    holder_span_size, holder_start = divmod(p, seq_len)
                    holder_end = holder_start + holder_span_size + 1
                    for k in range(self.max_n_targets_per_sentence):
                        q = target_ranks[i, k]
                        if target_scores[i, q] > 0:
                            target_span_size, target_start = divmod(q, seq_len)
                            target_end = target_start + target_span_size + 1
                            key = (holder_start, holder_end, target_start, target_end)
                            holder_and_target_to_label[key] = explicit_pred[i, j * self.max_n_targets_per_sentence + k].item()
            
            pred_tuples.append(dict(target_to_label=target_to_label, holder_and_target_to_label=holder_and_target_to_label))

        if label_tuples is not None:
            holder_label = torch.zeros((batch_size, seq_len * self.max_span_size), device=device)
            target_label = torch.zeros((batch_size, seq_len * self.max_span_size), device=device)
            
            for i in range(batch_size):
                
                for target_span in label_tuples[i]["target_to_label"]:
                    start, end = target_span
                    width = end - start
                    if width <= self.max_span_size:
                        target_label[i, seq_len * (width - 1) + start] = 1
                
                for holder_and_target_span in label_tuples[i]["holder_and_target_to_label"]:
                    holder_start, holder_end = holder_and_target_span[:2]
                    target_start, target_end = holder_and_target_span[2:]
                    holder_width = holder_end - holder_start
                    target_width = target_end - target_start
                    if holder_width <= self.max_span_size:
                        holder_label[i, seq_len * (holder_width - 1) + holder_start] = 1
                    if target_width <= self.max_span_size:
                        target_label[i, seq_len * (target_width - 1) + target_start] = 1

            mse_loss = nn.MSELoss()
            active_holder_scores = holder_scores[holder_span_mask == 1]
            active_holder_label = holder_label[holder_span_mask == 1]
            active_target_scores = target_scores[target_span_mask == 1]
            active_target_label = target_label[target_span_mask == 1]

            holder_loss = mse_loss(active_holder_scores, active_holder_label)
            target_loss = mse_loss(active_target_scores, active_target_label)

            implicit_label = torch.zeros((batch_size, self.max_n_targets_per_sentence), dtype=int, device=device)
            explicit_label = torch.zeros((batch_size, self.max_n_holders_per_sentence * self.max_n_targets_per_sentence), dtype=int, device=device)

            for i in range(batch_size):
                
                for j in range(self.max_n_targets_per_sentence):
                    r = target_ranks[i, j]
                    target_span_size, target_start = divmod(r, seq_len)
                    target_end = target_start + target_span_size + 1
                    target_key = (target_start, target_end)
                    
                    if target_key in label_tuples[i]["target_to_label"]:
                        label = label_tuples[i]["target_to_label"][target_key]
                        implicit_label[i, j] = label
                
                for j in range(self.max_n_holders_per_sentence):
                    p = holder_ranks[i, j]
                    holder_span_size, holder_start = divmod(p, seq_len)
                    holder_end = holder_start + holder_span_size + 1
                    
                    for k in range(self.max_n_targets_per_sentence):
                        q = target_ranks[i, k]
                        target_span_size, target_start = divmod(q, seq_len)
                        target_end = target_start + target_span_size + 1
                        key = (holder_start, holder_end, target_start, target_end)

                        if key in label_tuples[i]["holder_and_target_to_label"]:
                            label = label_tuples[i]["holder_and_target_to_label"][key]
                            explicit_label[i, j * self.max_n_targets_per_sentence + k] = label
            
            cross_entropy_loss = nn.CrossEntropyLoss()
            implicit_label[implicit_mask == 0] = cross_entropy_loss.ignore_index
            explicit_label[explicit_mask == 0] = cross_entropy_loss.ignore_index
            implicit_loss = cross_entropy_loss(implicit_logits.view(-1, 4), implicit_label.view(-1))
            explicit_loss = cross_entropy_loss(explicit_logits.view(-1, 4), explicit_label.view(-1))

            loss = holder_loss + target_loss + implicit_loss + explicit_loss
            return loss, pred_tuples
        
        else:
            return pred_tuples
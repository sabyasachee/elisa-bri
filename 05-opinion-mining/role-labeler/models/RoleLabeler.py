# author : Sabyasachee

import numpy as np
import torch
from torch import nn
from transformers import BertModel

class RoleScorer(nn.Module):

    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
    
    def forward(self, X):
        # X is tensor of shape batch-size x input-size

        A = self.fc1(X)
        A = torch.relu(A)
        # A is of shape batch-size x hidden-size

        B = self.fc2(A)
        # B is of shape batch-size x 1

        return B

class RoleLabeler(nn.Module):

    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams
        self.encoder = BertModel.from_pretrained(hparams.pretrained_model_name)
        
        enc_hidden_size = self.encoder.config.hidden_size
        self.holder_scorer = RoleScorer(2 * enc_hidden_size, hparams.scorer_hidden_size, 1)
        self.target_scorer = RoleScorer(2 * enc_hidden_size, hparams.scorer_hidden_size, 1)
        self.opinion_scorer = RoleScorer(4 * enc_hidden_size, hparams.scorer_hidden_size, 3)
    
    def forward(self, X_wordid, X_mask, Y=None, train=False, debug=False, use_gold_holders_and_targets=False):
        # X_wordid is the batch-size x max-seq-len matrix of WordPiece token indices
        # X_wordid[:,0] is always the [AUTHOR] token
        # X_mask is the batch-size x max-seq-len matrix of 0s and 1s.
        # X_mask = 0 for padded tokens, otherwise 1
        # 
        # Y is batch-size x 6 matrix
        # Y[i, 0] is the dataset sentence id
        # (Y[i, 1], Y[i, 2]) is the holder expression span
        # (Y[i, 3], Y[i, 4]) is the target expression span
        # Y[i, 5] is the opinion label: 0 = +ve sentiment, 1 = -ve sentiment, 2 = other
        # 
        # IMPORTANT: THE FIRST TOKEN IS THE [AUTHOR] TOKEN
        # 
        # if train is True, model is trained, else model predicts
        # if train is True:
        #   Y cannot be None
        #   model return loss  
        # 
        # if train is False:
        #   if use_gold_holders_and_targets is true, Y cannot be None
        # 
        #   if debug and use_gold_holders_and_targets is false:
        #       model returns Ypred
        #       Ypred is * x 6 matrix
        #       where each row y represents a attitude frame prediction
        #       y[0] is the batch sentence id. it lies between 0 and batch-size = X_wordid.shape[0]
        #       y[1,2] is the holder span
        #       y[3,4] is the target span
        #       y[5] is the opinion label, 0, 1, or 2
        # 
        #   if debug is True:
        #       model returns Ypred, Spred
        #       Ypred is as defined above
        #       Spred is * x 5 matrix
        #       where each row s represents an incomplete attitude frame prediction
        #       it consists of either only the holder or the target
        #       s[0] is the batch sentence id
        #       s[1] is 0 or 1, 0 for holder, 1 for target
        #       s[2,3] is the holder/target span
        #       s[4] is the holder/target score
        #       s is sorted by (s[0], s[1], s[4])
    
        #   if use_gold_holders_and_targets is True:
        #       Y cannot be None
        #       model returns Ypred
        #       Ypred is as defined above
        #       Ypred[:,0,1,2,3,4] is same as Y
        #       or the model uses the gold holders and targets for opinion representations
        #       Notice that now Ypred[:,0] is the dataset sentence id
        # 
        #   use_gold_holders_and_targets has precedence over debug

        device = self.hparams.device

        assert not (train or use_gold_holders_and_targets) or Y is not None

        X_word = self.encoder(X_wordid, X_mask).last_hidden_state
        # X_word is of shape batch-size x max-seq-len x enc-hidden-size

        if train:

            holder_representations = []
            target_representations = []
            # holder representations will be a list of span representation of holders
            # similarly, for target representations 

            for i in range(X_word.shape[0]):

                holder_representation = torch.cat([ X_word[i, Y[i, 1]], X_word[i, Y[i, 2] - 1] ])
                target_representation = torch.cat([ X_word[i, Y[i, 3]], X_word[i, Y[i, 4] - 1] ])
                # holder (target) representation is of shape 2*enc-hidden-size

                holder_representations.append(holder_representation)
                target_representations.append(target_representation)
            
            holder_representations = torch.vstack(holder_representations)
            target_representations = torch.vstack(target_representations)
            opinion_representations = torch.hstack([holder_representations, target_representations])
            # holder-representations is of shape batch-size x 2*enc-hidden-size, same for target
            # opinion-representations is of shape batch-size x 4*enc-hidden-size

            holder_scores = self.holder_scorer(holder_representations)
            target_scores = self.target_scorer(target_representations)
            opinion_scores = self.opinion_scorer(opinion_representations)
            # holder-scores is of shape batch-size x 1, same for target-scores
            # opinion-scores is of shape batch-size x 3

            opinion_scores = opinion_scores + holder_scores + target_scores
            non_opinion_scores = torch.zeros((X_word.shape[0], 1), device=device, requires_grad=True)
            scores = torch.hstack([opinion_scores, non_opinion_scores])
            # opinion-scores is of shape batch-size x 4

            # loss_function = nn.CrossEntropyLoss()
            # loss = loss_function(opinion_scores, Y[:,-1])
            logits = torch.log_softmax(scores, dim=1)
            for i, y in enumerate(Y[:,-1]):
                logits[i, y] = -logits[i, y]
            loss = torch.mean(logits)

            return loss
        
        else:

            if use_gold_holders_and_targets:

                Ypred = []
                holder_representations = []
                target_representations = []
                sentence_ids = []

                for i, y in enumerate(Y):
                    if y[5] != -1:
                        holder_representations.append(torch.cat([ X_word[i, y[1]], X_word[i, y[2] - 1] ]))
                        target_representations.append(torch.cat([ X_word[i, y[3]], X_word[i, y[4] - 1] ]))
                        sentence_ids.append(i)

                holder_representations = torch.vstack(holder_representations)
                target_representations = torch.vstack(target_representations)

                holder_scores = self.holder_scorer(holder_representations)
                target_scores = self.target_scorer(target_representations)

                scores = holder_scores + target_scores

                opinion_representations = torch.hstack([holder_representations, target_representations])
                opinion_scores = self.opinion_scorer(opinion_representations)
                opinion_scores += scores

                for i, score in enumerate(opinion_scores):
                    j = sentence_ids[i]
                    max_score = torch.max(score)
                    if max_score > 0:
                        opinion_label = torch.argmax(score).item()
                        Ypred.append(Y[j, :5].tolist() + [opinion_label])
                
                return Ypred

            else:

                holder_representations = []
                target_representations = []
                holder_pos = []
                target_pos = []
                holder_sentence_ids = []
                target_sentence_ids = []
                # holder-representations will be a list of span representation of holders
                # similarly for target-representations
                # holder-pos will be a list of list of 2 integers, defining the holder-span
                # similarly for target-pos
                # holder-sentence-ids will be a list of integers linking the holder representation to its sentence
                # target-sentence-ids will be a list of integers linking the target representation to its sentence

                for i in range(X_word.shape[0]):

                    sentence_length = int(X_mask[i].sum())

                    for span_length in range(1, self.hparams.max_holder_length + 1):

                        if span_length == 1:
                            sentence_start = 0
                        else:
                            sentence_start = 1

                        for span_start in range(sentence_start, sentence_length - span_length + 1):
                            
                            span_end = span_start + span_length
                            holder_representation = torch.cat([ X_word[i, span_start], X_word[i, span_end - 1] ])
                            holder_representations.append(holder_representation)
                            holder_pos.append([span_start, span_end])
                            holder_sentence_ids.append(i)
                
                for i in range(X_word.shape[0]):

                    sentence_length = int(X_mask[i].sum())

                    for span_length in range(1, self.hparams.max_target_length + 1):

                        for span_start in range(1, sentence_length - span_length + 1):

                            span_end = span_start + span_length
                            target_representation = torch.cat([ X_word[i, span_start], X_word[i, span_end - 1] ])
                            target_representations.append(target_representation)
                            target_pos.append([span_start, span_end])
                            target_sentence_ids.append(i)
                
                holder_representations = torch.vstack(holder_representations)
                target_representations = torch.vstack(target_representations)
                # holder-representations is of shape total-number-holders x 2*enc-hidden-size
                # target-representations is of shape total-number-targets x 2*enc-hidden-size
                # holder-sentence-ids, holder-pos is a list of size total-number-holders
                # target-sentence-ids, target-pos is a list of size total-number-targets

                holder_scores = self.holder_scorer(holder_representations)
                target_scores = self.target_scorer(target_representations)
                # holder-scores is of shape total-number-holders x 1
                # target-scores is of shape total-number-targets x 1

                holder_pos = np.array(holder_pos)
                target_pos = np.array(target_pos)
                holder_sentence_ids = np.array(holder_sentence_ids)
                target_sentence_ids = np.array(target_sentence_ids)
                # holder-pos is numpy int array of shape total-number-holders x 2
                # target-pos is numpy int array of shape total-number-targets x 2
                # holder-sentence-ids is numpy int array of shape total-number-holders
                # target-sentence-ids is numpy int array of shape total-number-targets
                
                opinion_representations = []
                sentence_ids = []
                scores = []
                pos = []

                if debug:
                    Spred = []

                for i in range(X_word.shape[0]):

                    sentence_holder_representations = holder_representations[holder_sentence_ids == i]
                    sentence_target_representations = target_representations[target_sentence_ids == i]
                    # sentence-holder-representations contains holder representations of sentence i
                    # similarly sentence-target-representations
                    # sentence-holder-representations is of shape sentence-number-holders x 2*enc-hidden-size
                    # sentence-target-representations is of shape sentence-number-targets x 2*enc-hidden-size

                    sentence_holder_scores = holder_scores[holder_sentence_ids == i].flatten()
                    sentence_target_scores = target_scores[target_sentence_ids == i].flatten()
                    # sentence-holder-scores is of shape sentence-number-holders
                    # sentence-target-scores is of shape sentence-number-targets

                    sentence_holder_pos = holder_pos[holder_sentence_ids == i]
                    sentence_target_pos = target_pos[target_sentence_ids == i]
                    # sentence-holder-pos is of shape sentence-number-holders x 2
                    # sentence-target-pos is of shape sentence-number-targets x 2

                    sentence_holder_scores_sort_index = torch.argsort(sentence_holder_scores, descending=True).cpu().numpy()
                    sentence_target_scores_sort_index = torch.argsort(sentence_target_scores, descending=True).cpu().numpy()
                    # sentence-holder-scores-sort-index is of shape sentence-number-holders
                    # it contains holder indices in descending order of holder scores
                    # similarly, sentence-target-scores-sort-index is of shape sentence-number-targets
                    # it contains target indices in descending order of target scores

                    if len(sentence_holder_scores) > 0 and len(sentence_target_scores) > 0:
                        # sentence has some holder and some target spans

                        sentence_top_holder_indices = sentence_holder_scores_sort_index[: self.hparams.n_holders]
                        sentence_top_target_indices = sentence_target_scores_sort_index[: self.hparams.n_targets]
                        
                        sentence_top_holder_representations = sentence_holder_representations[sentence_top_holder_indices]
                        sentence_top_target_representations = sentence_target_representations[sentence_top_target_indices]
                        sentence_top_holder_scores = sentence_holder_scores[sentence_top_holder_indices]
                        sentence_top_target_scores = sentence_target_scores[sentence_top_target_indices]
                        sentence_top_holder_pos = sentence_holder_pos[sentence_top_holder_indices]
                        sentence_top_target_pos = sentence_target_pos[sentence_top_target_indices]
                        # sentence-top-holder-representations is of shape sentence-number-top-holders x 2*enc-hidden-size
                        # sentence-top-holder-scores is of shape sentence-number-top-holders
                        # sentence-top-holder-pos is of shape sentence-number-top-holders x 2
                        # similarly for target (sentence-number-top-targets)
                        # sentence-number-top-holders <= hparams.n_holders, sentence-number-top-targets <= hparams.n_targets

                        if debug:
                            sentence_sorted_holder_pos = sentence_holder_pos[sentence_holder_scores_sort_index]
                            sentence_sorted_target_pos = sentence_target_pos[sentence_target_scores_sort_index]
                            sentence_sorted_holder_scores = sentence_holder_scores[sentence_holder_scores_sort_index]
                            sentence_sorted_target_scores = sentence_target_scores[sentence_target_scores_sort_index]

                            for u, v in zip(sentence_sorted_holder_pos, sentence_sorted_holder_scores):
                                Spred.append([i, 0] + list(u) + [v.item()])
                            
                            for u, v in zip(sentence_sorted_target_pos, sentence_sorted_target_scores):
                                Spred.append([i, 1] + list(u) + [v.item()])

                        for hi in range(sentence_top_holder_representations.shape[0]):
                            for ti in range(sentence_top_target_representations.shape[0]):

                                opinion_representations.append(torch.hstack([ sentence_top_holder_representations[hi], sentence_top_target_representations[ti] ]))
                                scores.append(sentence_top_holder_scores[hi] + sentence_top_target_scores[ti])
                                pos.append(sentence_top_holder_pos[hi].tolist() + sentence_top_target_pos[ti].tolist())
                                sentence_ids.append(i)
                
                opinion_representations = torch.vstack(opinion_representations)
                opinion_scores = self.opinion_scorer(opinion_representations)
                # opinion-representations is of shape num-opinions x 4*enc-hidden-size
                # opinion-scores is of shape num-opinions x 3

                scores = torch.Tensor(scores).reshape(-1, 1).to(device)
                # scores is of shape num-opinions x 1

                scores = scores + opinion_scores
                # scores is of shape num-opinions x 3

                Ypred = []

                for i in range(scores.shape[0]):

                    max_score = torch.max(scores[i])

                    if max_score > 0:

                        opinion_label = torch.argmax(scores[i]).item()
                        Ypred.append([sentence_ids[i]] + pos[i] + [opinion_label])
                
                if debug:
                    return Ypred, Spred

                return Ypred
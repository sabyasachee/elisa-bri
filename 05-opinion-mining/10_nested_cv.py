from collections import defaultdict
import json
import numpy as np
import config
import pandas as pd
import os
from tqdm import tqdm, trange
from transformers import BertTokenizer
import torch

def tokenize_and_preserve_labels(tokens, tokenizer):
    cumul = np.zeros(len(tokens) + 1, dtype=int)
    ttokens = []
    for i, token in enumerate(tokens):
        tt = tokenizer.tokenize(token)
        ttokens.extend(tt)
        cumul[i + 1] = cumul[i] + len(tt)
    return ttokens, cumul

def correct_span(span, cumul):
    return [cumul[span[0]], cumul[span[1] + 1]]

def correct_label_arr(label_arr, cumul):
    new_label_arr = np.full(cumul[-1], "O")
    for i in range(len(label_arr)):
        if label_arr[i] == "B":
            new_label_arr[cumul[i]] = "B"
            new_label_arr[cumul[i] + 1: cumul[i + 1]] = "I"
        elif label_arr[i] == "I":
            new_label_arr[cumul[i]: cumul[i + 1]] = "I"
    return new_label_arr.tolist()

def encode_attitude(attitude_type):
    if attitude_type == "sentiment-pos":
        return 1
    elif attitude_type == "sentiment-neg":
        return 2
    else:
        return 3

def create_label_dict(label_tuples, cumul):
    label_dict = dict(target_to_label={}, holder_and_target_to_label={})
    holder_sizes, target_entity_sizes, target_event_sizes = [], [], []
    implicit_attitude, explicit_attitude = np.zeros(3, dtype=int), np.zeros(3, dtype=int)
    
    for ex in label_tuples:
        if ex["target-type"] in ["entity","event"]:
            target_span = correct_span(ex["target"], cumul)
            attitude_label = encode_attitude(ex["attitude-type"])
            
            if ex["target-type"] == "entity":
                target_entity_sizes.append(target_span[1] - target_span[0])
            else:
                target_event_sizes.append(target_span[1] - target_span[0])
            
            if ex["holder-type"] == "span":
                holder_span = correct_span(ex["holder"], cumul)
                holder_sizes.append(holder_span[1] - holder_span[0])
                key = tuple(holder_span + target_span)
                if key not in label_dict["holder_and_target_to_label"] or label_dict["holder_and_target_to_label"][key] > attitude_label:
                    label_dict["holder_and_target_to_label"][key] = attitude_label
                    explicit_attitude[attitude_label - 1] += 1
            
            else:
                key = tuple(target_span)
                if key not in label_dict["target_to_label"] or label_dict["target_to_label"][key] > attitude_label:
                    label_dict["target_to_label"][tuple(target_span)] = attitude_label
                    implicit_attitude[attitude_label - 1] += 1
    
    return label_dict, holder_sizes, target_entity_sizes, target_event_sizes, implicit_attitude, explicit_attitude

def cross_validation():
    fold_file = os.path.join(config.RESULTS_FOLDER, "mpqa3/5fold.csv")
    fold_df = pd.read_csv(fold_file, index_col=None)
    n_folds = fold_df["fold"].max() + 1
    max_seq_len = -1000
    label_to_label_id = {"O":0, "B":1, "I":2}
    holder_sizes, entity_sizes, event_sizes = [], [], []
    implicit_attitude, explicit_attitude = np.zeros(3, dtype=int), np.zeros(3, dtype=int)
    n_holders_per_sentence, n_targets_per_sentence = [], []

    fold_tokens = [[] for _ in range(n_folds)]
    fold_lens = [[] for _ in range(n_folds)]
    fold_opinion_labels = [[] for _ in range(n_folds)]
    fold_holder_labels = [[] for _ in range(n_folds)]
    fold_entity_labels = [[] for _ in range(n_folds)]
    fold_event_labels = [[] for _ in range(n_folds)]
    fold_label_tuples = [[] for _ in range(n_folds)]
    fold_token_indices = [None for _ in range(n_folds)]
    fold_masks = [None for _ in range(n_folds)]
    fold_opinion_label_indices = [None for _ in range(n_folds)]
    fold_holder_label_indices = [None for _ in range(n_folds)]
    fold_entity_label_indices = [None for _ in range(n_folds)]
    fold_event_label_indices = [None for _ in range(n_folds)]

    print("loading {} Tokenizer".format(config.pretrained_model_name))
    tokenizer = BertTokenizer.from_pretrained(config.pretrained_model_name)

    special_token_to_indices = {}
    special_token_to_indices["CLS"] = tokenizer.vocab["[CLS]"]
    special_token_to_indices["SEP"] = tokenizer.vocab["[SEP]"]
    special_token_to_indices["PAD"] = tokenizer.vocab["[PAD]"]
    special_token_to_indices["HOLDER"] = tokenizer.vocab["[HOLDER]"] = tokenizer.vocab.pop("[unused1]")
    special_token_to_indices["TARGET"] = tokenizer.vocab["[TARGET]"] = tokenizer.vocab.pop("[unused2]")
    special_token_to_indices["OPINION"] = tokenizer.vocab["[OPINION]"] = tokenizer.vocab.pop("[unused3]")

    for _, row in tqdm(fold_df.iterrows(), total=len(fold_df), desc="reading mpqa3"):
        doc_file = os.path.join(config.MPQA3_PROCESSED_FOLDER, row["doc_id"], "tokenized.json")
        i = row["fold"]
        doc = json.load(open(doc_file))
        
        for sentence in doc:
            tokens, cumul = tokenize_and_preserve_labels(sentence["tokens"], tokenizer)
            opinion_labels = correct_label_arr(sentence["dse-opinion"], cumul)
            holder_labels = correct_label_arr(sentence["dse-holder"], cumul)
            entity_labels = correct_label_arr(sentence["dse-entity"], cumul)
            event_labels = correct_label_arr(sentence["dse-event"], cumul)
            
            label_dict, sentence_holder_sizes, sentence_entity_sizes, sentence_event_sizes, sentence_implicit_attitude, sentence_explicit_attitude = create_label_dict(sentence["dse"], cumul)
            
            fold_tokens[i].append(tokens)
            fold_lens[i].append(len(tokens))
            fold_opinion_labels[i].append(opinion_labels)
            fold_holder_labels[i].append(holder_labels)
            fold_entity_labels[i].append(entity_labels)
            fold_event_labels[i].append(event_labels)
            fold_label_tuples[i].append(label_dict)
            
            max_seq_len = max(max_seq_len, len(tokens))
            holder_sizes.extend(sentence_holder_sizes)
            entity_sizes.extend(sentence_entity_sizes)
            event_sizes.extend(sentence_event_sizes)
            implicit_attitude += sentence_implicit_attitude
            explicit_attitude += sentence_explicit_attitude

            target_spans = set()
            holder_spans = set()
            for target_span in label_dict["target_to_label"]:
                target_spans.add(target_span)
            for holder_and_target_span in label_dict["holder_and_target_to_label"]:
                holder_span = tuple(holder_and_target_span[:2])
                target_span = tuple(holder_and_target_span[2:])
                target_spans.add(target_span)
                holder_spans.add(holder_span)
            n_holders_per_sentence.append(len(holder_spans))
            n_targets_per_sentence.append(len(target_spans))

    print("max seq length       = {}".format(max_seq_len))
    print("holder span size     : 90%tile = {:.1f}, 95%tile = {:.1f}, 99%tile = {:.1f}, max = {:.1f}".format( np.quantile(holder_sizes, 0.9), np.quantile(holder_sizes, 0.95), np.quantile(holder_sizes, 0.99), np.max(holder_sizes) ))
    print("entity span size     : 90%tile = {:.1f}, 95%tile = {:.1f}, 99%tile = {:.1f}, max = {:.1f}".format( np.quantile(entity_sizes, 0.9), np.quantile(entity_sizes, 0.95), np.quantile(entity_sizes, 0.99), np.max(entity_sizes) ))
    print("event span size      : 90%tile = {:.1f}, 95%tile = {:.1f}, 99%tile = {:.1f}, max = {:.1f}".format( np.quantile(event_sizes, 0.9), np.quantile(event_sizes, 0.95), np.quantile(event_sizes, 0.99), np.max(event_sizes) ))
    print("num holders per sent : 90%tile = {:.1f}, 95%tile = {:.1f}, 99%tile = {:.1f}, max = {:.1f}".format( np.quantile(n_holders_per_sentence, 0.9), np.quantile(n_holders_per_sentence, 0.95), np.quantile(n_holders_per_sentence, 0.99), np.max(n_holders_per_sentence) ))
    print("num targets per sent : 90%tile = {:.1f}, 95%tile = {:.1f}, 99%tile = {:.1f}, max = {:.1f}".format( np.quantile(n_targets_per_sentence, 0.9), np.quantile(n_targets_per_sentence, 0.95), np.quantile(n_targets_per_sentence, 0.99), np.max(n_targets_per_sentence) ))
    print("implicit attitude    : pos = {:4d}, neg = {:4d}, other = {:4d}, total = {:4d}".format(implicit_attitude[0], implicit_attitude[1], implicit_attitude[2], implicit_attitude.sum()))
    print("explicit attitude    : pos = {:4d}, neg = {:4d}, other = {:4d}, total = {:4d}".format(explicit_attitude[0], explicit_attitude[1], explicit_attitude[2], explicit_attitude.sum()))
    print("total                : pos = {:4d}, neg = {:4d}, other = {:4d}, total = {:4d}".format(implicit_attitude[0] + explicit_attitude[0], implicit_attitude[1] + explicit_attitude[1], implicit_attitude[2] + explicit_attitude[2], implicit_attitude.sum() + explicit_attitude.sum()))

    for i in trange(n_folds, desc="creating tensors"):
        index = np.argsort(fold_lens[i])
        fold_tokens[i] = [fold_tokens[i][j] for j in index]
        fold_lens[i] = sorted(fold_lens[i])
        fold_opinion_labels[i] = [fold_opinion_labels[i][j] for j in index]
        fold_holder_labels[i] = [fold_holder_labels[i][j] for j in index]
        fold_entity_labels[i] = [fold_entity_labels[i][j] for j in index]
        fold_event_labels[i] = [fold_event_labels[i][j] for j in index]
        fold_label_tuples[i] = [fold_label_tuples[i][j] for j in index]
        fold_token_indices[i] = torch.zeros((len(fold_tokens[i]), max_seq_len), dtype=torch.long, device=config.device)
        fold_masks[i] = torch.zeros((len(fold_tokens[i]), max_seq_len), dtype=torch.long, device=config.device)
        fold_opinion_label_indices[i] = torch.zeros((len(fold_tokens[i]), max_seq_len), dtype=torch.long, device=config.device)
        fold_holder_label_indices[i] = torch.zeros((len(fold_tokens[i]), max_seq_len), dtype=torch.long, device=config.device)
        fold_entity_label_indices[i] = torch.zeros((len(fold_tokens[i]), max_seq_len), dtype=torch.long, device=config.device)
        fold_event_label_indices[i] = torch.zeros((len(fold_tokens[i]), max_seq_len), dtype=torch.long, device=config.device)
        for j in range(len(fold_tokens[i])):
            length = len(fold_tokens[i][j])
            fold_token_indices[i][j][:length] = torch.LongTensor(tokenizer.convert_tokens_to_ids(fold_tokens[i][j]))
            fold_masks[i][j][:length] = 1.
            fold_opinion_label_indices[i][j][:length] = torch.LongTensor([label_to_label_id[label] for label in fold_opinion_labels[i][j]])
            fold_holder_label_indices[i][j][:length] = torch.LongTensor([label_to_label_id[label] for label in fold_holder_labels[i][j]])
            fold_entity_label_indices[i][j][:length] = torch.LongTensor([label_to_label_id[label] for label in fold_entity_labels[i][j]])
            fold_event_label_indices[i][j][:length] = torch.LongTensor([label_to_label_id[label] for label in fold_event_labels[i][j]])

    # for i in range(n_folds):
    #     for j in range(n_folds):
    #         if i != j:
    #             print("test fold = {}, dev fold = {}".format(i + 1, j + 1))
    #             train(i, j, fold_token_indices, fold_masks, fold_opinion_label_indices, fold_holder_label_indices, fold_entity_label_indices, fold_event_label_indices, fold_label_tuples, special_token_to_indices)

if __name__ == "__main__":
    cross_validation()
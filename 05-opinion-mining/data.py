import json
import numpy as np
import config
import os
from tqdm import tqdm
from transformers import BertTokenizer
import torch

class OpinionDataset:

    def __init__(self, doc_ids, spacy=False):
        self.spacy = spacy
        self.tensorize(doc_ids)

    def tokenize_and_preserve_labels(self, tokens, tokenizer):
        cumul = np.zeros(len(tokens) + 1, dtype=int)
        ttokens = []
        for i, token in enumerate(tokens):
            tt = tokenizer.tokenize(token)
            ttokens.extend(tt)
            cumul[i + 1] = cumul[i] + len(tt)
        return ttokens, cumul

    def correct_span(self, span, cumul):
        return [cumul[span[0]], cumul[span[1] + 1]]

    def encode_attitude(self, attitude_type):
        if attitude_type == "sentiment-pos":
            return 1
        elif attitude_type == "sentiment-neg":
            return 2
        else:
            return 3

    def create_label_dict(self, label_tuples, cumul):
        label_dict = dict(target_to_label={}, holder_and_target_to_label={}, target_to_target_type={})
        holder_sizes, target_entity_sizes, target_event_sizes, target_span_sizes = [], [], [], []
        implicit_attitude, explicit_attitude = np.zeros(3, dtype=int), np.zeros(3, dtype=int)
        
        for ex in label_tuples:
            target_span = self.correct_span(ex["target"], cumul)
            attitude_label = self.encode_attitude(ex["attitude-type"])
            
            if ex["target-type"] == "entity":
                target_entity_sizes.append(target_span[1] - target_span[0])
            elif ex["target-type"] == "event":
                target_event_sizes.append(target_span[1] - target_span[0])
            else:
                target_span_sizes.append(target_span[1] - target_span[0])

            label_dict["target_to_target_type"][tuple(target_span)] = ex["target-type"]
            
            if ex["holder-type"] == "span":
                holder_span = self.correct_span(ex["holder"], cumul)
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
        
        return label_dict, holder_sizes, target_entity_sizes, target_event_sizes, target_span_sizes, implicit_attitude, explicit_attitude

    def tensorize(self, doc_ids):
        seq_lens, holder_sizes, entity_sizes, event_sizes, span_sizes = [], [], [], [], []
        implicit_attitude, explicit_attitude = np.zeros(3, dtype=int), np.zeros(3, dtype=int)
        n_holders_per_sentence, n_targets_per_sentence = [], []

        tokens, lengths, label_tuples, sentence_ids = [], [], [], []

        print("loading {} Tokenizer".format(config.pretrained_model_name))
        tokenizer = BertTokenizer.from_pretrained(config.pretrained_model_name)

        for doc_id in tqdm(doc_ids, desc="tokenizing"):
            doc_file = os.path.join(config.MPQA_PROCESSED_FOLDER, doc_id, "tokenized.json")
            doc = json.load(open(doc_file))

            for i, sentence in enumerate(doc):
                sentence_tokens, cumul = self.tokenize_and_preserve_labels(sentence["tokens"], tokenizer)

                if len(sentence_tokens) <= config.max_sentence_length:
                    label_dict, sentence_holder_sizes, sentence_entity_sizes, sentence_event_sizes, sentence_span_sizes, sentence_implicit_attitude, sentence_explicit_attitude = self.create_label_dict(sentence["dse"], cumul)

                    seq_lens.append(len(sentence_tokens))
                    tokens.append(sentence_tokens)
                    lengths.append(len(sentence_tokens))
                    label_tuples.append(label_dict)
                    sentence_ids.append((doc_id, i))

                    holder_sizes.extend(sentence_holder_sizes)
                    entity_sizes.extend(sentence_entity_sizes)
                    event_sizes.extend(sentence_event_sizes)
                    span_sizes.extend(sentence_span_sizes)
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

        print("sentence lens        : 90%tile = {:.1f}, 95%tile = {:.1f}, 99%tile = {:.1f}, max = {:.1f}".format( np.quantile(seq_lens, 0.9), np.quantile(seq_lens, 0.95), np.quantile(seq_lens, 0.99), np.max(seq_lens) ))
        print("holder size          : 90%tile = {:.1f}, 95%tile = {:.1f}, 99%tile = {:.1f}, max = {:.1f}".format( np.quantile(holder_sizes, 0.9), np.quantile(holder_sizes, 0.95), np.quantile(holder_sizes, 0.99), np.max(holder_sizes) ))
        if entity_sizes:
            print("target entity size   : 90%tile = {:.1f}, 95%tile = {:.1f}, 99%tile = {:.1f}, max = {:.1f}".format( np.quantile(entity_sizes, 0.9), np.quantile(entity_sizes, 0.95), np.quantile(entity_sizes, 0.99), np.max(entity_sizes) ))
        if event_sizes:
            print("target event size    : 90%tile = {:.1f}, 95%tile = {:.1f}, 99%tile = {:.1f}, max = {:.1f}".format( np.quantile(event_sizes, 0.9), np.quantile(event_sizes, 0.95), np.quantile(event_sizes, 0.99), np.max(event_sizes) ))
        print("target span size     : 90%tile = {:.1f}, 95%tile = {:.1f}, 99%tile = {:.1f}, max = {:.1f}".format( np.quantile(span_sizes, 0.9), np.quantile(span_sizes, 0.95), np.quantile(span_sizes, 0.99), np.max(span_sizes) ))
        print("num holders per sent : 90%tile = {:.1f}, 95%tile = {:.1f}, 99%tile = {:.1f}, max = {:.1f}".format( np.quantile(n_holders_per_sentence, 0.9), np.quantile(n_holders_per_sentence, 0.95), np.quantile(n_holders_per_sentence, 0.99), np.max(n_holders_per_sentence) ))
        print("num targets per sent : 90%tile = {:.1f}, 95%tile = {:.1f}, 99%tile = {:.1f}, max = {:.1f}".format( np.quantile(n_targets_per_sentence, 0.9), np.quantile(n_targets_per_sentence, 0.95), np.quantile(n_targets_per_sentence, 0.99), np.max(n_targets_per_sentence) ))
        print("implicit attitude    : pos = {:4d}, neg = {:4d}, other = {:4d}, total = {:4d}".format(implicit_attitude[0], implicit_attitude[1], implicit_attitude[2], implicit_attitude.sum()))
        print("explicit attitude    : pos = {:4d}, neg = {:4d}, other = {:4d}, total = {:4d}".format(explicit_attitude[0], explicit_attitude[1], explicit_attitude[2], explicit_attitude.sum()))
        print("total                : pos = {:4d}, neg = {:4d}, other = {:4d}, total = {:4d}".format(implicit_attitude[0] + explicit_attitude[0], implicit_attitude[1] + explicit_attitude[1], implicit_attitude[2] + explicit_attitude[2], implicit_attitude.sum() + explicit_attitude.sum()))

        index = np.argsort(lengths)
        self.token_indices = torch.zeros((len(tokens), config.max_sentence_length), dtype=torch.long, device=config.device)
        self.mask = torch.zeros((len(tokens), config.max_sentence_length), dtype=torch.long, device=config.device)

        for i, j in enumerate(index):
            self.token_indices[i, :lengths[j]] = torch.LongTensor(tokenizer.convert_tokens_to_ids(tokens[j]))
            self.mask[i, :lengths[j]] = 1
        
        self.tokens = [tokens[i] for i in index]
        self.label_tuples = [label_tuples[i] for i in index]
        self.sentence_ids = [sentence_ids[i] for i in index]

    def __len__(self) -> int:
        return len(self.tokens)
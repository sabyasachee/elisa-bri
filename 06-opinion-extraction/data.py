# author : Sabyasachee Baruah

import os
import json
import math
import torch
import numpy as np
from collections import namedtuple
from transformers import BertTokenizer
from keras.preprocessing.sequence import pad_sequences

import utils
import config

FrameBatch = namedtuple("FrameBatch", "sentence_ids, sentences, sentence_token_ids, mask, frames, frame_to_sentence")

class FrameDataset:

    def __init__(self, docids, train=False) -> None:
        '''
        params:
            docids      : list of str docids
            train       : bool
            sentence_ids: list of sentence ids
                          sentence id is a 2-dimensional list: file name (str) and sentence index (int)
            sentences   : list of sentence
                          sentence is list of str words
            sentence_token_ids
                        : torch long tensor of shape D x L
            mask        : torch float tensor of shape D x L
            frames      : torch long tensor of shape D' x F x L
            frame_to_sentence
                        : torch int tensor of shape D'
        
        D is number of sentences
        L is max seq length
        D' is total number of labels
        F = 3 for number of argument types: holder, predicate, target

        if train is true, only sentences with some frames are included
        otherwise all sentences are included
        '''
        self.docids = docids
        self.train = train

        sentences = []
        sentence_ids = []
        sentence_token_ids = []
        frames = []
        frame_to_sentence = []
        tokenizer = BertTokenizer.from_pretrained(config.embed_model_name)
        tokenizer.vocab["[AUTHOR]"] = tokenizer.vocab.pop("[unused1]")
        

        for docid in self.docids:
            doc = json.load(open(os.path.join(config.DATA_FOLDER, "mpqa2-processed", docid, "tokenized.json")))

            for i, sentence in enumerate(doc):
                frame_tuples = set()
                for frame in sentence["dse"]:
                    if isinstance(frame["dse-span"], list) and isinstance(frame["target-span"], list) and (isinstance(frame["holder-span"], list) or frame["holder-type"] in ["writer", "implicit"]):
                        if isinstance(frame["holder-span"], list):
                            frame_tuples.add(tuple(frame["holder-span"] + frame["dse-span"] + frame["target-span"]))
                        else:
                            frame_tuples.add(tuple([-1, 0] + frame["dse-span"] + frame["target-span"]))
                
                if frame_tuples or not self.train:
                    j = len(sentence_token_ids)
                    new_tokens, mapping = utils.tokenize(tokenizer, sentence["tokens"])
                    frame = utils.create_frame_label(mapping, frame_tuples, max_seq_len=config.max_seq_len)
                    sentences.append(new_tokens)
                    sentence_ids.append([docid, j])
                    sentence_token_ids.append(tokenizer.convert_tokens_to_ids(new_tokens))
                    if frame.size:
                        frames.append(frame)
                        frame_to_sentence.extend([j for _ in range(len(frame))])
                    
        
        self.sentences = pad_sequences(sentences, maxlen=config.max_seq_len, padding="post", truncating="post", value="[PAD]", dtype=object).astype(str)

        self.sentence_ids = np.array(sentence_ids)

        sentence_token_ids = pad_sequences(sentence_token_ids, maxlen=config.max_seq_len, padding="post", truncating="post", value=tokenizer.vocab["[PAD]"])
        self.sentence_token_ids = torch.LongTensor(sentence_token_ids)
        
        mask = np.zeros(self.sentence_token_ids.shape, dtype=float)
        for i, sentence in enumerate(self.sentences):
            mask[i, : len(sentence)] = 1
        self.mask = torch.FloatTensor(mask)

        new_frames = []
        for sentence_frames in frames:
            for frame in sentence_frames:
                new_frame = pad_sequences(frame, maxlen=config.max_seq_len, padding="post", truncating="post", value=0)
                new_frames.append(new_frame)

        self.frames = torch.LongTensor(np.array(new_frames))
        self.frame_to_sentence = torch.IntTensor(frame_to_sentence)
    
    def print_sentence(self, i):
        print("SENTENCE ID : {}".format(self.sentence_ids[i]))
        print("TOKENS : {}".format(self.sentences[i]))
        frames = self.frames[self.frame_to_sentence == i]
        if frames.size > 0:
            print("FRAMES :")
            for j, frame in enumerate(frames):
                holder = " ".join(self.sentences[i][frame[0] != 0])
                predicate = " ".join(self.sentences[i][frame[1] != 0])
                target = " ".join(self.sentences[i][frame[2] != 0])
                print("{:2d}. HOLDER = '{}' PREDICATE = '{}' TARGET = '{}'".format(j + 1, holder, predicate, target))
        else:
            print("NO FRAMES")

class FrameIterator:

    def __init__(self, ds: FrameDataset, batch_size: int, shuffle_batch=True, shuffle_sample=True) -> None:
        '''
        params:
            ds          : FrameDataset
            batch_size  : int
            shuffle_batch : bool
                          if shuffle_batch is true, shuffle the order of batches
            shuffle_sample : bool
                          if shuffle_sample is true, shuffle the order of samples in adjacent batches
        '''
        self.ds = ds
        self.batch_size = batch_size
        self.shuffle_batch = shuffle_batch
        self.shuffle_sample = shuffle_sample
        sentence_lens = (ds.sentences != "[PAD]").sum(axis = 1)
        self.sorted_index = np.argsort(sentence_lens)
        self.n_batches = math.ceil(len(self.sorted_index)/self.batch_size)
    
    def __iter__(self):
        if self.shuffle_batch:
            self.batch_sequence = np.random.permutation(self.n_batches)
        else:
            self.batch_sequence = np.arange(self.n_batches)
        self.index = self.sorted_index.copy()
        if self.shuffle_sample:
            for i in range(self.n_batches - 2):
                np.random.shuffle(self.index[i * self.batch_size: (i + 3) * self.batch_size])
        self.i = 0
        return self
    
    def __next__(self) -> FrameBatch:
        if self.i == self.n_batches:
            raise StopIteration
        i = self.batch_sequence[self.i]
        batch_index = self.index[i * self.batch_size : (i + 1) * self.batch_size]
        batch_sentences = self.ds.sentences[batch_index]
        batch_sentence_token_ids = self.ds.sentence_token_ids[batch_index]
        batch_sentence_ids = self.ds.sentence_ids[batch_index]
        batch_mask = self.ds.mask[batch_index]
        batch_frames = torch.cat([self.ds.frames[self.ds.frame_to_sentence == j] for j in batch_index], dim = 0)
        batch_frame_to_sentence = torch.cat([torch.full(((self.ds.frame_to_sentence == j).sum().item(),), k) for k, j in enumerate(batch_index)], dim = 0)
        self.i += 1
        return FrameBatch(batch_sentence_ids, batch_sentences, batch_sentence_token_ids, batch_mask, batch_frames, batch_frame_to_sentence)
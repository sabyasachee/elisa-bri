# author : Sabyasachee

# utility functions

import numpy as np
from typing import List, Tuple

import config

def spans_overlap(span_x, span_y) -> bool:
    '''
    true if span_x overlaps span_y
    '''
    return span_x[0] < span_y[1] and span_y[0] < span_x[1]

def tokenize(tokenizer, tokens) -> Tuple[List[str], List[int]]:
    '''
    tokenizer is pretrained transformer model tokenizer
    tokens is list of str

    return:
        new tokens  : list of str
        mapping     : list of int
    
    prepend [AUTHOR] to new tokens
    if old tokens has m words and new tokens has n words,
    then mapping is m + 2 dimensional and each value lies between 0 and n

    mapping[i] = j means that ith word of old tokens begins at jth
    position in new tokens
    '''
    mapping = np.zeros(len(tokens) + 2, dtype=int)
    new_tokens = ["[AUTHOR]"]

    for i, token in enumerate(tokens):
        ttokens = tokenizer.tokenize(token)
        new_tokens.extend(ttokens)
        mapping[i + 2] = mapping[i + 1] + len(ttokens)

    mapping[1:] += 1
    return new_tokens, mapping

def create_frame_label(mapping, frame_tuples, max_seq_len=None) -> np.ndarray:
    '''
    mapping         : list of int
    frame_tuples    : set of frame tuples
                      each tuple is 6-dimensional: holder, predicate, and target
                      correct it by adding 1
    max_seq_len     : int
                      max sentence length
                      if an argument's span exceeds max_seq_len, ignore the frame
    
    return:
        frame label : numpy int array B' x F x L
                      B' is the number of frame tuples
                      F = 3 because of 3 arguments: holder, predicate, and target
                      L is the sentence len
                      frame label is 0 = O, 1 = B, or 2 = I
    '''
    frame_labels = []

    for frame_tuple in frame_tuples:
        frame_label = np.zeros((3, mapping[-1]))
        if max_seq_len is not None and mapping[max(frame_tuple[1], frame_tuple[3], frame_tuple[5])] >= max_seq_len:
            continue

        for i in range(frame_tuple[0] + 1, frame_tuple[1] + 1):
            j = mapping[i]
            if i == frame_tuple[0] + 1:
                frame_label[0, j] = 1
            else:
                frame_label[0, j] = 2
        
        for i in range(frame_tuple[2] + 1, frame_tuple[3] + 1):
            j = mapping[i]
            if i == frame_tuple[2] + 1:
                frame_label[1, j] = 1
            else:
                frame_label[1, j] = 2
        
        for i in range(frame_tuple[4] + 1, frame_tuple[5] + 1):
            j = mapping[i]
            if i == frame_tuple[4] + 1:
                frame_label[2, j] = 1
            else:
                frame_label[2, j] = 2

        frame_labels.append(frame_label)
    
    frame_label = np.array(frame_labels)
    return frame_label
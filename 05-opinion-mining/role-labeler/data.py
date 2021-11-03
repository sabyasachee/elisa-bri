# author : Sabyasachee

import os
import json
import torch
import config
import numpy as np
from tqdm import tqdm
from transformers import BertTokenizer
from keras_preprocessing.sequence import pad_sequences
from torch.utils.data import TensorDataset, DataLoader

class RoleLabelerDataset:
    '''
    dataloader for role labeling task
    instances have two variables:
        data = dataloader of wordid X_wordid, mask X_mask, and label Y
        dataset_sentenceid_to_mpqa_doc_and_sentenceid = dictionary mapping sentence id (first dimension of label Y) to [mpqaX, docid, sentenceid] where X can be 2 or 3
    '''

    @staticmethod
    def encode_attitude_type(attitude_type):
        # 0 for positive sentiment
        # 1 for negative sentiment
        # 2 for everything else

        if attitude_type == "sentiment-pos":
            return 0
        elif attitude_type == "sentiment-neg":
            return 1
        else:
            return 2

    def __init__(self, docids=None, mpqa2=False, ignore_negatives=False) -> None:
        # docids is list of doc ids
        # if docids is None, all docs are considered
        # 
        # mpqa2 is whether we choose from mpqa2 or mpqa3 (default)
        # 
        # if ignore_negatives is True, exclude sentences that do not have any attitude frames
        # set ignore_negatives if you want to use data for training and only want positive examples
        # 
        # self.data is a data loader of wordids, mask, and label
        # wordids is a LongTensor, mask is a FloatTensor, and label is an IntTensor of * x 6 shape
        # 
        # The first token of the wordids is always [AUTHOR] token
        # if y = label[i] is the label of sentence i
        # y[0] is the sentence id
        # (y[1], y[2]) is the holder span
        # (y[3], y[4]) is the target span
        # y[4] is the attitude type
        # 
        # attitude type is 0 for pos sentiment, 1 for neg sentiment, and 2 for other attitude
        # 
        # if ignore_negatives is false, sentences with no attitude frames have y = [sentence-id, 0, 0, 0, 0, -1]

        self.docids = docids
        self.mpqa2 = mpqa2
        self.ignore_negatives = ignore_negatives
        self.dataset_sentenceid_to_mpqa_doc_and_sentenceid = {}

        if mpqa2:
            
            data_folder = os.path.join(config.PROCESSED_FOLDER, "mpqa2-processed")
            if docids is None:
                docids = open(os.path.join(config.DATA_FOLDER, "database.mpqa.2.0/doclist.attitudeSubset")).read().splitlines()
            filename = "heuristic.json"
            key = "dse-heuristic"
            target_types = ["span"]

        else:
            
            data_folder = os.path.join(config.PROCESSED_FOLDER, "mpqa3-processed")
            if docids is None:
                docids = open(os.path.join(config.DATA_FOLDER, "database.mpqa.3.0/doclist")).read().splitlines()
            filename = "tokenized.json"
            key = "dse"
            target_types = ["entity", "event"]

        tokenizer = BertTokenizer.from_pretrained("SpanBERT/spanbert-large-cased")
        tokenizer.vocab["[AUTHOR]"] = tokenizer.vocab.pop("[unused1]")

        X_wordid = []
        Y = []
        sentenceid = 0

        for docid in tqdm(docids, desc="creating data tensors"):

            docfile = os.path.join(data_folder, docid, filename)
            doc = json.load(open(docfile))

            for k, sentence in enumerate(doc):

                self.dataset_sentenceid_to_mpqa_doc_and_sentenceid[sentenceid] = ["mpqa{}".format(3 - mpqa2), docid, k]

                n_previous_wordpiece_tokens = np.zeros(len(sentence["tokens"]) + 2, dtype=int)
                n_previous_wordpiece_tokens[1] = 1
                wordpiece_tokens = ["[AUTHOR]"]

                for i, token in enumerate(sentence["tokens"]):
                    
                    wp_tokens = tokenizer.tokenize(token)
                    n_previous_wordpiece_tokens[i + 2] = n_previous_wordpiece_tokens[i + 1] + len(wp_tokens)
                    wordpiece_tokens.extend(wp_tokens)

                sentence_Y = []

                for attitude_frame in sentence[key]:
                    
                    if attitude_frame["target-type"] in target_types:
                        
                        y = np.zeros(6, dtype=int)
                        y[0] = sentenceid
                        
                        if attitude_frame["holder-type"] == "span":

                            y[1] = attitude_frame["holder-span"][0] + 1
                            y[2] = attitude_frame["holder-span"][1] + 1
                        
                        else:

                            y[1] = 0
                            y[2] = 1

                        y[3] = attitude_frame["target-span"][0] + 1
                        y[4] = attitude_frame["target-span"][1] + 1

                        y[5] = self.encode_attitude_type(attitude_frame["attitude-type"])

                        sentence_Y.append(y)
                
                if sentence_Y:

                    sentence_Y = np.vstack(sentence_Y)
                    sentence_Y = np.unique(sentence_Y, axis=0)

                    for i in range(len(sentence_Y)):

                        sentence_Y[i, 1] = n_previous_wordpiece_tokens[sentence_Y[i, 1]]
                        sentence_Y[i, 2] = n_previous_wordpiece_tokens[sentence_Y[i, 2]]
                        sentence_Y[i, 3] = n_previous_wordpiece_tokens[sentence_Y[i, 3]]
                        sentence_Y[i, 4] = n_previous_wordpiece_tokens[sentence_Y[i, 4]]

                    wordpiece_tokenids = tokenizer.convert_tokens_to_ids(wordpiece_tokens)

                    for _ in range(len(sentence_Y)):

                        X_wordid.append(wordpiece_tokenids)
                    
                    Y.append(sentence_Y)
                
                elif not ignore_negatives:

                    wordpiece_tokenids = tokenizer.convert_tokens_to_ids(wordpiece_tokens)
                    X_wordid.append(wordpiece_tokenids)
                    y = np.zeros(6, dtype=int)
                    y[0] = sentenceid
                    y[5] = -1
                    Y.append(y)
                
                sentenceid += 1
                
        X_wordid = pad_sequences(X_wordid, maxlen=config.max_seq_len, padding="post", truncating="post", value=tokenizer.vocab["[PAD]"])
        X_mask = (X_wordid != tokenizer.vocab["[PAD]"]).astype(float)

        Y = np.vstack(Y)

        indices = []

        for i, y in enumerate(Y):
            if y[2] <= config.max_seq_len and y[4] <= config.max_seq_len:
                indices.append(i)

        X_wordid = X_wordid[indices]
        X_mask = X_mask[indices]
        Y = Y[indices]

        length = X_mask.sum(axis=1)
        sort_index = np.argsort(length)
        
        X_wordid = X_wordid[sort_index]
        X_mask = X_mask[sort_index]
        Y = Y[sort_index]

        X_wordid = torch.cuda.LongTensor(X_wordid, device=config.device)
        X_mask = torch.cuda.FloatTensor(X_mask, device=config.device)
        Y = torch.cuda.LongTensor(Y, device=config.device)

        self.data = DataLoader(TensorDataset(X_wordid, X_mask, Y), batch_size=config.batch_size, shuffle=True)
    
    def __len__(self):
        return len(self.data)
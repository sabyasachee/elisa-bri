# author : Sabyasachee Baruah

import numpy as np
import torch
from torchcrf import CRF
from torch import nn
from transformers import BertModel

class TargetLabeler(nn.Module):
    '''
    Model to find target (entity and event) spans given the opinion expression and MPQA2 target span
    Hparams:
        Bert pretrained model name
        Max seq len
        Position Embedding Size
        BiLSTM hidden size
        BiLSTM num layers
    '''

    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams
        self.encoder = BertModel.from_pretrained(hparams.pretrained_model_name)
        self.expression_position_embedding = nn.Embedding(2 * hparams.max_seq_len, hparams.position_embedding_size)
        self.target_position_embedding = nn.Embedding(2 * hparams.max_seq_len, hparams.position_embedding_size)
        self.bilstm = nn.LSTM(self.encoder.config.hidden_size + 2 * hparams.position_embedding_size, hparams.hidden_size, num_layers=hparams.bilistm_n_layers, bidirectional=True, batch_first=True)
        self.output = nn.Linear(2 * hparams.hidden_size, 3)
        self.crf = CRF(3, batch_first=True)

    def forward(self, X_wordid, X_mask, pos, Y=None):
        # X_wordid is the batch-size x max-seq-len matrix of WordPiece token indices
        # X_mask is the batch-size x max-seq-len matrix of 0s and 1s.
        # X_mask = 0 for padded tokens, otherwise 1
        # 
        # pos is the batch-size x 4 matrix
        # (pos[i, 0], pos[i, 1]) is the opinion expression span
        # (pos[i, 2], pos[i, 3]) is the target span
        # 
        # Y is the batch-size x max-seq-len matrix of the token labels
        # O = 0, B = 1, I = 2
        # Padded tokens also have label O (= 0)
        # 
        # If Y is not None, return loss
        # otherwise return Y_pred
        
        X_word = self.encoder(X_wordid, X_mask).last_hidden_state

        batch_size, max_seq_len = X_wordid.shape
        X_eposid = torch.tile(torch.LongTensor(np.arange(max_seq_len)), (batch_size, 1))
        X_tposid = torch.tile(torch.LongTensor(np.arange(max_seq_len)), (batch_size, 1))

        for i in range(batch_size):
            estart, eend, tstart, tend = pos[i]
            length = X_mask[i].sum()
            
            X_eposid[:estart] = estart - X_eposid[:estart]
            X_eposid[estart: eend] = 0
            X_eposid[eend:] = X_eposid[eend:] - eend + max_seq_len
            X_eposid[length:] = 2*max_seq_len - 1

            X_tposid[:tstart] = tstart - X_tposid[:tstart]
            X_tposid[tstart: tend] = 0
            X_tposid[tend:] = X_tposid[tend:] - tend + max_seq_len
            X_tposid[length:] = 2*max_seq_len - 1
        
        X_epos = self.expression_position_embedding(X_eposid)
        X_tpos = self.target_position_embedding(X_tposid)

        X = torch.cat((X_word, X_epos, X_tpos), dim=2)
        # X is of shape batch-size x max-seq-len x (Hw + 2 * Hpos)
        # Hw is the encoder hidden size, Hpos is the position embedding size

        A, _ = self.bilstm(X)
        A = A.contiguous()
        # A is of shape batch-size x max-seq-len x 2H
        # H is the hidden size

        B = self.output(A)
        # B is of shape batch-size x max-seq-len x 3

        crf_mask = X_mask.byte()
        Y_pred = self.crf.decode(B, crf_mask)
        # Y_pred is a list of list of integers

        if Y is not None:
            loss = -self.crf(B, Y, crf_mask)
            return loss, Y_pred
        else:
            return Y_pred

# author : Sabyasachee

import torch
from torch import nn
from transformers import BertModel

class FrameExtractor(nn.Module):

    def __init__(self, hparams):
        super().__init__()

        # TODO: dropout for BertModel
        self.embed = BertModel.from_pretrained(hparams.embed_model_name)
        self.embedding_size = self.embed.config.hidden_size

        if hparams.encoder_num_layers:
            self.encoder = nn.LSTM(self.embedding_size, hparams.encoder_hidden_size, 
                                    num_layers = hparams.encoder_num_layers, batch_first = True, 
                                    bidirectional = True, dropout = hparams.dropout)
            self.encoder_hidden_size = hparams.encoder_hidden_size
        else:
            self.encoder_hidden_size = self.embedding_size

        # 3 labels: B, I, O
        # 3 arguments: holder, target, predicate
        self.decoder = nn.LSTM(self.encoder_hidden_size + 3 * 3, hparams.decoder_hidden_size,
                                    batch_first = True)
        self.decoder_hidden_size = hparams.decoder_hidden_size

        self.holder_predictor = nn.Linear(self.decoder_hidden_size, 4)
        self.predicate_predictor = nn.Linear(self.decoder_hidden_size, 4)
        self.target_predictor = nn.Linear(self.decoder_hidden_size, 4)

    def forward(self, sentences, mask, frames=None, frame_to_sentence=None):
        '''
        sentences   : long tensor of shape B x L
        mask        : float tensor of shape B x L
        frames      : long tensor of shape B' x F x L
        frame_to_sentence 
                    : int tensor of shape B'

        B = batch size
        L = max seq len
        F = number of frame arguments = 3
        B' = corresponding label batch size
        '''

        embedding = self.embed(sentences, mask).last_hidden_state
        # embedding : float tensor of shape B x L x self.embedding_size

        encoding = self.encoder(embedding)
        # encoding  : float tensor of shape B x L x self.encoder_hidden_size


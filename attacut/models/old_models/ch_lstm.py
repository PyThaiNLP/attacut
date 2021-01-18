# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F

from slimcut import utils
from . import SyllableCharacterBaseModel

class Model(SyllableCharacterBaseModel):

    def __init__(self, data_config, model_config="emb:32|cell:64|l1:64"):
        super(Model, self).__init__()

        window_size = data_config['window_size']
        no_chars = data_config['num_tokens']

        self.num_embs = 2*window_size + 1

        config = utils.parse_model_params(model_config)
        emb_dim = config['emb']
        cell = config['cell']

        self.embeddings = nn.Embedding(
            no_chars,
            emb_dim,
            padding_idx=0
        )

        self.lstm = nn.LSTM(emb_dim, cell, batch_first=True)

        self.linear1 = nn.Linear(self.num_embs*cell, config['l1'])
        self.linear2 = nn.Linear(config['l1'], 1)

        self.model_params = model_config

    def forward(self, inputs):
        x, seq_lengths = inputs

        stsp_indices = utils.create_start_stop_indices(seq_lengths)

        embeds = list(map(
            lambda idx: self._embedding_and_lstm(x[:, idx[0]:idx[1]]),
            stsp_indices
        ))

        concat_emb = torch.cat(embeds, 1)

        out = F.relu(self.linear1(concat_emb))
        out = self.linear2(out)

        return out

    def _embedding_and_lstm(self, x):
        o, (h_n, c_n) = self.lstm(self.embeddings(x))

        return h_n.permute(1, 0, 2).view(x.size()[0], -1)
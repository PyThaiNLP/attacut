# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F

from . import SyllableBaseModel

class Model(SyllableBaseModel):
    def __init__(self, no_vocabs, embedding_dim=16, window_size=1):
        super(Model, self).__init__()

        self.input_size = 2*window_size+1
        
        self.embeddings = nn.ModuleList([
            nn.Embedding(
                no_vocabs,
                embedding_dim,
                padding_idx=0
            ) for i in range(self.input_size)
        ])

        self.linear1 = nn.Linear(self.input_size*embedding_dim, 64)
        self.linear2 = nn.Linear(64, 32)
        self.linear3 = nn.Linear(32, 1)

    def forward(self, inputs):
        embeds = list(map(
            lambda p: p[1](inputs[:, p[0]]),
            zip(range(self.input_size), self.embeddings)
        ))

        x = torch.cat(embeds, 1)

        out = F.relu(self.linear1(x))
        out = self.linear2(out)

        out = F.relu(out)
        out = self.linear3(out)

        return out
# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

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

        self.pooling = nn.AvgPool1d(embedding_dim)

        self.linear1 = nn.Linear(embedding_dim, 8)
        self.linear2 = nn.Linear(8, 1)

    def forward(self, inputs):
        embeds = list(map(
            lambda p: p[1](inputs[:, p[0]]), zip(range(self.input_size), self.embeddings)
        ))

        x = torch.stack(embeds, 2)
        x = self.pooling(x).view(x.size()[0], -1)

        out = F.relu(self.linear1(x))
        out = self.linear2(out)

        return out
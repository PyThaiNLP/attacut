# -*- coding: utf-8 -*-
import torch.nn as nn
import torch.nn.functional as F

from slimcut import utils

from . import SyllableBaseModel
class Model(SyllableBaseModel):
    def __init__(self, data_config, model_config="emb:32|l1:64"):
        super(Model, self).__init__()
        
        window_size = data_config['window_size']
        no_vocabs = data_config['num_tokens']

        config = utils.parse_model_params(model_config)
        emb_dim = config['emb']
        l1 = config['l1']

        self.embeddings = nn.Embedding(
            no_vocabs,
            emb_dim,
            padding_idx=0
        )

        self.linear1 = nn.Linear((2*window_size+1)*emb_dim, l1)
        self.linear2 = nn.Linear(l1, 1)

        self.model_params = model_config

    def forward(self, inputs):
        embeds = self.embeddings(inputs).view(inputs.size()[0], -1)
        out = F.relu(self.linear1(embeds))
        out = self.linear2(out)
        return out
import torch
import torch.nn as nn
import torch.nn.functional as F

from slimcut import utils
from . import SyllableCharacterBaseModel

class Model(SyllableCharacterBaseModel):

    def __init__(self, data_config, model_config="emb:16|l1:64"):
        super(Model, self).__init__()

        window_size = data_config['window_size']
        no_chars = data_config['num_tokens']
        max_length = data_config['max_seq_length']

        config = utils.parse_model_params(model_config)
        emb_dim = config['emb']
        
        self.embeddings = nn.Embedding(
            no_chars,
            emb_dim,
            padding_idx=0
        )

        self.num_embs = 2*window_size + 1
        self.pooling = nn.MaxPool1d(max_length)

        self.linear1 = nn.Linear(self.num_embs*emb_dim, config['l1'])
        self.linear2 = nn.Linear(config['l1'], 1)

    def forward(self, inputs):
        embeds = list(map(
            lambda i: self._embedding_and_pooling(inputs[:, i, :]),
            range(self.num_embs)
        ))

        concat_emb = torch.cat(embeds).view(inputs.size()[0], -1)

        out = F.relu(self.linear1(concat_emb))
        out = self.linear2(out)

        return out


    def _embedding_and_pooling(self, x):
        return self.pooling(self.embeddings(x).permute(0, 2, 1))
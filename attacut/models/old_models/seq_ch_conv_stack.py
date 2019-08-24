import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from slimcut import utils
from . import CharacterSeqBaseModel
from ..models import ConvolutionLayer


class Model(CharacterSeqBaseModel):
    def __init__(self, data_config, model_config="emb:32|c1:16|l1:8|do:0.1"):
        super(Model, self).__init__()

        no_chars = data_config['num_tokens']

        config = utils.parse_model_params(model_config)
        emb_dim = config['emb']
        dropout_rate = config.get("do", 0)

        self.embeddings = nn.Embedding(
            no_chars,
            emb_dim,
            padding_idx=0
        )

        self.dropout = torch.nn.Dropout(p=dropout_rate)

        kernel_size = 2
        self.conv1 = ConvolutionLayer(emb_dim, config['c1'], kernel_size)
        self.conv2 = ConvolutionLayer(config['c1'], config['c1'], kernel_size)
        self.conv3 = ConvolutionLayer(config['c1'], config['c1'], kernel_size)

        self.linear1 = nn.Linear(config['c1'], config['lu1'])
        self.linear2 = nn.Linear(config['l1'], 1)

        self.model_params = model_config

    def forward(self, inputs):
        x, seq_lengths = inputs

        embedding = self.embeddings(x).permute(0, 2, 1)

        conv1 = self.dropout(self.conv1(embedding)[:, :, :-1])
        conv2 = self.dropout(self.conv2(conv1)[:, :, :-1])
        conv3 = self.dropout(self.conv3(conv2)[:, :, :-1])

        out = conv3.permute(0, 2, 1)

        out = F.relu(self.linear1(out))
        out = self.linear2(out)

        out = out.view(-1)

        return out
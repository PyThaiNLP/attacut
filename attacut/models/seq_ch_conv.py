import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from attacut import utils, dataloaders
from . import BaseModel, ConvolutionLayer


class Model(BaseModel):
    dataset = dataloaders.CharacterSeqDataset

    def __init__(self, data_config, model_config="emb:32|conv:48|l1:16|do:0.1"):
        super(Model, self).__init__()

        no_chars = data_config['num_tokens']

        config = utils.parse_model_params(model_config)
        emb_dim = config['emb']
        conv_filters = config['conv']
        dropout_rate = config.get("do", 0)

        self.embeddings = nn.Embedding(
            no_chars,
            emb_dim,
            padding_idx=0
        )

        self.dropout= torch.nn.Dropout(p=dropout_rate)

        self.conv1 = ConvolutionLayer(emb_dim, conv_filters, 3)
        self.conv2 = ConvolutionLayer(emb_dim, conv_filters, 5, dilation=3)
        self.conv3 = ConvolutionLayer(emb_dim, conv_filters, 9, dilation=2)

        self.linear1 = nn.Linear(conv_filters, config['l1'])
        self.linear2 = nn.Linear(config['l1'], 1)

        self.model_params = model_config

    def forward(self, inputs):
        x, seq_lengths = inputs

        embedding = self.embeddings(x).permute(0, 2, 1)

        conv1 = self.conv1(embedding).permute(0, 2, 1)
        conv2 = self.conv2(embedding).permute(0, 2, 1)
        conv3 = self.conv3(embedding).permute(0, 2, 1)

        out = torch.stack((conv1, conv2, conv3), 3)

        out, _ = torch.max(out, 3)
        out = self.dropout(out)

        out = F.relu(self.linear1(out))
        out = self.linear2(out)

        out = out.view(-1)

        return out
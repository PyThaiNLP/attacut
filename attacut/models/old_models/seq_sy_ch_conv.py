import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from slimcut import utils
from . import SyllableCharacterSeqBaseModel


class Model(SyllableCharacterSeqBaseModel):
    def __init__(self, data_config, model_config="emb:8|conv:16|l1:16|do:0.0"):
        super(Model, self).__init__()

        no_chars = data_config['num_char_tokens']
        print("We have %d characters" % no_chars)
        no_syllables = data_config['num_tokens']
        print("We have %d syllables" % no_syllables)

        config = utils.parse_model_params(model_config)
        emb_dim = config['emb']
        conv_filters = config['conv']
        dropout_rate = config.get("do", 0)

        self.ch_embeddings = nn.Embedding(
            no_chars,
            emb_dim,
            padding_idx=0
        )

        self.sy_embeddings = nn.Embedding(
            no_syllables,
            emb_dim,
            padding_idx=0
        )

        self.dropout= torch.nn.Dropout(p=dropout_rate)

        kernel_size = 3
        self.conv1 = nn.Conv1d(
            emb_dim,
            conv_filters,
            kernel_size,
            stride=1,
            dilation=1,
            padding=kernel_size // 2
        )

        kernel_size = 5
        dilation = 3
        padding = kernel_size // 2
        padding += padding * (dilation-1)
        self.conv2 = nn.Conv1d(
            emb_dim,
            conv_filters,
            kernel_size,
            stride=1,
            dilation=dilation,
            padding=6
        )

        kernel_size = 9
        dilation = 2
        padding = kernel_size // 2
        padding += padding * (dilation-1)
        self.conv3 = nn.Conv1d(
            emb_dim,
            conv_filters,
            kernel_size,
            stride=1,
            dilation=dilation,
            padding=padding
        )


        self.linear1 = nn.Linear(conv_filters, config['l1'])
        self.linear2 = nn.Linear(config['l1'], 1)

        self.model_params = model_config

    def forward(self, inputs):
        x, seq_lengths = inputs

        x_char, x_syllable = x[:, 0, :], x[:, 1, :]

        ch_embedding = self.ch_embeddings(x_char).permute(0, 2, 1)
        sy_embedding = self.sy_embeddings(x_syllable).permute(0, 2, 1)

        embedding = ch_embedding  + sy_embedding

        conv1 = self.dropout(self.conv1(embedding).permute(0, 2, 1))
        conv2 = self.dropout(self.conv2(embedding).permute(0, 2, 1))
        conv3 = self.dropout(self.conv3(embedding).permute(0, 2, 1))

        out = torch.stack((conv1, conv2, conv3), 3)

        out, _ = torch.max(out, 3)

        out = F.relu(self.linear1(out))
        out = self.linear2(out)

        out = out.view(-1)

        return out
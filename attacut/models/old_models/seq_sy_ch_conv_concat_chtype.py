import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from slimcut import utils
from . import SyllableCharacterSeqWithChTypeBaseModel, ConvolutionBatchNorm
from slimcut import utils, character_type, dataloaders

class Model(SyllableCharacterSeqWithChTypeBaseModel):
    dataset =  dataloaders.SyllableCharacterSeqWithCharacterTypeDataset
    def __init__(self, data_config, model_config="embc:16|embs:8|embct:8|conv:16|l1:16|do:0.0"):
        super(Model, self).__init__()

        no_chars = data_config['num_char_tokens']
        print("We have %d characters" % no_chars)
        no_syllables = data_config['num_tokens']
        print("We have %d syllables" % no_syllables)

        config = utils.parse_model_params(model_config)
        conv_filters = config['conv']
        dropout_rate = config.get("do", 0)

        self.ch_embeddings = nn.Embedding(
            no_chars,
            config['embc'],
            padding_idx=0
        )

        self.ch_type_embeddings = nn.Embedding(
            character_type.TOTAL_CHARACTER_TYPES,
            config['embct'],
            padding_idx=0
        )

        self.sy_embeddings = nn.Embedding(
            no_syllables,
            config['embs'],
            padding_idx=0
        )

        emb_dim = config['embc'] + config['embs'] + config['embct']

        self.dropout= torch.nn.Dropout(p=dropout_rate)

        self.conv1 = ConvolutionBatchNorm(emb_dim, conv_filters, 3)
        self.conv2 = ConvolutionBatchNorm(emb_dim, conv_filters, 5, dilation=3)
        self.conv3 = ConvolutionBatchNorm(emb_dim, conv_filters, 9, dilation=2)

        self.linear1 = nn.Linear(conv_filters, config['l1'])
        self.linear2 = nn.Linear(config['l1'], 1)

        self.model_params = model_config

    def forward(self, inputs):
        x, seq_lengths = inputs

        x_char, x_char_type, x_syllable = x[:, 0, :], x[:, 1, :], x[:, 2, :]

        ch_embedding = self.ch_embeddings(x_char)
        ch_type_embedding = self.ch_type_embeddings(x_char_type)
        sy_embedding = self.sy_embeddings(x_syllable)

        embedding = torch.cat(
            (ch_embedding, ch_type_embedding, sy_embedding),
            dim=2
        )

        embedding = embedding.permute(0, 2, 1)

        conv1 = self.dropout(self.conv1(embedding).permute(0, 2, 1))
        conv2 = self.dropout(self.conv2(embedding).permute(0, 2, 1))
        conv3 = self.dropout(self.conv3(embedding).permute(0, 2, 1))

        out = torch.stack((conv1, conv2, conv3), 3)

        out, _ = torch.max(out, 3)

        out = F.relu(self.linear1(out))
        out = self.linear2(out)

        out = out.view(-1)

        return out
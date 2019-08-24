import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from slimcut import utils, character_type, dataloaders
from . import CharacterSeqWithChTypeBaseModel, ConvolutionLayer


class Model(CharacterSeqWithChTypeBaseModel):
    dataset = dataloaders.CharacterSeqWithCharacterTypeDataset

    def __init__(self, data_config, model_config="emb_c:8|emb_t:8|conv:8|l1:6|do:0.1"):
        super(Model, self).__init__()

        no_chars = data_config['num_tokens']

        config = utils.parse_model_params(model_config)
        emb_c_dim = config['emb_c']
        emb_t_dim = config['emb_t']

        conv_filters = config['conv']
        dropout_rate = config.get("do", 0)

        self.ch_embeddings = nn.Embedding(
            no_chars,
            emb_c_dim,
            padding_idx=0
        )

        self.ch_type_embeddings = nn.Embedding(
            character_type.TOTAL_CHARACTER_TYPES,
            emb_t_dim,
            padding_idx=0
        )

        self.dropout= torch.nn.Dropout(p=dropout_rate)


        emb_dim = emb_c_dim + emb_t_dim

        self.conv1 = ConvolutionLayer(emb_dim, conv_filters, 3)
        self.conv2 = ConvolutionLayer(emb_dim, conv_filters, 5, dilation=3)
        self.conv3 = ConvolutionLayer(emb_dim, conv_filters, 9, dilation=2)

        self.linear1 = nn.Linear(conv_filters, config['l1'])
        self.linear2 = nn.Linear(config['l1'], 1)

        self.model_params = model_config

    def forward(self, inputs):
        x, seq_lengths = inputs
        x_char, x_char_type = x[:, 0, :], x[:, 1, :]

        ch_embedding = self.ch_embeddings(x_char)
        ch_type_embedding = self.ch_type_embeddings(x_char_type)

        embedding = torch.cat((ch_embedding, ch_type_embedding), dim=2)

        embedding = embedding.permute(0, 2, 1)

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
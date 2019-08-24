import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from slimcut import utils
from . import SyllableSeqBaseModel

class Model(SyllableSeqBaseModel):
    def __init__(self, data_config, model_config="emb:32|cell:64|l1:64"):
        super(Model, self).__init__()

        no_chars = data_config['num_tokens']

        config = utils.parse_model_params(model_config)
        emb_dim = config['emb']
        cell = config['cell']

        self.embeddings = nn.Embedding(
            no_chars,
            emb_dim,
            padding_idx=0
        )

        self.lstm = nn.LSTM(emb_dim, cell, batch_first=True)

        self.linear1 = nn.Linear(cell, config['l1'])
        self.linear2 = nn.Linear(config['l1'], 1)

        self.model_params = model_config

    def forward(self, inputs):
        x, seq_lengths = inputs

        embedding = self.embeddings(x)
        packed_input = pack_padded_sequence(
            embedding,
            seq_lengths.cpu().numpy(),
            batch_first=True
        )

        packed_output, (ht, ct) = self.lstm(packed_input)
        output, input_sizes = pad_packed_sequence(packed_output, batch_first=True)

        out = F.relu(self.linear1(output))
        out = self.linear2(out)
        out = out.view(-1)

        return out
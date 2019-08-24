import numpy as np

import torch
from torch.utils.data import Dataset

from attacut import utils, pipeline

class SequenceDataset(Dataset):
    def __init__(self, path: str=None):
        if path:
            self.load_preprocessed_data(path)

    def __len__(self):
        return self.total_samples

    @staticmethod
    def _process_line(line):
        label, indices = line.split("::")

        y = np.array(list(label)).astype(int)
        x = np.array(indices.split(" ")).astype(int)

        seq = len(y)

        return (x, seq), y

    def __getitem__(self, index):
        return self.data[index]

    def load_preprocessed_data(self, path):
        self.data = []

        suffix = path.split("/")[-1]
        with open(path) as f, \
            utils.Timer("load-seq-data--%s" % suffix) as timer:
            for line in f:
                self.data.append(SequenceDataset._process_line(line))

        self.total_samples = len(self.data)

    @staticmethod
    def collate_fn(batch):
        total_samples = len(batch)

        seq_lengths = np.array(list(map(lambda x: x[0][1], batch)))
        max_length = np.max(seq_lengths)

        features = np.zeros((total_samples, max_length), dtype=np.int64)
        labels = np.zeros((total_samples, max_length), dtype=np.int64)

        for i, s in enumerate(batch):
            b_feature = s[0][0]
            total_features = len(b_feature)
            features[i, :total_features] = b_feature
            labels[i, :total_features] = s[1]

        seq_lengths = torch.from_numpy(seq_lengths)
        seq_lengths, perm_idx = seq_lengths.sort(0, descending=True)


        inputs = (torch.from_numpy(features)[perm_idx], seq_lengths)

        labels = torch.from_numpy(labels)[perm_idx]

        return inputs, labels

    @staticmethod
    def prepare_model_inputs(inputs, device="cpu"):

        x, seq_lengths = inputs[0]
        x = x.to(device)
        y = inputs[1].float().to(device).reshape(-1)

        return (x, seq_lengths), y, y.shape[0]

    def make_feature(self, txt):
        raise NotImplementedError("make_feature")

class CharacterSeqDataset(SequenceDataset):
    def setup_featurizer(self, path: str):
        self.dict = utils.load_dict(f"{path}/characters.json")

        return dict(num_tokens=len(self.dict))

    # def make_feature(self, txt):
    def make_feature(self, txt):
        characters = list(txt)
        ch_ix = list(map(lambda c: pipeline.mapping_char(self.dict, c), characters))

        features = np.array(ch_ix, dtype=np.int64).reshape((1, -1))

        seq_lengths = np.array([features.shape[-1]], dtype=np.int64)

        return characters, (torch.from_numpy(features), torch.from_numpy(seq_lengths))


class SyllableCharacterSeqDataset(SequenceDataset):
    def __getitem__(self, index):
        label, character_indices, syllable_indices = self.data[index]
        y = np.array(list(label)).astype(int)

        cx = np.array(character_indices.split(" ")).astype(int)
        sx = np.array(syllable_indices.split(" ")).astype(int)

        seq = len(y)

        x = np.stack((cx, sx), axis=0)

        return (x, seq), y

    @staticmethod
    def collate_fn(batch):
        total_samples = len(batch)

        seq_lengths = np.array(list(map(lambda x: x[0][1], batch)))
        max_length = np.max(seq_lengths)

        features = np.zeros((total_samples, 2, max_length), dtype=np.int64)
        labels = np.zeros((total_samples, max_length), dtype=np.int64)

        for i, s in enumerate(batch):
            b_feature = s[0][0]
            total_features = b_feature.shape[1]
            features[i, :, :total_features] = b_feature
            labels[i, :total_features] = s[1]

        seq_lengths = torch.from_numpy(seq_lengths)
        seq_lengths, perm_idx = seq_lengths.sort(0, descending=True)

        inputs = (torch.from_numpy(features)[perm_idx], seq_lengths)

        labels = torch.from_numpy(labels)[perm_idx]

        return inputs, labels

    @staticmethod
    def prepare_model_inputs(inputs, device="cpu"):

        x, seq_lengths = inputs[0]
        x = x.to(device)
        y = inputs[1].float().to(device).reshape(-1)

        return (x, seq_lengths), y, y.shape[0]

class CharacterSeqWithCharacterTypeDataset(SyllableCharacterSeqDataset):
    pass

class SyllableCharacterSeqWithCharacterTypeDataset(SyllableCharacterSeqDataset):
    def __getitem__(self, index):
        label, ch_indices, sy_indices, ch_type_indices = self.data[index]

        y = np.array(list(label)).astype(int)

        cx = np.array(ch_indices.split(" ")).astype(int)
        ct = np.array(ch_type_indices.split(" ")).astype(int)
        sx = np.array(sy_indices.split(" ")).astype(int)

        seq = len(y)

        x = np.stack((cx, ct, sx), axis=0)

        return (x, seq), y
    @staticmethod
    def collate_fn(batch):
        total_samples = len(batch)

        seq_lengths = np.array(list(map(lambda x: x[0][1], batch)))
        max_length = np.max(seq_lengths)

        dim = batch[0][0][0].shape[0]
        features = np.zeros((total_samples, dim, max_length), dtype=np.int64)
        labels = np.zeros((total_samples, max_length), dtype=np.int64)

        for i, s in enumerate(batch):
            b_feature = s[0][0]
            total_features = b_feature.shape[1]
            features[i, :, :total_features] = b_feature
            labels[i, :total_features] = s[1]

        seq_lengths = torch.from_numpy(seq_lengths)
        seq_lengths, perm_idx = seq_lengths.sort(0, descending=True)

        inputs = (torch.from_numpy(features)[perm_idx], seq_lengths)

        labels = torch.from_numpy(labels)[perm_idx]

        return inputs, labels
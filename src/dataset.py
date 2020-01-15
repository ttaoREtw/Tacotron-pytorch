import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from .symbols import txt2seq
from functools import partial


def getDataLoader(mode, meta_path, data_dir, batch_size, r, n_jobs, use_gpu, **kwargs):
    bs = batch_size
    if mode == 'train':
        shuffle = True
    elif mode == 'test':
        shuffle = False
    else:
        raise NotImplementedError
    DS = MyDataset(meta_path, data_dir)
    DL = DataLoader(
            DS, batch_size=batch_size, shuffle=shuffle, drop_last=False,
            num_workers=n_jobs, collate_fn=partial(collate_fn, r=r), pin_memory=use_gpu)
    return DL


def _pad(seq, max_len):
    seq = np.pad(seq, (0, max_len - len(seq)),
            mode='constant', constant_values=0)
    return seq


def _pad_2d(x, max_len):
    x = np.pad(x, [(0, max_len - len(x)), (0, 0)],
            mode="constant", constant_values=0)
    return x


class MyDataset(Dataset):
    """Dataset
    """
    def __init__(self, meta_path, data_dir):
        # Load meta
        # ---------
        # text: texts
        # mel : filenames of mel-spectrogram
        # spec: filenames of (linear) spectrogram
        #
        meta = {'text':[], 'mel': [], 'spec': []}
        with open(meta_path) as f:
            for line in f.readlines():
                # If there is '\n' in text, it will be discarded when calling symbols.txt2seq
                fmel, fspec, n_frames, text = line.split('|')
                meta['text'].append(text)
                meta['mel'].append(fmel)
                meta['spec'].append(fspec)

        self.X = meta['text']
        self.Y_mel = [os.path.join(data_dir, f) for f in meta['mel']]
        self.Y_spec = [os.path.join(data_dir, f) for f in meta['spec']]
        assert len(self.X) == len(self.Y_mel) == len(self.Y_spec)
        # Text to id sequence
        self.X = [txt2seq(x) for x in self.X]

    def __getitem__(self, idx):
        item = (self.X[idx],
                np.load(self.Y_mel[idx]),
                np.load(self.Y_spec[idx]))
        return item

    def __len__(self):
        return len(self.X)


def collate_fn(batch, r):
    """Create batch"""
    input_lengths = [len(x[0]) for x in batch]
    max_input_len = np.max(input_lengths)
    # (r9y9's comment) Add single zeros frame at least, so plus 1
    max_target_len = np.max([len(x[1]) for x in batch]) + 1
    if max_target_len % r != 0:
        max_target_len += r - max_target_len % r
        assert max_target_len % r == 0

    a = np.array([_pad(x[0], max_input_len) for x in batch], dtype=np.int)
    x_batch = torch.LongTensor(a)

    input_lengths = torch.LongTensor(input_lengths)

    b = np.array([_pad_2d(x[1], max_target_len) for x in batch],
                 dtype=np.float32)
    mel_batch = torch.FloatTensor(b)

    c = np.array([_pad_2d(x[2], max_target_len) for x in batch],
                 dtype=np.float32)
    spec_batch = torch.FloatTensor(c)
    return x_batch, input_lengths, mel_batch, spec_batch




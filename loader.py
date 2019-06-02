from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pickle
import json
import numpy as np
import random
import scipy.signal
import torch
import torch.autograd as autograd
import torch.utils.data as tud

class createDataset(tud.Dataset):

    def __init__(self, dataset_path, batch_size):

        self.data = read_data(dataset_path)
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        datum = self.data[idx]
        datum = make_pairs(datum)
        return datum

class BatchRandomSampler(tud.sampler.Sampler):
    """
    Batches the data consecutively and randomly samples
    by batch without replacement.
    """

    def __init__(self, data_source, batch_size):
        it_end = len(data_source) - batch_size + 1
        self.batches = [range(i, i + batch_size)
                for i in range(0, it_end, batch_size)]
        self.data_source = data_source

    def __iter__(self):
        random.shuffle(self.batches)
        return (i for b in self.batches for i in b)

    def __len__(self):
        return len(self.data_source)
    
def make_loader(dataset_path, batch_size, num_workers=4):
    dataset = createDataset(dataset_path, batch_size)
    sampler = BatchRandomSampler(dataset, batch_size)
    loader = tud.DataLoader(dataset,
                batch_size=batch_size,
                sampler=sampler,
                num_workers=num_workers,
                collate_fn=lambda batch : zip(*batch),
                drop_last=True)
    return loader

def read_data(dataset_path):
    pickles = {}
    with (open(dataset_path, "rb")) as openfile:
        while True:
            try:
                pickles.update(pickle.load(openfile, encoding = 'latin1')) 
            except EOFError:
                break
    data = []
    for i in pickles.keys():
        data.append(pickles[i])
    return data

def make_pairs(datum):
    return datum['audio'], datum['text']
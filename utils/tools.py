# Tools

import random
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import torch
import torch.utils.data
from kornia.geometry.transform import resize
from sklearn.model_selection import train_test_split


def upscale_tensors_like_2d(like: torch.Tensor, tensors: Sequence[torch.Tensor]):
    h, w = like.shape[-1], like.shape[-2]
    return [resize(t, size=(h, w), interpolation='nearest') for t in tensors]


def read_subject_label(label_path: Path):
    assert label_path.is_file(), f"Label path is not exists at ${label_path}"

    label_file = pd.read_excel(str(label_path))
    subject_id, subject_class = np.array(label_file.ID), np.array(label_file.Disease)
    return {v: subject_class[k]  for k,v in enumerate(subject_id)}


def split_data(label, test_size: float, random_state: int = 42):
    train_key, test_key = train_test_split(list(label.keys()), test_size=test_size, stratify=list(label.values()), random_state=random_state)
    train_label, test_label = list(map(lambda l: {k: label[k] for k in l}, [train_key, test_key]))
    return train_label, test_label


class BalancedBatchSampler(torch.utils.data.sampler.Sampler):
    def __init__(self, dataset, labels=None):
        self.labels = labels
        self.dataset = dict()
        self.balanced_max = 0
        # Save all the indices for all the classes
        for idx in range(0, len(dataset)):
            label = self._get_label(dataset, idx)
            if label not in self.dataset:
                self.dataset[label] = list()
            self.dataset[label].append(idx)
            self.balanced_max = len(self.dataset[label]) \
                if len(self.dataset[label]) > self.balanced_max else self.balanced_max
        
        # Oversample the classes with fewer elements than the max
        for label in self.dataset:
            while len(self.dataset[label]) < self.balanced_max:
                self.dataset[label].append(random.choice(self.dataset[label]))
        self.keys = list(self.dataset.keys())
        self.currentkey = 0
        self.indices = [-1]*len(self.keys)

    def __iter__(self):
        while self.indices[self.currentkey] < self.balanced_max - 1:
            self.indices[self.currentkey] += 1
            yield self.dataset[self.keys[self.currentkey]][self.indices[self.currentkey]]
            self.currentkey = (self.currentkey + 1) % len(self.keys)
        self.indices = [-1]*len(self.keys)
    
    def _get_label(self, dataset, idx, labels = None):
        if self.labels is not None:
            return self.labels[idx]
        else:
            raise Exception("You should pass the tensor of labels to the constructor as second argument")

    def __len__(self):
        return self.balanced_max*len(self.keys)

import pickle as pkl
import torch
from itertools import zip_longest
import random
from . import constants
from . import utils


class Dataset(object):
    def __init__(self, tsv_file, vocab_file, batch_size=32, task='vp'):
        self.batch_size = batch_size
        with open(vocab_file, 'rb') as f:
            self.dict = pkl.load(f)
        deps = utils.deps_from_tsv(tsv_file)
        self.task = task
        if task == 'vp':
            self.task_vp(deps)
        else:
            self.task_lm(deps)
        self.shuffle()

    def __len__(self):
        return self.num_batches

    def __getitem__(self, index):
        assert index < self.num_batches
        return self._data[index]

    def shuffle(self):
        if self.task == 'vp':
            self.shuffle_vp()
        else:
            self.shuffle_lm()

    def shuffle_lm(self):
        random.shuffle(self.data)
        self._data = []
        for i in range(0, len(self.data), self.batch_size):
            mb = self.data[i: i+self.batch_size]
            mb.sort(key=len, reverse=True)
            mb = torch.LongTensor(
                list(zip_longest(*mb, fillvalue=0)))
            self._data += [mb]

        self.num_batches = len(self._data)

    def shuffle_vp(self):
        random.shuffle(self.data)
        self._data = []
        for i in range(0, len(self.data), self.batch_size):
            mb = self.data[i: i+self.batch_size]
            mb.sort(key=lambda x: len(x[0]), reverse=True)
            mbx = [x[0] for x in mb]
            mby = [x[1] for x in mb]
            mbx = torch.LongTensor(
                list(zip_longest(*mbx, fillvalue=0)))
            mby = torch.FloatTensor(mby)
            self._data += [(mbx, mby)]

        self.num_batches = len(self._data)

    def task_lm(self, deps):
        self.data = []
        for dep in deps:
            xs = dep['sentence'].split()
            xs = [self.dict.get(x, constants.unk_idx) for x in xs]
            xs = [constants.bos_idx] + xs + [constants.eos_idx]
            self.data += [xs]

    def task_vp(self, deps):
        self.data = []
        self.class_to_code = {'VBZ': 0, 'VBP': 1}
        self.code_to_class = {x: y for y, x in self.class_to_code.items()}
        for dep in deps:
            v = int(dep['verb_index']) - 1
            x = dep['sentence'].split()[:v]
            y = self.class_to_code[dep['verb_pos']]
            x = [self.dict.get(w, constants.unk_idx) for w in x]
            self.data += [(x, y)]

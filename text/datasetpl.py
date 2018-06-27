import torch
from itertools import zip_longest
import random
import os


class Dataset(object):
    def __init__(self, dir):
        n = 13
        self.trainfiles = [os.path.join(dir, 'train%i' % i) for i in range(n)]
        self.validfiles = [os.path.join(dir, 'valid%i' % i) for i in range(n)]
        self.testfiles = [os.path.join(dir, 'test%i' % i) for i in range(n)]
        self.token2idx, self.class2idx = self.build_dict()
        self.idx2token = {}
        for w, idx in self.token2idx.items():
            self.idx2token[idx] = w

        self.idx2class = {}
        for c, idx in self.class2idx.items():
            self.idx2class[idx] = c
        self.make_data()

    def build_dict(self):
        class2idx = {}
        token2idx = {'<pad>': 0}
        for fname in self.trainfiles:
            with open(fname) as f:
                for line in f:
                    col = line.split('\t')
                    if col[0] not in class2idx:
                        class2idx[col[0]] = len(class2idx)
                    for w in col[1].split() + col[2].split():
                        if w not in token2idx:
                            token2idx[w] = len(token2idx)
        return token2idx, class2idx

    def tensor2str(self, x):
        return ' '.join([self.idx2token[w] for w in x.view(-1).tolist()])

    def numberize(self, file):
        data = []
        for line in open(file, 'r'):
            col = line.split('\t')
            y = self.class2idx[col[0]]
            xs1 = [self.token2idx[w] for w in col[1].split()]
            xs2 = [self.token2idx[w] for w in col[2].split()]
            data.append((y, xs1, xs2))
        return data

    def make_data(self):
        self.train_data = {}
        for i, fname in enumerate(self.trainfiles):
            self.train_data[i] = self.numberize(fname)

        self.test_data = {}
        for i, fname in enumerate(self.testfiles):
            self.test_data[i] = self.numberize(fname)

        self.valid_data = {}
        for i, fname in enumerate(self.validfiles):
            self.valid_data[i] = self.numberize(fname)

    @staticmethod
    def batchify(data, batch_size):
        batched_data = []
        for i in range(0, len(data), batch_size):
            mb = data[i:i+batch_size]
            ys = torch.LongTensor([m[0] for m in mb])
            xs1 = [m[1] for m in mb]
            xs1 = torch.LongTensor(
                list(zip_longest(*xs1, fillvalue=0)))
            xs2 = [m[2] for m in mb]
            xs2 = torch.LongTensor(
                list(zip_longest(*xs2, fillvalue=0)))
            batched_data += [(ys, xs1, xs2)]
        return batched_data

    def get_train(self, maxop=5, batch_size=16):
        data = []
        for i in range(maxop):
            data.extend(self.train_data[i])
        random.shuffle(data)
        return self.batchify(data, batch_size)

    def get_test(self, nops, batch_size):
        data = self.test_data[nops]
        return self.batchify(data, batch_size)

    def get_valid(self, nops, batch_size):
        data = self.valid_data[nops]
        return self.batchify(data, batch_size)

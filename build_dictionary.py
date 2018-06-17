# import text.utils as utils
from text import utils
import pickle as pkl
from text import constants
from sys import argv
import os
# import text.constants as constants

infile = 'data/agr_50_mostcommon_10K.tsv'
worddict = {}
worddict[constants.pad] = constants.pad_idx
worddict[constants.unk] = constants.unk_idx  # probably we won't need this
worddict[constants.bos] = constants.bos_idx
worddict[constants.eos] = constants.eos_idx


for dep in utils.deps_from_tsv(infile):
    for w in dep['sentence'].split():
        if w not in worddict:
            worddict[w] = len(worddict)
vocab_file = os.path.join(argv[1], 'vocab.pkl')
print('| write vocabulary to %s' % vocab_file)
with open(vocab_file, 'wb') as f:
    pkl.dump(worddict, f)
print('| vocabulary size %d' % len(worddict))
print('| done!')

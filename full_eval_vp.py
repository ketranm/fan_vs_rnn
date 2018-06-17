import argparse
import pickle as pkl
import torch
from torch.autograd import Variable
from text.utils import gen_inflect_from_vocab, deps_from_tsv
from itertools import zip_longest
from collections import Counter
from layers import TFNVP, RNNVP


parser = argparse.ArgumentParser(description='Evaluate LM')
parser.add_argument('-checkpoint', required=True,
                    help='file saved trained parameters')
parser.add_argument('-input', required=True, help='input file.')
parser.add_argument('-output', required=True,
                    help='output stats for plotting.')
parser.add_argument('-batch_size', type=int, default=256,
                    help='batch size, set larger for speed.')
opt = parser.parse_args()

checkpoint = torch.load(opt.checkpoint)
saved_opt = checkpoint['opt']
print('| trained configuration')
print(saved_opt)
print('-' * 42)
saved_opt.input = opt.input
saved_opt.output = opt.output
saved_opt.batch_size = opt.batch_size
opt = saved_opt
print('| reconstruct network')

if opt.arch == 'rnn':
    model = RNNVP(opt.word_vec_size, opt.n_words, opt.layers,
                  opt.dropout)
else:
    model = TFNVP(opt.word_vec_size, opt.n_words, opt.num_heads,
                  opt.head_size, opt.layers, opt.inner_size,
                  opt.dropout)
print(model)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = model.to(device)

print('| load parameters')
model.load_state_dict(checkpoint['params'])
model.eval()


# build inverse dictionary
w2idx = pkl.load(open(opt.dict, 'rb'))
print('| vocab size: %d' % len(w2idx))

inflect_verb, _ = gen_inflect_from_vocab('data/wiki.vocab')

# read data here
deps = deps_from_tsv(opt.input)
dis_hit = Counter()
dis_tot = Counter()
dif_hit = Counter()
dif_tot = Counter()

bidx = 0
for i in range(0, len(deps), opt.batch_size):
    mb = []
    bidx += 1
    if bidx % 100 == 0:
        print('process {:5d} / {:d}'.format(i, len(deps)))
    for dep in deps[i: i+opt.batch_size]:
        n = int(dep['n_intervening'])
        n_diff = int(dep['n_diff_intervening'])
        d = int(dep['distance'])
        if n > 4 or d > 16:
            continue
        y = 1 if dep['verb_pos'] == 'VBP' else 0
        tokens = dep['sentence'].split()
        v = int(dep['verb_index']) - 1
        ws = ['<bos>'] + tokens[:v]
        ws = [w2idx.get(w, 1) for w in ws]
        mb += [(ws, y, d, n, n_diff)]
    mb.sort(key=lambda t: len(t[0]), reverse=True)
    xs = [x[0] for x in mb]
    xs = torch.LongTensor(list(zip_longest(*xs, fillvalue=0))).to(device)
    scores = model(xs)
    yhat = scores.view(-1).ge(0.5).tolist()
    for i, v in enumerate(yhat):
        y = mb[i][1]
        c = 1 if y == v else 0
        dx = mb[i][2]
        dis_tot[dx] += 1
        dis_hit[dx] += c
        if mb[i][3] == mb[i][4]:
            n = mb[i][3]
            dif_tot[n] += 1
            dif_hit[n] += c


dis_acc = {}
dis_acc = torch.zeros(17)
dif_acc = torch.zeros(5)
print("Accuracy by distance")
for k in sorted(dis_hit.keys()):
    v = dis_hit[k]
    acc = v / dis_tot[k]
    dis_acc[k] = acc
    print("%d | %.2f" % (k, acc))

print("Accuracy by intervenings!")
for k in sorted(dif_hit.keys()):
    v = dif_hit[k]
    acc = v * 1./dif_tot[k]
    dif_acc[k] = acc
    print("%d | %.2f" % (k, acc))

stats = {'distance': dis_acc, 'intervenings': dif_acc}
torch.save(stats, opt.output)

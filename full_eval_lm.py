import torch
import layers
import argparse
import pickle as pkl
from text.utils import gen_inflect_from_vocab, deps_from_tsv
from itertools import zip_longest
from collections import Counter


parser = argparse.ArgumentParser(description='Evaluate LM')
parser.add_argument('-checkpoint', required=True,
                    help='file saved trained parameters')
parser.add_argument('-input', required=True, help='input file.')
parser.add_argument('-output', required=True, help='output for plotting.')
parser.add_argument('-batch_size', type=int, default=256,
                    help='batch size, set larger for speed.')
opt = parser.parse_args()

checkpoint = torch.load(opt.checkpoint)
saved_opt = checkpoint['opt']
saved_opt.input = opt.input
saved_opt.output = opt.output
saved_opt.batch_size = opt.batch_size
opt = saved_opt
print('| reconstruct network')

if opt.arch == 'rnn':
    model = layers.RNNLM(opt.word_vec_size, opt.n_words, opt.layers,
                         opt.dropout, opt.tied)
else:
    model = layers.Transformer(opt.word_vec_size, opt.n_words,
                               opt.num_heads, opt.head_size,
                               opt.layers, opt.inner_size,
                               opt.dropout, opt.tied)

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

        t_verb = dep['verb']
        f_verb = inflect_verb[t_verb]
        t_verb = w2idx.get(t_verb, 1)
        if t_verb == 1:
            continue
        f_verb = w2idx.get(f_verb, 1)

        tokens = dep['sentence'].split()
        v = int(dep['verb_index']) - 1
        ws = ['<bos>'] + tokens[:v]
        ws = [w2idx.get(w, 1) for w in ws]
        mb += [(ws, t_verb, f_verb, d, n, n_diff)]
    mb.sort(key=lambda t: len(t[0]), reverse=True)
    xs = [x[0] for x in mb]
    tv = [x[1] for x in mb]
    fv = [x[2] for x in mb]
    xs = torch.LongTensor(list(zip_longest(*xs, fillvalue=0))).to(device)
    scores, _ = model(xs, True)
    b = [i for i in range(len(tv))]
    true_scores = scores[b, tv]  # advance indexing
    fake_scores = scores[b, fv]
    corrects = true_scores.gt(fake_scores).view(-1).tolist()
    for i, v in enumerate(corrects):
        dx = mb[i][3]
        dis_tot[dx] += 1
        dis_hit[dx] += v
        if mb[i][4] == mb[i][5]:
            n = mb[i][4]
            dif_tot[n] += 1
            dif_hit[n] += v


dis_acc = {}
dis_acc = torch.zeros(17)
dif_acc = torch.zeros(5)
print(dis_tot)
print('Accuracy by distance')
for k in sorted(dis_hit.keys()):
    v = dis_hit[k]
    acc = v / dis_tot[k]
    dis_acc[k] = acc
    print("%d | %.2f" % (k, acc))

# print(dis_acc)
print('Accuracy by intervenings')
for k in sorted(dif_hit.keys()):
    v = dif_hit[k]
    acc = v * 1./dif_tot[k]
    print("%d | %.2f" % (k, acc))
    dif_acc[k] = acc

stats = {'distance': dis_acc, 'intervenings': dif_acc}
torch.save(stats, opt.output)

import argparse
import torch
import torch.nn as nn
import layers
import opts
from text.dataset import Dataset
import time
import math


parser = argparse.ArgumentParser(description='Language Model')

parser.add_argument('-train', required=True, default='train.txt', type=str,
                    help='train file, one sentence per line.')
parser.add_argument('-valid', required=True, default='valid.txt', type=str,
                    help='validation file.')
# dictionaries
parser.add_argument('-dict', required=True, default='vocab.pkl',
                    help='vocabulary file.')

# opts.py
opts.add_md_help_argument(parser)
opts.model_opts(parser)
opts.train_opts(parser)
opts.preprocess_opts(parser)

opt = parser.parse_args()
# for grid search
opt.inner_size = 2 * opt.word_vec_size
opt.head_size = opt.word_vec_size // opt.num_heads

print(opt)
print('-' * 42)

torch.manual_seed(opt.seed)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    torch.cuda.manual_seed(opt.seed)


def prepare_batch(mb):
    mb = mb.to(device)
    x = mb[:-1, :].clone()
    y = mb[1:, :].clone()
    return x, y.view(-1)


def build_crit(n_words):
    weight = torch.ones(n_words)
    weight[0] = 0
    crit = nn.NLLLoss(weight, size_average=False)
    return crit.to(device)


def eval(model, valid, crit):
    model.eval()
    valid_nlls = []
    n_words = 0
    for i in range(len(valid)):
        x, y = prepare_batch(valid[i])
        log_prob, _ = model(x)
        nll = crit(log_prob, y)
        valid_nlls.append(nll.item())
        n_words += y.ne(0).int().sum().item()
    model.train()
    nll = torch.FloatTensor(valid_nlls).sum().item() / n_words
    return math.exp(nll)


def train(opt):
    print('| build data iterators')
    train = Dataset(opt.train, opt.dict, opt.batch_size, task='lm')
    valid = Dataset(opt.valid, opt.dict, opt.batch_size, task='lm')

    print('| build model')
    if opt.n_words < 0:
        opt.n_words = len(train.dict)

    print('| vocab size %d' % opt.n_words)
    print('| build criterion')
    crit = build_crit(opt.n_words)

    if opt.arch == 'rnn':
        print('| build LSTM LM')
        model = layers.RNNLM(opt.word_vec_size, opt.n_words, opt.layers,
                             opt.dropout, tied=opt.tied)
    else:
        print('| build Transformer')
        model = layers.Transformer(opt.word_vec_size, opt.n_words,
                                   opt.num_heads, opt.head_size,
                                   opt.layers, opt.inner_size,
                                   opt.dropout, tied=opt.tied)
    print(model)
    model = model.to(device)
    # eval(model, valid, crit)
    optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr)

    best_valid_ppl = 1e10
    min_lr = opt.lr * math.pow(0.5, 5)
    for eidx in range(opt.epochs):
        model.train()
        tot_loss = 0
        n_words = 0
        train.shuffle()
        num_batches = len(train)
        ud_start = time.time()
        for i in range(len(train)):
            optimizer.zero_grad()
            x, y = prepare_batch(train[i])
            log_prob, _ = model(x)
            loss = crit(log_prob, y)
            nx = y.data.ne(0).int().sum().item()
            loss.backward()
            if opt.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(),
                                               opt.max_grad_norm)

            optimizer.step()
            tot_loss += loss.item()
            n_words += nx
            if i % opt.report_every == 0 and i > 0:
                ud = time.time() - ud_start
                args = [eidx, i, num_batches, math.exp(tot_loss/n_words),
                        opt.report_every/ud]
                print("| Epoch {:2d} | {:d} / {:d} | ppl {:.3f} "
                      "| speed {:.1f} b/s".format(*args))
                ud_start = time.time()

        print('| Evaluate')
        model.eval()
        valid_ppl = eval(model, valid, crit)
        print('| Epoch {:2d} | valid ppl {:.3f}'
              .format(eidx, valid_ppl))
        if valid_ppl <= best_valid_ppl:
            print('| Save checkpoint: %s | Valid ppl: %.3f' %
                  (opt.save_model, valid_ppl))
            checkpoint = {'params': model.state_dict(),
                          'opt': opt,
                          'best_valid_ppl': valid_ppl}
            torch.save(checkpoint, opt.save_model)
            best_valid_ppl = valid_ppl
        else:
            opt.lr = opt.lr * 0.5
            if opt.lr < min_lr:
                print('reach minimum learning rate!')
                exit()
            print('decay learning rate %f' % opt.lr)
            for group in optimizer.param_groups:
                group['lr'] = opt.lr


train(opt)

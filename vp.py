import argparse
import torch
import torch.nn as nn
from text.dataset import Dataset
from layers import TFNVP, RNNVP
import math
import time
import opts


# build opt parser
parser = argparse.ArgumentParser(description='Verb Inflection Model')

parser.add_argument('-train', required=True, default='train.tsv', type=str,
                    help='train file.')
parser.add_argument('-valid', required=True, default='valid.tsv', type=str,
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

opt.inner_size = 2 * opt.word_vec_size
opt.head_size = opt.word_vec_size // opt.num_heads

print(opt)
print('-' * 42)
torch.manual_seed(opt.seed)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    torch.cuda.manual_seed(opt.seed)


def prepare_batch(mb):
    x, y = mb[0].to(device), mb[1].to(device)
    return x, y


def eval(model, valid):
    model.eval()
    tot_val = 0.0
    crt_val = 0.0
    for i in range(len(valid)):
        x, y = prepare_batch(valid[i])
        log_prob = model(x)
        yhat = log_prob.view(-1).ge(0.5)
        crt_val += yhat.eq(y.byte()).sum().item()
        tot_val += yhat.numel()
    val_acc = crt_val / tot_val
    return val_acc


def train(opt):
    print('| build data iterators')
    train = Dataset(opt.train, opt.dict, batch_size=32, task='vp')
    valid = Dataset(opt.valid, opt.dict, batch_size=32, task='vp')

    if opt.n_words < 0:
        opt.n_words = len(train.dict)
    print('| vocab size %d' % opt.n_words)

    crit = nn.BCELoss(size_average=False).to(device)
    if opt.arch == 'rnn':
        print('Build LSTM model')
        model = RNNVP(opt.word_vec_size, opt.n_words, opt.layers,
                      opt.dropout)
    else:
        print('Build FAN model')
        model = TFNVP(opt.word_vec_size, opt.n_words, opt.num_heads,
                      opt.head_size, opt.layers, opt.inner_size,
                      opt.dropout)
    print(model)
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr)
    # tracking validation accuracy
    best_valid_acc = 0
    for eidx in range(opt.epochs):
        tot_loss = 0
        n_samples = 0
        model.train()  # make sure we are in training mode
        train.shuffle()
        ud_start = time.time()
        for i in range(len(train)):
            optimizer.zero_grad()
            x, y = prepare_batch(train[i])
            log_prob = model(x)
            loss = crit(log_prob.view(-1), y)
            n_samples += x.size(1)
            tot_loss += loss.item()
            loss.backward()
            if opt.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(),
                                               opt.max_grad_norm)
            optimizer.step()
            if i % opt.report_every == 0 and i > 0:
                ud = time.time() - ud_start
                args = [eidx, i, len(train), math.exp(tot_loss/n_samples),
                        opt.report_every/ud]
                print("| Epoch {:2d} | {:d} / {:d} | ppl {:.3f} "
                      "| speed {:.1f} b/s".format(*args))
                ud_start = time.time()

        print('| Evaluate')
        val_acc = eval(model, valid)

        if val_acc >= best_valid_acc:
            print('| Save checkpoint: %s | Valid acc: %f' %
                  (opt.save_model, val_acc))
            checkpoint = {'params': model.state_dict(),
                          'opt': opt,
                          'best_valid_acc': best_valid_acc}
            best_valid_acc = val_acc
            torch.save(checkpoint, opt.save_model)


train(opt)

import argparse
import torch
import torch.nn as nn
from text.datasetpl import Dataset
from layers import PLRNN, PLTFN
import math
import time
import opts
import numpy as np

# build opt parser
parser = argparse.ArgumentParser(description='propositional logic')

parser.add_argument('-data_dir', required=True,
                    default='propositionallogic', type=str,
                    help='train file, one sentence per line.')
parser.add_argument('-max_bin', type=int, default=7,
                    help='max number of logical operations.')

# opts.py
opts.add_md_help_argument(parser)
opts.model_opts(parser)
opts.train_opts(parser)
opts.preprocess_opts(parser)
opt = parser.parse_args()
opt.cuda = len(opt.gpuid) > 0
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
    y, x1, x2 = mb
    return y.to(device), x1.to(device), x2.to(device)


def eval(model, dataset, valid=True):
    print('-' * 42)
    print('| Evaluating')
    g_tot, g_crt = 0, 0
    bin_acc = []
    for i in range(1, 13):
        if valid:
            test_data_i = dataset.get_valid(i, 64)
        else:
            test_data_i = dataset.get_test(i, 64)
        tot, crt = 0., 0.
        for mb in test_data_i:
            y, x1, x2 = prepare_batch(mb)
            pred = model(x1, x2)
            _, yhat = pred.max(1)
            crt += y.eq(yhat).long().sum().item()
            tot += y.numel()
        acc = crt * 100. / tot
        g_tot += tot
        g_crt += crt
        bin_acc += [acc]
        print("| n_ops: %2d | accuracy: %.2f | %4d" % (i, acc, tot))
    g_acc = g_crt * 100 / g_tot
    print('| Average accuracy: %.2f' % g_acc)
    print('-' * 42)
    report = {'bin_acc': bin_acc, 'avg_acc': g_acc}
    return report


def noam_scheduler(optimizer, step):
    lr = (np.power(opt.word_vec_size, -0.5)
          * min(np.power(step, -0.5),
                step * np.power(opt.warmup, -1.5)))
    optimizer.param_groups[0]['lr'] = lr
    return lr


def train():
    dataset = Dataset(opt.data_dir)
    vocab_size = len(dataset.token2idx)
    n_classes = len(dataset.class2idx)

    if opt.arch == 'rnn':
        model = PLRNN(opt.word_vec_size, vocab_size, n_classes,
                      opt.layers, opt.dropout)
    else:
        model = PLTFN(opt.word_vec_size, vocab_size,
                      n_classes, opt.num_heads,
                      opt.head_size, opt.layers, opt.inner_size,
                      opt.dropout)
    print(model)
    print('initialize model')
    for p in model.parameters():
        p.data.uniform_(-opt.param_init, opt.param_init)
    crit = nn.NLLLoss(size_average=False)
    model = model.to(device)
    crit = crit.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr)
    best_valid_acc = 0

    for eidx in range(opt.epochs):
        tot_loss = 0
        n_samples = 0
        train_data = dataset.get_train(opt.max_bin, opt.batch_size)
        num_batches = len(train_data)
        ud_start = time.time()
        model.train()
        for i, mb in enumerate(train_data):
            y, x1, x2 = prepare_batch(mb)
            optimizer.zero_grad()
            pred = model(x1, x2)
            loss = crit(pred, y)
            loss.backward()
            if opt.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(),
                                               opt.max_grad_norm)
            optimizer.step()
            tot_loss += loss.item()
            n_samples += y.numel()

            if i % opt.report_every == 0 and i > 0:
                ud = time.time() - ud_start
                args = [eidx, i / num_batches, math.exp(tot_loss/n_samples),
                        opt.report_every/ud, opt.lr]
                print("| epoch {:2d} | {:.2f}% | ppl {:.3f} "
                      "| speed {:.1f} b/s | lr {:.5f}".format(*args))
                ud_start = time.time()
        model.eval()
        report = eval(model, dataset, True)
        if report['avg_acc'] > best_valid_acc:
            # save checkpoint here!
            best_valid_acc = report['avg_acc']
            # print('| Run model on test data!')
            test_report = eval(model, dataset, False)
            checkpoint = {'params': model.state_dict(),
                          'opt': opt,
                          'report': test_report}
            torch.save(checkpoint, opt.save_model)
            print('| Save checkpoint: %s | Valid ppl: %.3f' %
                  (opt.save_model, best_valid_acc))


train()

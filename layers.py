import torch
import torch.nn as nn
import torch.nn.functional as F
import modules
import math
from torch.nn.utils.rnn import pack_padded_sequence
from torch.nn.utils.rnn import pad_packed_sequence


__author__ = "Ke Tran"


def add_timing_signal(x, max_timescale=1e4):
    batch, length, channels = x.size()
    nts = channels // 2
    log_inc = math.log(max_timescale) / nts
    log_inv_inc = -log_inc * torch.arange(0, nts)
    inv_inc = log_inv_inc.exp().view(1, -1).expand(length, nts)
    pos_idx = torch.arange(0, length).view(-1, 1).expand(length, channels)
    pos_emb = torch.FloatTensor(length, channels)
    pos_emb[:, 0::2] = (pos_idx[:, 0::2] * inv_inc).sin()
    pos_emb[:, 1::2] = (pos_idx[:, 1::2] * inv_inc).cos()
    return pos_emb.type_as(x.data) + x
    # return Variable(pos_emb.type_as(x.data), requires_grad=False) + x


def add_timing_signal_t(x, t, max_timescale=1e4):
    r"""Adds timing signal at time-step t to x"""
    batch, _, channels = x.size()
    nts = channels // 2
    log_inc = math.log(max_timescale) / nts
    log_inv_inc = -log_inc * torch.arange(0, nts)
    inv_inc = log_inv_inc.exp().view(1, nts)
    pos_emb = torch.FloatTensor(1, channels)
    pos_emb[:, 0::2] = (inv_inc * t).sin()
    pos_emb[:, 1::2] = (inv_inc * t).cos()
    return pos_emb.type_as(x.data) + x
    # return Variable(pos_emb.type_as(x.data), requires_grad=False) + x


def get_padding_mask(q, k):
    r"""Gets padding mask when use query q for key k

    Args:
        q: a Variable LongTensor with shape (batch, length_q)
        k: a Variable LongTensor with shape (batch, length_k)

    Returns:
        a ByteTensor with shape (batch, length_q, length_k)
    """

    masked_pads = k.data.eq(0)
    return masked_pads[:, None, :].expand(k.size(0), q.size(1), k.size(1))


def get_causal_mask(q):
    r"""Gets causal mask. This prevents attention mechanism looks into future

    Args:
        q: a LongTensor with shape (batch, length_q)

    Returns:
        a ByteTensor with shape (batch, length_q, length_q)
    """
    batch, length = q.size()
    tt = torch.cuda if q.is_cuda else torch
    mask = tt.ByteTensor(length, length).fill_(1).triu_(1)
    causal_mask = mask.unsqueeze(0).expand(batch, length, length)
    return causal_mask


class SubLayer(nn.Module):
    r'''A layer consists of one attention layer and a residual feed forward'''
    def __init__(self, input_size, num_heads, head_size, inner_size, dropout):
        super(SubLayer, self).__init__()
        self.attn = modules.MultiHeadAttention(input_size,
                                               num_heads,
                                               head_size)
        self.layer_norm = nn.LayerNorm(input_size)
        self.resff = modules.ResFF(input_size, inner_size, dropout)

    def forward(self, input, mask=None):
        output, attn = self.attn(input, input, input, mask)
        output = self.layer_norm(output + input)
        return self.resff(output), attn


class Encoder(nn.Module):
    r'''Attentive Encoder'''

    def __init__(self, input_size, vocab_size, num_heads, head_size,
                 num_layers, inner_size, dropout):
        super(Encoder, self).__init__()
        self.lut = nn.Embedding(vocab_size, input_size, padding_idx=0)
        self.layers = nn.ModuleList(
            [SubLayer(input_size, num_heads, head_size, inner_size, dropout)
             for i in range(num_layers)])

    def forward(self, input):
        mask = get_padding_mask(input, input)
        word_vec = self.lut(input)
        outputs = [add_timing_signal(word_vec, 1000)]
        attns = []
        for i, layer in enumerate(self.layers):
            output, attn = layer(outputs[i], mask)
            attns += [attn]
            outputs += [output]
        self.attns = attns  # dirty hack to expose attention for visualization
        return outputs


class RNNLM(nn.Module):
    r'''Baseline LSTM Language Model'''
    def __init__(self, input_size, vocab_size, num_layers,
                 dropout, tied=False):
        super(RNNLM, self).__init__()
        self.lut = nn.Embedding(vocab_size, input_size, padding_idx=0)
        self.generator = nn.Linear(input_size, vocab_size)
        self.rnn = nn.LSTM(input_size, input_size, num_layers,
                           dropout=dropout)
        if tied:
            self.generator.weight = self.lut.weight

    def forward(self, input, last=False):
        '''forward pass
            input: a Variable Tensor with shape (batch x bptt)
            last: boolean, return the predictions of the last element if True
        '''
        lengths = list(input.data.ne(0).sum(0).view(-1))
        word_vec = self.lut(input)
        packed_vec = pack_padded_sequence(word_vec, lengths)
        outputs, hidden = self.rnn(packed_vec)
        outputs = pad_packed_sequence(outputs)[0]
        if last:
            hx, _ = hidden
            logits = self.generator(hx[-1])
        else:
            logits = self.generator(outputs.contiguous()
                                    .view(-1, outputs.size(-1)))
        return F.log_softmax(logits, dim=-1), hidden


class Transformer(nn.Module):
    r'''Fully Attentional Language Model'''

    def __init__(self, input_size, vocab_size, num_heads, head_size,
                 num_layers, inner_size, dropout, tied=False):
        super(Transformer, self).__init__()
        self.lut = nn.Embedding(vocab_size, input_size, padding_idx=0)
        self.layers = nn.ModuleList(
            [SubLayer(input_size, num_heads, head_size, inner_size, dropout)
             for i in range(num_layers)])
        self.generator = nn.Linear(input_size, vocab_size)
        if tied:
            self.lut.weight = self.generator.weight

    def forward(self, input, last=False):
        input = input.t()
        mask = get_causal_mask(input)
        word_vec = self.lut(input)
        outputs = add_timing_signal(word_vec, 1000)
        attns = []
        for i, layer in enumerate(self.layers):
            outputs, attn = layer(outputs, mask)
            attns += [attn]
        if last:
            last_outputs = []
            lengths = list(input.data.ne(0).sum(1) - 1)
            b = [i for i in range(len(lengths))]
            last_outputs = outputs[b, lengths, :]
            logits = self.generator(last_outputs)
        else:
            outputs = outputs.transpose(0, 1).contiguous()
            logits = self.generator(outputs.view(-1, outputs.size(-1)))
        return F.log_softmax(logits, dim=-1), attns


# VERB prediction nets
class TFNVP(nn.Module):
    r'''Transformer network for verb prediction'''

    def __init__(self, input_size, vocab_size, num_heads, head_size,
                 num_layers, inner_size, dropout, tied=False):
        super(TFNVP, self).__init__()
        self.lut = nn.Embedding(vocab_size, input_size, padding_idx=0)
        self.layers = nn.ModuleList(
            [SubLayer(input_size, num_heads, head_size, inner_size, dropout)
             for i in range(num_layers)])
        self.generator = nn.Linear(input_size, 1)

    def forward(self, input):
        input = input.t()
        mask = get_causal_mask(input)
        word_vec = self.lut(input)
        outputs = add_timing_signal(word_vec, 1000)
        attns = []
        for i, layer in enumerate(self.layers):
            outputs, attn = layer(outputs, mask)
            attns += [attn]
        last_outputs = []
        # expose attns for visualization
        self.attns = attns
        lengths = list(input.data.ne(0).sum(1) - 1)
        b = [i for i in range(len(lengths))]
        last_outputs = outputs[b, lengths, :]
        logits = self.generator(last_outputs)
        return F.sigmoid(logits)


class RNNVP(nn.Module):
    r"""Baseline LSTM Language Model"""
    def __init__(self, input_size, vocab_size, num_layers, dropout):
        super(RNNVP, self).__init__()
        self.lut = nn.Embedding(vocab_size, input_size, padding_idx=0)
        self.generator = nn.Linear(input_size, 1)
        self.rnn = nn.LSTM(input_size, input_size, num_layers,
                           dropout=dropout)

    def forward(self, input):
        '''forward pass
            input: a Variable Tensor with shape (batch x bptt)
        '''
        lengths = list(input.data.ne(0).sum(0).view(-1))
        word_vec = self.lut(input)
        packed_vec = pack_padded_sequence(word_vec, lengths)
        _, (hx, _) = self.rnn(packed_vec)
        logits = self.generator(hx[-1])
        return F.sigmoid(logits)


class PLTFN(nn.Module):
    '''Transformer for propositional logic task, fully self-attention'''
    def __init__(self, input_size, vocab_size, n_classes, num_heads, head_size,
                 num_layers, inner_size, dropout):
        super(PLTFN, self).__init__()
        self.k = num_heads
        self.lut = nn.Embedding(vocab_size, input_size, padding_idx=0)
        self.encoder = Encoder(input_size, vocab_size, num_heads, head_size,
                               num_layers, inner_size, dropout)
        self.q = nn.Parameter(torch.randn(1, self.k, input_size))
        self.attn = modules.MultiHeadAttention(input_size,
                                               num_heads,
                                               head_size)

        self.classifier = nn.Sequential(
            nn.Linear(input_size * 2 * self.k, input_size),
            nn.ReLU(),
            nn.Linear(input_size, input_size),
            nn.ReLU(),
            nn.Linear(input_size, n_classes),
            nn.LogSoftmax(dim=-1)
        )

    def fw(self, x):
        x = x.t()  # batch x length
        h = self.encoder(x)
        bsize = x.size(0)
        q = self.q.repeat(bsize, 1, 1)
        mask = x.data.eq(0)
        mask = mask[:, None, :].expand(bsize, self.k, x.size(1))
        hx, attn = self.attn(q, h[-1], h[-1], mask)
        return hx.view(bsize, -1)

    def forward(self, x1, x2):
        # hack to expose attentions
        self.attns = []
        h1 = self.fw(x1)
        self.attns += [self.encoder.attns]
        h2 = self.fw(x2)
        self.attns += [self.encoder.attns]
        h = torch.cat([h1, h2], 1)

        return self.classifier(h)


class PLTFNx(nn.Module):
    '''Transformer for propositional logic task,
    this model does not access future context'''
    def __init__(self, input_size, vocab_size, n_classes, num_heads, head_size,
                 num_layers, inner_size, dropout):
        super(PLTFNx, self).__init__()
        self.lut = nn.Embedding(vocab_size, input_size, padding_idx=0)
        self.layers = nn.ModuleList(
            [SubLayer(input_size, num_heads, head_size, inner_size, dropout)
             for i in range(num_layers)])
        self.classifier = nn.Sequential(
            nn.Linear(input_size * 2, input_size),
            nn.ReLU(),
            nn.Linear(input_size, input_size),
            nn.ReLU(),
            nn.Linear(input_size, n_classes),
            nn.LogSoftmax(dim=-1)
        )

    def fw(self, x):
        length = list(x.data.ne(0).sum(0) - 1)
        b = [i for i in range(x.size(1))]
        # transpose for transformer
        x = x.t()
        mask = get_causal_mask(x)
        e = self.lut(x)
        o = add_timing_signal(e, 1000)
        for i, layer in enumerate(self.layers):
            o, attn = layer(o, mask)

        return o[b, length, ]

    def forward(self, x1, x2):
        h1 = self.fw(x1)
        h2 = self.fw(x2)
        h = torch.cat([h1, h2], 1)
        return self.classifier(h)


class PLRNN(nn.Module):
    '''Recurrent model for propositional logic task, using tensor network
    but the result does not changes significantly'''
    def __init__(self, input_size, vocab_size, n_classes, num_layers, dropout):
        super(PLRNN, self).__init__()
        self.lut = nn.Embedding(vocab_size, input_size, padding_idx=0)
        if num_layers == 1:
            self.rnn = nn.LSTM(input_size, input_size, 1,
                               dropout=dropout)
        else:
            self.rnns = nn.ModuleList(
                [nn.LSTM(input_size, input_size, 1, dropout=dropout)
                 for i in range(num_layers)])

        self.classifier = nn.Sequential(
            nn.Linear(input_size * 2, input_size),
            nn.ReLU(),
            nn.Linear(input_size, input_size),
            nn.ReLU(),
            nn.Linear(input_size, n_classes),
            nn.LogSoftmax(dim=-1)
        )

    def fw(self, x):
        length = list(x.data.ne(0).sum(0) - 1)
        b = [i for i in range(x.size(1))]
        prev_o = self.lut(x)
        if hasattr(self, 'rnns'):
            for i, rnn in enumerate(self.rnns):
                o, _ = rnn(prev_o)
                prev_o = prev_o + o  # residual connection
            output = prev_o
        else:
            output, _ = self.rnn(prev_o)
        return output[length, b, ]

    def forward(self, x1, x2):
        h1 = self.fw(x1)
        h2 = self.fw(x2)
        h = torch.cat([h1, h2], 1)
        return self.classifier(h)


class PLBase(nn.Module):
    '''Baseline for checking the difficulty of the task!'''
    def __init__(self, input_size, vocab_size, n_classes, num_layers=0,
                 dropout=0.1, op='max'):
        super(PLBase, self).__init__()
        self.lut = nn.Embedding(vocab_size, input_size, padding_idx=0)
        self.op = op
        if num_layers > 0:
            self.rnn = nn.LSTM(input_size, input_size//2, num_layers,
                               dropout=dropout, bidirectional=True)
            # using RNN
        self.classifier = nn.Sequential(
            nn.Linear(input_size * 2, input_size),
            nn.ReLU(),
            nn.Linear(input_size, input_size),
            nn.ReLU(),
            nn.Linear(input_size, n_classes),
            nn.LogSoftmax(dim=-1)
        )

    def fw(self, x):
        x = self.lut(x)  # bptt, batch, size
        if hasattr(self, 'rnn'):
            length = list(x.data.ne(0).sum(0))
            packed_x = pack_padded_sequence(x, length)
            x, _ = self.rnn(packed_x)
            x = pad_packed_sequence(x)[0]

        if self.op == 'max':
            return x.max(0)[0]
        elif self.op == 'avg':
            return x.mean(0)
        else:
            return x.sum(0)

    def forward(self, x1, x2):
        h1 = self.fw(x1)
        h2 = self.fw(x2)
        h = torch.cat([h1, h2], 1)
        return self.classifier(h)

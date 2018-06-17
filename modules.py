import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np


def scaled_dot_attn(q, k, v, scale=1.0, mask=None):
    r"""Simplified attention without dropout

    Args:
        q: a Variable Tensor with shape (batch, length_q, dim)
        k: a Variable Tensor with shape (batch, length_k, dim)
        v: a Variable Tensor with shape (batch, length_v, dim)

    Returns:
        a Variable Tensor with shape (batch, length_q, dim)
    """

    attn = torch.bmm(q, k.transpose(1, 2)) * scale
    if mask is not None:
        attn.data.masked_fill_(mask, -math.inf)
    attn = F.softmax(attn, dim=-1)
    return torch.bmm(attn, v), attn


class MultiHeadAttention(nn.Module):
    r"""Multi-Head Attention [1]: https://arxiv.org/abs/1706.03762

    Args:
        input_size: size of each input sample
        num_heads: number of attention heads
        head_size: size of each output head

    Inputs:
        q: a Tensor with shape (batch, length_q, input_size)
        k: a Tensor with shape (batch, length_k, input_size)
        v: a Tensor with shape (batch, length_k, input_size)

    Outputs:
        a Tensor with shape (batch, length_q, input_size)
    """

    def __init__(self, input_size, num_heads, head_size):
        self.num_heads = num_heads
        self.head_size = head_size
        self.scale = np.power(head_size, -0.5)

        super(MultiHeadAttention, self).__init__()
        output_size = num_heads * head_size
        self.linear_q = nn.Linear(input_size, output_size, bias=False)
        self.linear_k = nn.Linear(input_size, output_size, bias=False)
        self.linear_v = nn.Linear(input_size, output_size, bias=False)
        self.linear_o = nn.Linear(output_size, input_size, bias=False)

    def tf(self, fn, x):
        """Transforms inputs with resepect to multi-heads
        Args:
            fn: linear function
            x: a Tensor with shape (batch, length, dim)

        Returns:
            a Tensor with shape (batch * num_heads, length, head_size)
        """

        batch, length, dim = x.size()
        x = fn(x.view(-1, dim)).view(batch, length, self.num_heads, -1)
        return x.transpose(1, 2).contiguous().view(-1, length, self.head_size)

    def forward(self, q, k, v, mask=None):
        batch, length_q, dim = q.size()
        qs = self.tf(self.linear_q, q)
        ks = self.tf(self.linear_k, k)
        vs = self.tf(self.linear_v, v)
        if mask is not None:
            mask = mask.unsqueeze(1).repeat(1, self.num_heads, 1, 1) \
                    .view(-1, mask.size(1), mask.size(2))
        output, attn = scaled_dot_attn(qs, ks, vs, self.scale, mask)
        output_2d = output.view(batch, -1, length_q, self.head_size) \
            .transpose(1, 2).contiguous().view(batch * length_q, -1)

        output_3d = self.linear_o(output_2d).view(batch, length_q, -1)
        return output_3d, attn


class LayerNorm(nn.Module):
    r"""Layer normalization class. Normalization is done on the last dimension

    Args:
        input_size: size of input sample

    Inputs:
        a Tensor with shape (batch, length, input_size) or (batch, input_size)

    Outputs:
        a Tensor with shape (batch, length, input_size) or (batch, input_size)
    """

    def __init__(self, input_size, eps=1e-10):
        super(LayerNorm, self).__init__()
        self.eps = eps
        self.a = nn.Parameter(torch.ones(input_size))
        self.b = nn.Parameter(torch.zeros(input_size))

    def forward(self, input):
        # perform normalization on the last dimension
        mu = input.mean(-1).unsqueeze(-1)
        sigma = input.std(-1).unsqueeze(-1)
        output = (input - mu) / (sigma + self.eps)
        output = output * self.a.expand_as(output) + self.b.expand_as(output)
        return output


class ResFF(nn.Module):
    r"""Position-wise Feed-Forward Net"""
    def __init__(self, input_size, inner_size, res_dropout):
        super(ResFF, self).__init__()
        self.ffn = nn.Sequential(nn.Linear(input_size, inner_size),
                                 nn.ReLU(),
                                 nn.Linear(inner_size, input_size))
        self.layer_norm = nn.LayerNorm(input_size)
        self.dropout = nn.Dropout(res_dropout)

    def forward(self, input):
        output = self.ffn(input.view(-1, input.size(-1))).view_as(input)
        output = self.dropout(output)
        return self.layer_norm(output + input)

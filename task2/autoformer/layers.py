"""Autoformer building blocks.

SeriesDecomposition, delay_scores and aggregate_delays are the Task 1 implementations
(task1/Assignment1.ipynb cells 13, 19, 22), copied unchanged. The layer order of the
encoder and decoder follows Wu et al. (2021) §3 and thuml/Autoformer
(layers/AutoCorrelation.py, layers/Autoformer_EncDec.py); see CODE_ATTRIBUTION.md.

Delay convention (both halves use it): R(tau) = sum_t q[t] * k[t - tau] and
z[t] = sum_j w_j * v[t - tau_j], indices mod L.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class SeriesDecomposition(nn.Module):
    def __init__(self, kernel):
        super().__init__()
        if kernel < 1 or kernel % 2 == 0:
            raise ValueError("kernel must be positive and odd")
        self.kernel = kernel

    def forward(self, x):                       # x [B,L,C] -> (remainder, trend), each [B,L,C]
        q = self.kernel // 2
        padded = torch.cat([x[:, :1].repeat(1, q, 1), x, x[:, -1:].repeat(1, q, 1)], dim=1)  # [B,L+2q,C]
        trend = padded.unfold(1, self.kernel, 1).mean(-1)  # [B,L,C,kernel] -> [B,L,C]
        return x - trend, trend


def delay_scores(queries, keys):
    # queries, keys: [B,heads,features,L] -> scores [B,L], one per delay tau = 0 .. L-1.
    queries = queries - queries.mean(-1, keepdim=True)
    keys = keys - keys.mean(-1, keepdim=True)
    spectrum = torch.fft.rfft(queries, dim=-1) * torch.fft.rfft(keys, dim=-1).conj()
    return torch.fft.irfft(spectrum, n=queries.shape[-1], dim=-1).mean((1, 2))


def aggregate_delays(values, delays, weights):
    # values: [B,heads,features,L]; delays, weights: [B,K] -> [B,heads,features,L]
    batch, heads, features, length = values.shape
    index = (torch.arange(length, device=values.device) - delays[..., None]) % length  # [B,K,L]
    index = index[:, :, None, None, :].expand(-1, -1, heads, features, -1)       # [B,K,h,f,L]
    shifted = values[:, None].expand(-1, delays.shape[1], -1, -1, -1).gather(-1, index)
    return (shifted * weights[:, :, None, None, None]).sum(1)                      # [B,h,f,L]


class AutoCorrelation(nn.Module):
    """Multi-head Auto-Correlation: FFT delay scores -> top-k delays -> softmax -> roll-and-sum.

    k = floor(factor * ln L). In training the k delays are shared across the batch (chosen
    from the batch-mean score); at inference each example picks its own, as in the paper.
    """

    def __init__(self, d_model, n_heads, factor):
        super().__init__()
        self.n_heads, self.factor = n_heads, factor
        self.query, self.key, self.value = (nn.Linear(d_model, d_model) for _ in range(3))
        self.out = nn.Linear(d_model, d_model)

    def forward(self, q_in, kv_in):             # q_in [B,L,d], kv_in [B,S,d] -> [B,L,d]
        batch, length, width = q_in.shape
        src = kv_in.shape[1]
        if src < length:                        # match key/value length to the query length
            kv_in = F.pad(kv_in, (0, 0, 0, length - src))
        else:
            kv_in = kv_in[:, :length]
        shape = (batch, length, self.n_heads, width // self.n_heads)
        q = self.query(q_in).view(shape).permute(0, 2, 3, 1)    # [B,h,d_h,L]
        k = self.key(kv_in).view(shape).permute(0, 2, 3, 1)
        v = self.value(kv_in).view(shape).permute(0, 2, 3, 1)
        scores = delay_scores(q, k)                              # [B,L]
        top_k = max(1, int(self.factor * math.log(length)))
        if self.training:
            delays = scores.mean(0).topk(top_k).indices.expand(batch, -1)  # [B,K] shared
        else:
            delays = scores.topk(top_k, dim=-1).indices                     # [B,K] per example
        weights = scores.gather(-1, delays).softmax(-1)
        mixed = aggregate_delays(v, delays, weights)             # [B,h,d_h,L]
        return self.out(mixed.permute(0, 3, 1, 2).reshape(batch, length, width))


class SeasonalLayerNorm(nn.Module):
    """LayerNorm, then subtract the temporal mean (the official my_Layernorm)."""

    def __init__(self, d_model):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        x = self.norm(x)
        return x - x.mean(1, keepdim=True)


class TokenEmbedding(nn.Module):
    """Circular Conv1d over time; no positional or calendar embedding (none exists here)."""

    def __init__(self, c_in, d_model, dropout):
        super().__init__()
        self.conv = nn.Conv1d(c_in, d_model, 3, padding=1, padding_mode="circular", bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):                       # [B,L,c_in] -> [B,L,d]
        return self.dropout(self.conv(x.transpose(1, 2)).transpose(1, 2))


def _feed_forward(d_model, d_ff):
    return nn.Sequential(nn.Conv1d(d_model, d_ff, 1, bias=False), nn.GELU(),
                         nn.Conv1d(d_ff, d_model, 1, bias=False))


class EncoderLayer(nn.Module):
    """AC + residual -> decomp (keep seasonal) -> FF + residual -> decomp."""

    def __init__(self, d_model, n_heads, d_ff, kernel, factor, dropout):
        super().__init__()
        self.attn = AutoCorrelation(d_model, n_heads, factor)
        self.ff = _feed_forward(d_model, d_ff)
        self.decomp1, self.decomp2 = SeriesDecomposition(kernel), SeriesDecomposition(kernel)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x, _ = self.decomp1(x + self.dropout(self.attn(x, x)))
        y = self.dropout(self.ff(x.transpose(1, 2)).transpose(1, 2))
        x, _ = self.decomp2(x + y)
        return x


class DecoderLayer(nn.Module):
    """Self-AC -> decomp -> cross-AC -> decomp -> FF -> decomp; the three trends are summed
    and projected to the target channel, to be accumulated into the trend stream."""

    def __init__(self, d_model, n_heads, d_ff, kernel, factor, dropout, c_out=1):
        super().__init__()
        self.self_attn = AutoCorrelation(d_model, n_heads, factor)
        self.cross_attn = AutoCorrelation(d_model, n_heads, factor)
        self.ff = _feed_forward(d_model, d_ff)
        self.decomp1, self.decomp2, self.decomp3 = (SeriesDecomposition(kernel) for _ in range(3))
        self.trend_proj = nn.Conv1d(d_model, c_out, 3, padding=1, padding_mode="circular", bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, cross):
        x, t1 = self.decomp1(x + self.dropout(self.self_attn(x, x)))
        x, t2 = self.decomp2(x + self.dropout(self.cross_attn(x, cross)))
        y = self.dropout(self.ff(x.transpose(1, 2)).transpose(1, 2))
        x, t3 = self.decomp3(x + y)
        trend = self.trend_proj((t1 + t2 + t3).transpose(1, 2)).transpose(1, 2)
        return x, trend

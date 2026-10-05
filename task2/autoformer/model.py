"""Encoder-decoder Autoformer (Wu et al. 2021, §3) for one target channel plus optional covariates."""
import torch
import torch.nn as nn

from task2.guards import assert_decoder_future_target_masked
from task2.autoformer.layers import (DecoderLayer, EncoderLayer, SeasonalLayerNorm,
                                     SeriesDecomposition, TokenEmbedding)


class Autoformer(nn.Module):
    """x_enc [B,seq_len,1+n_past] (target first) and optional dec_exog [B,label_len+pred_len,n_future]
    -> forecast [B,pred_len,1] in scaled units."""

    def __init__(self, n_past=0, n_future=0, label_len=168, pred_len=168, d_model=32, n_heads=4,
                 e_layers=2, d_layers=1, d_ff=64, moving_avg=25, factor=1, dropout=0.1):
        super().__init__()
        self.label_len, self.pred_len = label_len, pred_len
        self.decomp = SeriesDecomposition(moving_avg)
        self.enc_embedding = TokenEmbedding(1 + n_past, d_model, dropout)
        self.dec_embedding = TokenEmbedding(1 + n_future, d_model, dropout)
        layer_args = (d_model, n_heads, d_ff, moving_avg, factor, dropout)
        self.encoder = nn.ModuleList(EncoderLayer(*layer_args) for _ in range(e_layers))
        self.decoder = nn.ModuleList(DecoderLayer(*layer_args) for _ in range(d_layers))
        self.enc_norm, self.dec_norm = SeasonalLayerNorm(d_model), SeasonalLayerNorm(d_model)
        self.projection = nn.Linear(d_model, 1)

    def forward(self, x_enc, dec_exog=None):
        target = x_enc[..., :1]
        seasonal_init, trend_init = self.decomp(target)
        batch = target.shape[0]
        zeros = target.new_zeros(batch, self.pred_len, 1)
        mean = target.mean(1, keepdim=True).expand(-1, self.pred_len, -1)
        dec_target = torch.cat([seasonal_init[:, -self.label_len:], zeros], dim=1)
        trend = torch.cat([trend_init[:, -self.label_len:], mean], dim=1)
        assert_decoder_future_target_masked(dec_target)
        dec_in = dec_target if dec_exog is None else torch.cat([dec_target, dec_exog], dim=-1)

        enc = self.enc_embedding(x_enc)
        for layer in self.encoder:
            enc = layer(enc)
        enc = self.enc_norm(enc)

        x = self.dec_embedding(dec_in)
        for layer in self.decoder:
            x, residual_trend = layer(x, enc)
            trend = trend + residual_trend
        seasonal = self.projection(self.dec_norm(x))
        return (trend + seasonal)[:, -self.pred_len:]


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

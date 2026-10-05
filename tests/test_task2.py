import math

import numpy as np
import pytest
import torch

from task2.autoformer.layers import AutoCorrelation, SeriesDecomposition, aggregate_delays, delay_scores
from task2.autoformer.model import Autoformer
from task2.data import metrics, val_origins
from task2.guards import assert_decoder_future_target_masked, assert_submission_string


def test_decomposition_worked_example():
    x = torch.tensor([1., 2., 3., 10., 5.]).view(1, 5, 1)
    seasonal, trend = SeriesDecomposition(3)(x)
    assert torch.allclose(trend.flatten(), torch.tensor([4 / 3, 2, 5, 6, 20 / 3]))
    assert torch.allclose(seasonal + trend, x)


def test_delay_scores_match_direct_loop_and_find_period():
    L = 96
    t = torch.arange(L, dtype=torch.float32)
    v = torch.sin(2 * math.pi * t / 24).view(1, 1, 1, L)
    s = delay_scores(v, v)[0]
    vc = v.flatten() - v.mean()
    direct = torch.stack([(vc * vc.roll(tau)).sum() for tau in range(L)])   # sum_t q[t] k[t - tau]
    assert torch.allclose(s, direct, atol=1e-4)
    assert int(s[1:].argmax()) + 1 in (24, 48, 72)


def test_aggregate_direction():
    v = torch.arange(5.).view(1, 1, 1, 5)
    z = aggregate_delays(v, torch.tensor([[1]]), torch.tensor([[1.]]))
    assert z.flatten().tolist() == [4., 0., 1., 2., 3.]                   # z[t] = v[t - 1]


def test_model_has_both_mechanisms_and_no_dot_product_attention():
    m = Autoformer(n_past=10, n_future=10)
    for layer in list(m.encoder) + list(m.decoder):
        assert any(isinstance(x, SeriesDecomposition) for x in layer.modules())
        assert any(isinstance(x, AutoCorrelation) for x in layer.modules())
    assert not any(isinstance(x, torch.nn.MultiheadAttention) for x in m.modules())
    out = m(torch.randn(2, 336, 11), torch.randn(2, 336, 10))
    assert out.shape == (2, 168, 1)


def test_masking_guard():
    assert_decoder_future_target_masked(torch.cat([torch.randn(2, 168, 1), torch.zeros(2, 168, 1)], 1))
    with pytest.raises(AssertionError):
        assert_decoder_future_target_masked(torch.randn(2, 336, 1))


def test_validation_origins_and_smape_zero():
    o = val_origins()
    assert len(o) == 78 and o[0] == 41641 and o[-1] + 167 <= 43656
    m = metrics(np.zeros((1, 168)), np.zeros((1, 168)))
    assert m["smape"][0] == 0 and m["rmse"][0] == 0


def test_submission_string_guard():
    v = np.linspace(1, 2, 168)
    assert_submission_string(", ".join(f"{x:.10g}" for x in v), v)
    with pytest.raises(AssertionError):
        assert_submission_string(", ".join(f"{x:.10g}" for x in v[::-1]), v)

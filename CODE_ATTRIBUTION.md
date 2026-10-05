# Code attribution

## Task 1
`task1/Assignment1.ipynb` and `task1/harness/` are the course's starter code. Only cells 6, 13, 22 and 33 were written for this submission.

## Task 2: the Autoformer
The Autoformer in `task2/autoformer/` was written for this assignment. It follows the architecture of

- Wu, H., Xu, J., Wang, J., Long, M. (2021). *Autoformer: Decomposition Transformers with Auto-Correlation for Long-Term Series Forecasting.* NeurIPS 2021, §3.
- The authors' reference implementation, https://github.com/thuml/Autoformer (`layers/AutoCorrelation.py`, `layers/Autoformer_EncDec.py`, `models/Autoformer.py`). It was used to check layer order and decoder initialisation. No code was copied from it.

| Lines | Component | Source |
|---|---|---|
| `task2/autoformer/layers.py` 18–46 | `SeriesDecomposition`, `delay_scores`, `aggregate_delays` | Task 1 cells 13, 19 and 22 of this repo, code copied unchanged, comments shortened (cell 19, `delay_scores`, is supplied course code) |
| `task2/autoformer/layers.py` 49–81 | `AutoCorrelation`: top-k with k = floor(c ln L), batch-shared delays in training and per-example delays at inference, truncate or zero-pad keys to the query length | paper §3.2; selection and padding as in `AutoCorrelation.py` (`time_delay_agg_training`, `time_delay_agg_inference`, `AutoCorrelation.forward`) |
| `task2/autoformer/layers.py` 84–93 | `SeasonalLayerNorm`: LayerNorm minus the temporal mean | `my_Layernorm` in `Autoformer_EncDec.py` |
| `task2/autoformer/layers.py` 96–105 | `TokenEmbedding`: circular Conv1d, no positional or time embedding | `TokenEmbedding` / `DataEmbedding_wo_pos` in the reference repo, without time marks |
| `task2/autoformer/layers.py` 108–149 | encoder and decoder layers: sub-block order, the decoder's three trends summed and projected by a circular Conv1d | paper §3.1, Fig. 1; `EncoderLayer`, `DecoderLayer` in `Autoformer_EncDec.py` |
| `task2/autoformer/model.py` 10–48 | decoder initialisation (seasonal: last `label_len` seasonal values then zeros; trend: last `label_len` trend values then the input mean) and output = trend + projected seasonal | paper §3.1 eq. 2; `models/Autoformer.py` `forward` |

Where this code deviates from the reference implementation:
- **Delay direction and centring:** it uses the Task 1 convention, R(tau) = sum_t q[t] k[t - tau], with q and k mean-centred in time, and z[t] = sum_j w_j v[t - tau_j].
- **Target-only decoder streams:** the trend stream and the decoder's seasonal target carry only the target channel. Exogenous covariates enter the encoder input (past) or the decoder input (future) as extra channels.

## Other references
- Validation design, metrics and baselines follow the assignment PDF §2.5–2.6.
- DLinear (Zeng et al. 2023) is cited in the report only; it is not implemented.

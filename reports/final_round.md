# Final selection round

`python -m src.experiment_final`. Folds B, C, D (Phase 0 backtest) and E = Phase 1 (train on Phase 0, forecast 2023-10-09 → 2024-01-01). Fold A shown for reference only (prior-year holiday is smoothed 2021 data). Phase 2 not read. Runtime 230 s.

Cell = VN1 score (signed bias). Candidates:

- `lgbm_w0.5`, `lgbm_w0.7`: w × frozen LightGBM + (1 − w) × blend_0.25
- `routing`: frozen LightGBM, except top-1% volume and lumpy series (segments from data ≤ origin) which get 0.5 × LightGBM + 0.5 × blend_0.25. Series routed per fold: A: 1,369, B: 1,460, C: 1,592, D: 1,841, E: 1,934.

| candidate | A | B | C | D | E | mean B–E | beats frozen on |
|---|---|---|---|---|---|---|---|
| lgbm (frozen) | 0.6235 (-7.0%) | 0.4904 (-3.9%) | 0.4478 (-1.4%) | 0.4739 (-1.8%) | 0.5712 (+5.9%) | 0.4958 | – |
| lgbm_w0.5 | 0.6907 (-13.3%) | 0.5363 (+2.6%) | 0.4666 (-1.5%) | 0.4896 (-2.9%) | 0.5086 (+0.5%) | 0.5003 | 1/4 |
| lgbm_w0.7 | 0.6599 (-10.8%) | 0.4804 (+0.0%) | 0.4554 (-1.5%) | 0.4807 (-2.5%) | 0.5290 (+2.7%) | 0.4864 | 2/4 |
| routing | 0.6488 (-9.2%) | 0.4969 (-2.2%) | 0.4640 (-2.4%) | 0.4788 (-2.5%) | 0.5525 (+4.6%) | 0.4981 | 1/4 |
| blend_0.25 (baseline) | 0.7893 (-19.5%) | 0.7010 (+9.0%) | 0.5097 (-1.5%) | 0.5284 (-4.0%) | 0.5816 (-4.9%) | 0.5802 | – |

Rule: lowest mean over B, C, D, E, and beats frozen LightGBM on ≥ 3 of 4 folds.
Best candidate by mean: `lgbm_w0.7`. Eligible: none. **Chosen: `lgbm`.**

Determinism check: fold E LightGBM vs the forecast from the one-off Phase 1 run, max abs difference 0.00e+00.

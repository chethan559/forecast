# Forecast layer summary: VN1

Weekly demand forecasts, 13 weeks ahead, for 15,053 client × warehouse × product series.
Built on Phase 0 (2020-07-06 → 2023-10-02), validated on Phase 1 (2023-10-09 → 2024-01-01),
tested once on the sealed Phase 2 (2024-01-08 → 2024-04-01).

## Final method

One global LightGBM, direct multi-horizon (a row per series × origin × horizon 1–13). Full spec and evidence:
`reports/final_config.md`.

- **Training:** origins every 2 weeks from 2021-07-05; every target must fall before the forecast origin.
  Each origin's features are recomputed from data up to that origin only, so no feature can see the future.
- **Target:** units scaled by max(13-week mean, 52-week mean). Tweedie loss, rows weighted by scale, so the
  loss counts units the way the VN1 score does.
- **Features (45):** lags, rolling statistics, trends, intermittency, age, last year's same weeks, last
  year's uplift ratio (series and client), calendar and US holiday weeks, product/client context, relative
  price, data-quality shares, demand type at origin, horizon. Client and warehouse IDs are **not** used.
- **Level-shift restart:** when most of a warehouse's or client's series change level in the same week, the
  history before that week is ignored, and training rows that cross the event are dropped.
- **Cap:** forecast ≤ 3 × the series' highest week in the last 52 weeks. A safety net only.
- **No** bias multiplier, **no** blending with the baseline, default LightGBM parameters (not tuned).

## Scores (VN1 metric: (Σ|F−A| + |Σ(F−A)|) / ΣA, lower is better)

| | frozen LightGBM | blend_0.25 baseline | moving average 13 |
|---|---|---|---|
| Backtest folds B–D mean (Phase 0) | 0.471 | 0.580 | 0.583 |
| Holiday fold A (prior year smoothed) | 0.624 | 0.789 | 0.835 |
| **Phase 1** (Oct–Dec 2023, holiday) | **0.571** (accuracy 0.512, bias 0.059, +5.9%) | 0.582 | 0.634 |
| **Phase 2** (Jan–Apr 2024, sealed) | **0.488** (accuracy 0.445, bias 0.043, −4.3%) | 0.784 | – |

VN1 Phase 2 leaderboard winner: **0.4637**. We are 0.024 behind it, but the comparison is harsher on us. The
official scoring removed stockout weeks from the actuals; we have no stockout data, so every zero week counts
against us.

Phase 2 by segment (LightGBM): top 1% of series 0.454 (−9.8%), top 1–10% 0.435 (−2.8%), rest 0.744 (+3.4%);
smooth 0.396, erratic 0.634, lumpy 1.155, intermittent 1.470.
The baseline over-forecast Phase 2 by 19%: its moving average carried the Oct–Dec holiday level (~300k
units/week) into a quarter that ran at ~250k. LightGBM handled the post-holiday drop.

## Comparison with the VN1 competition (Phase 2)

| entry | VN1 score | our gap |
|---|---|---|
| 1st place | 0.4637 | +0.024 |
| TimeGPT, zero-shot (not an official entry) | 0.4651 | +0.023 |
| 2nd place | 0.4657 | +0.022 |
| 3rd place | 0.4758 | +0.012 |
| 4th place | 0.4774 | +0.011 |
| 5th place | 0.4808 | +0.007 |
| **our frozen LightGBM** | **0.4879** | – |

Source: the results table in Nixtla's nixtlar VN1 vignette, which gives places only, no team names. Scores
below 5th are not published, so our exact rank is unknown: just outside the top 5 if the scoring is
identical. We are 5.2% behind the winner and 1.5% behind 5th.

- **Scoring may differ.** The vignette does not define the metric's handling of stockout weeks. Our Phase 2
  actuals have no missing weeks (all 195,689 series-weeks are scored, zeros included). If the official
  scoring removed stockout weeks, as noted when the task was set, our score is harsher than the
  leaderboard's. Not verified; the competition page or Nixtla's reproduction scripts would settle it.
- **Method gap.** The top five "used multiple models and carefully crafted features" with fine-tuning or
  manual adjustment. Ours is one LightGBM with default parameters (100 trees), not tuned, and every choice
  was fixed before Phase 2 was read.
- **TimeGPT as a yardstick.** Zero-shot, no price data, leading zeros removed only: 0.4651. Most of our gap
  is likely model capacity and tuning, not data preparation.
- **Likely sources of the gap:** the top 1% of series (38.5% of units, −9.8% bias), warehouse 68's mid-March
  reopening (most of the last three weeks' under-forecast), and no tuning or ensembling.

## Main lessons

1. **The older history is smoothed.** Before 2022-08-29 the data looks like monthly totals spread evenly
   over weeks (runs of identical values, twice the repeat rate of true weekly data). It still carries level
   and seasonality, so we keep it (full history beat weekly-only). But any holiday shape learned from 2021 is
   blurred.
2. **Warehouses can change level overnight.** Warehouse 251 (client 2) jumped about 500× on 2022-11-07.
   With warehouse as a feature, LightGBM memorised it as a multiplier and blew up (fold B scores 2.3 and 3.2).
   Removing the ID features and adding group level-shift detection with restart fixed it.
   - **Warehouse 68 (client 38), the same mechanism in Phase 2:** it collapsed from ~20k to ~20 units/week
     in mid-December 2023. The restart correctly forecast near-zero, which was right for 10 of 13 weeks.
   - **It then reopened in mid-March** (15–26k/week). No model could foresee this, and it explains most of
     LightGBM's under-forecast in the last three Phase 2 weeks.
3. **The backtest's holiday fold was biased.** Fold A's prior-year holiday is the smoothed 2021 one. That
   crippled the seasonal-naive part of the baseline (0.789), so the backtest suggested LightGBM was ~0.17
   better on holidays. On Phase 1 the gap was 0.010. Model selection after that excluded fold A and used
   Phase 1 as a fold.
4. **The loss has to match the metric.** Unweighted tweedie on scaled targets chased series with a tiny scale
   that restarted. Weighting rows by scale was the single biggest model improvement.
5. **Flags computed on the full history leak.** Stockout runs, run lengths and price-shift locations use
   weeks after a given date. Features recompute them at every origin; tests check that scrambling the future
   changes nothing.

## Known weaknesses

- **Post-peak over-forecast.** In Phase 1, LightGBM caught the Thanksgiving jump (−6%) but stayed high
  afterwards: +19% early December, +19% Christmas, +23% New Year week. A 0.7 LightGBM / 0.3 baseline blend
  would have scored 0.529 on Phase 1. It was rejected because it lost on 2 of the 3 non-holiday folds.
- **Top sellers.** The top 1% of series by volume (35–39% of units):
  - Phase 1: worse than the baseline (0.476 vs 0.456, +7.6% over-forecast).
  - Phase 2: 9.8% under-forecast.
  - Their errors dominate the score, and they get no special treatment.
- **Lumpy and intermittent series.** Phase 1 lumpy 1.63 (baseline 1.38); Phase 2 intermittent 1.47. A
  zero-inflated or Croston-type model for these segments has not been tried.
- **Level shifts at the origin.** The restart helps after a structural change, but a closure that later
  reopens (warehouse 68) or a jump that happens during the forecast window cannot be anticipated.
- **Untuned model.** Default LightGBM parameters, 100 trees. No hyperparameter search was done.
- **Stockouts are only flagged.** Suspected-stockout zeros are still in the training targets, which pulls
  forecasts down for affected series.

## Reproduce

```bash
python -m src.run_cleaning        # cleaned data + data quality report
python -m src.backtest            # backtest grid (reports/backtest.md)
python -m src.experiment_final    # final selection round (reports/final_round.md)
python -m src.run_phase1          # one-off Phase 1 score (frozen; refuses to re-run)
python -m src.run_phase2          # one-off Phase 2 score (frozen; refuses to re-run)
```

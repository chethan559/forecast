# Generalisation: the VN1 engine on M5

**Question:** does the VN1 forecasting engine, with every setting left as frozen on VN1, still beat simple
baselines on a dataset it has never seen?
**Answer: yes.** On M5's 13 sealed weeks, run once, it scored **0.452** against **0.554** for the best
baseline (blend_0.25), 18% better, and it was better in every volume band. Its 80% intervals covered 77.8%.
In the M5 backtest the lead was much smaller than on VN1:
- **VN1 backtest:** engine 0.471 vs blend 0.580, 19% better.
- **M5 backtest:** engine 0.523 vs blend 0.522, a tie.

The engine always had the lowest WAPE. The two folds it lost, it lost on bias: it under-forecasts a growing
business.

## Setup

| | VN1 | M5 |
|---|---|---|
| data | 15,053 client × warehouse × product series, weekly | 30,490 item × store series (Walmart, 10 stores), daily → Monday weeks |
| modelled | all series | fixed random sample of 7,000 series (seed 0), chosen before any result (memory, see below) |
| engine | frozen VN1 config | the same: no model, feature or threshold change (`reports/m5_config.md`) |
| backtest folds | B, C, D (chosen by hand; A dropped: its previous year was smoothed) | F1–F4, automatic rolling 13-week folds back from the seal (`src/folds.py`); F4 is the holiday fold, kept because its previous year is clean |
| held out, scored once | E = Phase 1 (2023-10-09 → 2023-12-25) and P2 = Phase 2 (2024-01-08 → 2024-04-01) | last 13 weeks (2016-02-22 → 2016-05-16), sealed right after the adapter |
| best baseline | blend_0.25 (chosen on VN1) | blend_0.25 (lowest mean M5 backtest score, chosen at the freeze) |

- **Metrics:**
  - `score` = VN1 score (WAPE + |bias|).
  - `wape` = Σ|F−A| / ΣA.
  - `bias` = Σ(F−A) / ΣA (negative means an under-forecast).
  - `coverage_80` = share of series-weeks inside the engine's conformal [p10, p90].
- **VN1 rows:** the saved, already-scored forecasts, with baselines recomputed at each origin.
- **VN1 Phase 2:** no intervals were produced for it, so it has no coverage.
- **M5 leaderboard:** not comparable. It uses WRMSSE on 28 daily forecasts over 12 aggregation levels.

## VN1 score by fold

| dataset | fold | origin | engine | moving_avg_13 | seasonal_naive | blend_0.25 |
|---|---|---|---|---|---|---|
| VN1 | B | 2023-01-30 | 0.4904 | 0.7645 | 0.7257 | 0.7010 |
| VN1 | C | 2023-04-03 | 0.4478 | 0.4832 | 0.7276 | 0.5097 |
| VN1 | D | 2023-07-03 | 0.4739 | 0.5004 | 0.7742 | 0.5284 |
| VN1 | E | 2023-10-02 | 0.5712 | 0.6339 | 0.7599 | 0.5816 |
| VN1 | P2 | 2024-01-01 | 0.4879 | 0.8132 | 0.8519 | 0.7844 |
| M5 | F1 | 2015-02-16 | 0.5112 | 0.5726 | 0.7212 | 0.5468 |
| M5 | F2 | 2015-05-18 | 0.5712 | 0.5722 | 0.6939 | 0.5568 |
| M5 | F3 | 2015-08-17 | 0.4886 | 0.4882 | 0.7053 | 0.4969 |
| M5 | F4 | 2015-11-16 | 0.5209 | 0.5274 | 0.7177 | 0.4857 |
| M5 | test | 2016-02-15 | 0.4516 | – | – | 0.5535 |

## Summary

| dataset | set | method | score | wape | bias | coverage_80 |
|---|---|---|---|---|---|---|
| VN1 | backtest mean | engine | 0.4707 | 0.4470 | -2.4% | 79.2% |
| VN1 | backtest mean | moving_avg_13 | 0.5827 | 0.5283 | +3.2% | – |
| VN1 | backtest mean | seasonal_naive | 0.7425 | 0.6941 | -4.8% | – |
| VN1 | backtest mean | blend_0.25 | 0.5797 | 0.5313 | +1.2% | – |
| VN1 | E | engine | 0.5712 | 0.5119 | +5.9% | 71.6% |
| VN1 | E | moving_avg_13 | 0.6339 | 0.5476 | -8.6% | – |
| VN1 | E | seasonal_naive | 0.7599 | 0.6965 | +6.3% | – |
| VN1 | E | blend_0.25 | 0.5816 | 0.5327 | -4.9% | – |
| VN1 | P2 | engine | 0.4879 | 0.4446 | -4.3% | – |
| VN1 | P2 | moving_avg_13 | 0.8132 | 0.6087 | +20.4% | – |
| VN1 | P2 | seasonal_naive | 0.8519 | 0.7079 | +14.4% | – |
| VN1 | P2 | blend_0.25 | 0.7844 | 0.5951 | +18.9% | – |
| M5 | backtest mean | engine | 0.5230 | 0.4651 | -5.8% | 80.9% |
| M5 | backtest mean | moving_avg_13 | 0.5401 | 0.5031 | -2.4% | – |
| M5 | backtest mean | seasonal_naive | 0.7095 | 0.6549 | -3.9% | – |
| M5 | backtest mean | blend_0.25 | 0.5216 | 0.4940 | -2.7% | – |
| M5 | test | engine | 0.4516 | 0.4074 | -4.4% | 77.8% |
| M5 | test | blend_0.25 | 0.5535 | 0.4482 | -10.5% | – |

## By volume band (mean over every fold and test, incl. held-out)

| dataset | band | method | score | bias | coverage_80 | unit_share |
|---|---|---|---|---|---|---|
| VN1 | top 1% | engine | 0.3703 | +0.5% | 96.2% | 36.2% |
| VN1 | top 1-10% | engine | 0.4761 | -0.1% | 79.0% | 40.7% |
| VN1 | rest 90% | engine | 0.7299 | +2.8% | 78.4% | 21.4% |
| VN1 | first sale after origin | engine | 2.0000 | -100.0% | 45.5% | 1.7% |
| VN1 | top 1% | moving_avg_13 | 0.4512 | +5.7% | – | 36.2% |
| VN1 | top 1-10% | moving_avg_13 | 0.6849 | +8.0% | – | 40.7% |
| VN1 | rest 90% | moving_avg_13 | 0.9669 | +4.3% | – | 21.4% |
| VN1 | first sale after origin | moving_avg_13 | 2.0000 | -100.0% | – | 1.7% |
| VN1 | top 1% | seasonal_naive | 0.6164 | -2.1% | – | 36.2% |
| VN1 | top 1-10% | seasonal_naive | 0.7837 | +5.3% | – | 40.7% |
| VN1 | rest 90% | seasonal_naive | 1.0284 | +7.7% | – | 21.4% |
| VN1 | first sale after origin | seasonal_naive | 2.0000 | -100.0% | – | 1.7% |
| VN1 | top 1% | blend_0.25 | 0.4404 | +3.8% | – | 36.2% |
| VN1 | top 1-10% | blend_0.25 | 0.6521 | +7.3% | – | 40.7% |
| VN1 | rest 90% | blend_0.25 | 0.9364 | +5.1% | – | 21.4% |
| VN1 | first sale after origin | blend_0.25 | 2.0000 | -100.0% | – | 1.7% |
| M5 | top 1% | engine | 0.3714 | -4.5% | 97.3% | 15.0% |
| M5 | top 1-10% | engine | 0.4374 | -4.1% | 79.6% | 34.5% |
| M5 | rest 90% | engine | 0.5972 | -6.3% | 80.3% | 50.3% |
| M5 | first sale after origin | engine | 2.0000 | -100.0% | 13.6% | 0.2% |
| M5 | top 1% | moving_avg_13 | 0.4655 | +2.5% | – | 15.0% |
| M5 | top 1-10% | moving_avg_13 | 0.4834 | +1.5% | – | 34.5% |
| M5 | rest 90% | moving_avg_13 | 0.6425 | -5.9% | – | 50.3% |
| M5 | first sale after origin | moving_avg_13 | 2.0000 | -100.0% | – | 0.2% |
| M5 | top 1% | seasonal_naive | 0.4809 | +5.3% | – | 15.0% |
| M5 | top 1-10% | seasonal_naive | 0.6556 | +0.4% | – | 34.5% |
| M5 | rest 90% | seasonal_naive | 0.8563 | -9.1% | – | 50.3% |
| M5 | first sale after origin | seasonal_naive | 2.0000 | -100.0% | – | 0.2% |
| M5 | top 1% | blend_0.25 | 0.4210 | +0.7% | – | 15.0% |
| M5 | top 1-10% | blend_0.25 | 0.4570 | -0.1% | – | 34.5% |
| M5 | rest 90% | blend_0.25 | 0.6510 | -8.2% | – | 50.3% |
| M5 | first sale after origin | blend_0.25 | 2.0000 | -100.0% | – | 0.2% |

## Where the engine beat the baselines

- **Held-out weeks, both datasets:**

  | | engine | blend_0.25 |
  |---|---|---|
  | M5 test | 0.452 | 0.554 |
  | VN1 Phase 1 | 0.571 | 0.582 |
  | VN1 Phase 2 | 0.488 | 0.784 |

  The M5 test win is spread across every band: 0.298 vs 0.405 for the top 1%, 0.382 vs 0.443 for the top
  1–10%, and 0.541 vs 0.667 for the rest. Blend_0.25 lagged the level in early 2016. Its MA13 half averages
  the Christmas and January dip, and its seasonal half reads the lower 2015 level, so it ran 10.5% under.
- **WAPE, every M5 fold:**

  | fold | engine | blend_0.25 |
  |---|---|---|
  | F1 | 0.487 | 0.524 |
  | F2 | 0.475 | 0.489 |
  | F3 | 0.447 | 0.477 |
  | F4 | 0.452 | 0.485 |

  Series by series, it is the most accurate method in every fold.
- **Every volume band** (mean over folds and the held-out weeks), on both datasets. The largest margin is on the top 1%: M5 0.371 vs
  0.421 for blend_0.25, VN1 0.370 vs 0.440.
- **Seasonal naive** loses badly on both datasets (M5 0.71, VN1 0.74). Last year's week alone is too noisy at
  item × store level.

## Where it didn't

- **Lost two M5 folds on score:**

  | fold | engine | blend_0.25 | engine bias |
  |---|---|---|---|
  | F2 (May–Aug 2015) | 0.571 | 0.557 | −9.7% |
  | F4 (holiday, Nov 2015–Feb 2016) | 0.521 | 0.486 | −6.9% |

  Its WAPE was lower in both, so the losses are the bias term alone.
- **Under-forecast in every M5 fold and in the test:** −2.4%, −9.7%, −4.2%, −6.9% and −4.4%. The sample grew
  6.5% in 2015 and about 8% in early 2016. The model learns scaled levels, and the trees don't extrapolate
  growth. On VN1 the backtest bias was small (−2.4%), and the bias multiplier was rejected there.
- **The week after Christmas (2015-12-28):** forecast 51k units, actual 60k. The engine called the right dips
  in Thanksgiving week and Christmas week, but repeated the dip a week too long. A grocery and general
  retailer like Walmart has a New Year rebound that VN1's holiday pattern doesn't show.
- **Rising sales late in the window:** under-forecast in February 2016 by about 12% (61k vs 70k).
- **Series that hadn't sold yet:** both methods score 2.0 on series with no sales before the origin, on both
  datasets. That's 0.2% of M5 units and 1.7% of VN1.

## VN1 assumptions that didn't transfer

1. **Smoothed era.** VN1's early data are monthly totals spread over weeks. The changepoint detector found a
   clear drop there (20.0% repeat rate before vs 10.1% after).
   - It always returns a date, and on M5 it picked 2014-03-17, where the repeat rate *rises* (10.9% → 12.0%)
     with the slow growth of low-volume items.
   - 121,864 M5 weeks (1.9%) were flagged as "smoothed" by mistake.
   - Impact here is small: it only feeds `smoothed_share_*` at origins before 2015-03.
   - The detector needs a direction and size test, e.g. accept a switch only if the rate before is ≥ 1.5× the
     rate after. Not changed in this stage.
2. **Level shifts.** The VN1 rule (> 10× or < 1/10 change, level ≥ 5) was built for structural breaks like
   warehouse 251's 533× jump.
   - On M5 it fires 29,796 times in 43% of series, 71% downward: stockouts and delistings of slow items, not
     structural changes.
   - There were zero group events, so the restart logic never ran. Neutral for the forecasts, but the
     series-level flag is noise on M5.
3. **Bias.** VN1 needed no bias correction. M5 is a growing business, and the engine is consistently 4–7%
   under. A trend feature or a level correction would help, but choosing one would be M5 tuning, so it waits
   for a stage that's allowed to change the model.
4. **Holidays.** The US calendar (Thanksgiving, Cyber week, Christmas) fits Walmart, but the shape around it
   differs (the New Year rebound above). M5's own events aren't used: SNAP days, Super Bowl, Easter.
5. **Volume concentration.**
   - VN1's top 1% of series sell 36% of units. M5's top 1% sell 15%, and the bottom 90% sell 50%.
   - Accuracy on slow movers decides M5 (rest-90% score 0.60). On VN1 the top bands carry most of the score.
6. **Price rules and price shifts.** The VN1 price rule is about one VN1 client, so it doesn't carry over. M5
   prices are clean: only 86 series have a level shift.
7. **Scale.** The training rule (every 2 weeks from data start + 52 weeks, all series) grows with series ×
   history. All 30,490 M5 series would need about 33M training rows, more than 8 GB holds. M5 therefore used
   a 7,000-series sample. Its demand-type mix matches the full set within 0.5 points: smooth 51.2% vs 51.6%,
   intermittent 34.4% vs 34.1%. Running on the full catalogue needs streamed or subsampled training rows.

## What did transfer

- **One model, no retuning.** The same global LightGBM with the same features had the lowest WAPE on every fold and held-out
  period of both datasets, and won both datasets' held-out tests.
- **The cap at 3× the best recent week, and the point-in-time feature design.** Nothing blew up on M5.
- **Conformal intervals.**
  - Close to 80%: M5 80.9% in the backtest and 77.8% on the test; VN1 79.2% in the backtest.
  - Same weak spot as VN1: the top 1% are over-covered (M5 97%, VN1 96%). That band has fewer series (70 in
    the M5 sample) than a calibration group needs (200 rows), so it borrows the wider residuals of smaller
    series.
- **The holiday-fold rule, automated.** It reproduces the VN1 decision (a 2022 holiday fold would read
  smoothed 2021 data, so it's skipped; tested in `tests/test_m5_generalisation.py`). On M5 it keeps the 2015
  holiday fold.

## Files

- `reports/m5/data_quality.md`: M5 cleaning summary.
- `reports/m5/backtest.md`: folds, every fold × method × band.
- `reports/m5_config.md`: frozen config and every difference from VN1.
- `reports/m5/test.md`: sealed test, run once.
- Code: `src/experiment_m5.py` (prepare / backtest / freeze / test), `src/folds.py`, `src/sealed.py`,
  `src/report_generalisation.py`.
- Numbers: `outputs/m5/generalisation.csv`.

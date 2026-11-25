# Baselines: Phase 0 → Phase 1

Train on Phase 0 (cleaned, 15,053 series), forecast the 13 Phase 1 weeks (2023-10-09 to 2024-01-01), score with the VN1 metric. Lower is better; all-zero = 2.0. Phase 2 is not used. Command: `python -m src.run_baselines`.

`signed_bias` = sum(F − A) / sum(A): negative = under-forecast.

## Overall

| method | score | accuracy | bias | signed_bias |
|---|---|---|---|---|
| moving_avg_13 | 0.6339 | 0.5476 | 0.0863 | -8.6% |
| moving_avg_13_inactive_zero | 0.6339 | 0.5476 | 0.0863 | -8.6% |
| moving_avg_8 | 0.6369 | 0.5404 | 0.0965 | -9.6% |
| moving_avg_4 | 0.6400 | 0.5440 | 0.0960 | -9.6% |
| moving_avg_26 | 0.6587 | 0.5717 | 0.0870 | -8.7% |
| seasonal_naive | 0.7599 | 0.6965 | 0.0635 | +6.3% |
| zero | 2.0000 | 1.0000 | 1.0000 | -100.0% |

Winner: **moving_avg_13**.

`moving_avg_13_inactive_zero` scores exactly the same as `moving_avg_13`: a series with no sales in the last 13 weeks already has a 13-week mean of 0, so setting it to 0 changes nothing.

## Score by group

VN1 score computed inside each group. The `units` and `error share` columns are for the winner (`moving_avg_13`): share of Phase 1 units, and share of the total absolute error |F − A|.

### demand type

| demand_type | series | units | error share | signed bias (moving_avg_13) | moving_avg_13 | moving_avg_13_inactive_zero | moving_avg_8 | moving_avg_4 | moving_avg_26 | seasonal_naive | zero |
|---|---|---|---|---|---|---|---|---|---|---|---|
| smooth | 2,909 | 58.1% | 46.2% | -1.9% | 0.455 | 0.455 | 0.458 | 0.468 | 0.462 | 0.596 | 2.000 |
| erratic | 2,006 | 27.8% | 33.8% | -14.2% | 0.809 | 0.809 | 0.825 | 0.829 | 0.866 | 1.037 | 2.000 |
| intermittent | 7,236 | 6.1% | 6.6% | -15.8% | 0.754 | 0.754 | 0.745 | 0.696 | 0.811 | 1.241 | 2.000 |
| lumpy | 2,902 | 8.1% | 13.4% | -32.4% | 1.225 | 1.225 | 1.196 | 1.181 | 1.246 | 1.090 | 2.000 |

### is_new (< 52 weeks of history)

| is_new | series | units | error share | signed bias (moving_avg_13) | moving_avg_13 | moving_avg_13_inactive_zero | moving_avg_8 | moving_avg_4 | moving_avg_26 | seasonal_naive | zero |
|---|---|---|---|---|---|---|---|---|---|---|---|
| False | 10,228 | 77.4% | 73.0% | -9.6% | 0.612 | 0.612 | 0.624 | 0.635 | 0.614 | 0.806 | 2.000 |
| True | 4,825 | 22.6% | 27.0% | -5.3% | 0.708 | 0.708 | 0.682 | 0.702 | 0.812 | 0.708 | 2.000 |

### is_inactive (no sales in last 13 training weeks)

| is_inactive | series | units | error share | signed bias (moving_avg_13) | moving_avg_13 | moving_avg_13_inactive_zero | moving_avg_8 | moving_avg_4 | moving_avg_26 | seasonal_naive | zero |
|---|---|---|---|---|---|---|---|---|---|---|---|
| False | 11,236 | 99.6% | 99.3% | -8.3% | 0.628 | 0.628 | 0.631 | 0.634 | 0.654 | 0.717 | 2.000 |
| True | 3,817 | 0.4% | 0.7% | -100.0% | 2.000 | 2.000 | 2.000 | 2.000 | 1.815 | 11.406 | 2.000 |

### volume band (by Phase 0 units)

| volume_band | series | units | error share | signed bias (moving_avg_13) | moving_avg_13 | moving_avg_13_inactive_zero | moving_avg_8 | moving_avg_4 | moving_avg_26 | seasonal_naive | zero |
|---|---|---|---|---|---|---|---|---|---|---|---|
| top 1% | 151 | 32.4% | 24.5% | +3.0% | 0.444 | 0.444 | 0.398 | 0.426 | 0.477 | 0.621 | 2.000 |
| top 1-10% | 1,355 | 42.1% | 41.0% | -9.6% | 0.630 | 0.630 | 0.631 | 0.631 | 0.659 | 0.804 | 2.000 |
| rest 90% | 13,547 | 25.5% | 34.5% | -21.7% | 0.957 | 0.957 | 0.949 | 0.926 | 0.985 | 0.918 | 2.000 |

## Where the error comes from (`moving_avg_13`)

Share of total absolute error by volume band × demand type:

| volume band | smooth | erratic | intermittent | lumpy |
|---|---|---|---|---|
| top 1% | 16.1% | 7.9% | 0.3% | 0.2% |
| top 1-10% | 17.7% | 15.7% | 1.9% | 5.7% |
| rest 90% | 12.4% | 10.3% | 4.3% | 7.4% |

Concentration across series: the worst 1% of series (151) make 36.3% of absolute error; the worst 10% make 79.3%.

Total units per week, actual vs forecast:

| ds | actual | forecast | signed_bias |
|---|---|---|---|
| 2023-10-09 | 329,137 | 274,508 | -16.6% |
| 2023-10-16 | 246,151 | 274,508 | +11.5% |
| 2023-10-23 | 271,351 | 274,508 | +1.2% |
| 2023-10-30 | 254,928 | 274,508 | +7.7% |
| 2023-11-06 | 272,795 | 274,508 | +0.6% |
| 2023-11-13 | 276,448 | 274,508 | -0.7% |
| 2023-11-20 | 408,746 | 274,508 | -32.8% |
| 2023-11-27 | 368,713 | 274,508 | -25.5% |
| 2023-12-04 | 334,951 | 274,508 | -18.0% |
| 2023-12-11 | 375,764 | 274,508 | -26.9% |
| 2023-12-18 | 278,600 | 274,508 | -1.5% |
| 2023-12-25 | 240,185 | 274,508 | +14.3% |
| 2024-01-01 | 248,037 | 274,508 | +10.7% |

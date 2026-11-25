# Features: global direct multi-horizon model

Built by `python -m src.run_features` from `data/processed/phase0_clean.parquet` only. No Phase 1 or Phase 2 file is read. One row per (series, origin T, horizon h); target = y at T + h.

## Rules

- Every feature uses weeks ≤ T only. Flags from the cleaned file (smoothed, stockout, price regimes) and series_table stats are **not** used: they were computed on all of Phase 0. They are recomputed at each origin from data ≤ T.
- A series is included at origin T only if its first sale is ≤ T. Weeks before the first sale are missing (NaN), not zero.
- Scale = mean of the last 13 weeks at T (fallback: mean since first sale; floor 0.1). Target and features marked *scaled* are divided by it; unscaled copies are kept as `<name>_raw` (not model inputs).
- Client 0 prices before 2022-08-29 are treated as missing.
- The weekly-data switch date (2022-08-29) is used as a fixed calendar constant for `smoothed_share_*`.

## Dataset

- Training origins: 16, every 2 weeks from 2022-09-05 to 2023-04-03 (the last origin whose 13th target is ≤ the validation origin).
- Validation origin: 2023-07-03, targets 2023-07-10 to 2023-10-02 (the last 13 weeks of Phase 0).
- Rows: **2,499,211** (2,320,643 train, 178,568 valid), 70 columns, 43 model features. 764 MB in memory (float32), 40.0 MB on disk.

## Feature list

`% missing` is over all rows (train + valid).

| name | group | definition | scaled | % missing |
|---|---|---|---|---|
| lag_1 | recent demand | y at week T-0 (lag 1 = origin week) | yes | 0.0% |
| lag_2 | recent demand | y at week T-1 (lag 1 = origin week) | yes | 0.7% |
| lag_4 | recent demand | y at week T-3 (lag 1 = origin week) | yes | 2.3% |
| lag_8 | recent demand | y at week T-7 (lag 1 = origin week) | yes | 5.2% |
| lag_13 | recent demand | y at week T-12 (lag 1 = origin week) | yes | 8.7% |
| roll_mean_4 | recent demand | mean y over the last 4 weeks (post first sale) | yes | 0.0% |
| roll_std_4 | recent demand | std of y over the last 4 weeks | yes | 0.0% |
| roll_mean_8 | recent demand | mean y over the last 8 weeks (post first sale) | yes | 0.0% |
| roll_std_8 | recent demand | std of y over the last 8 weeks | yes | 0.0% |
| roll_mean_13 | recent demand | mean y over the last 13 weeks (post first sale) | yes | 0.0% |
| roll_std_13 | recent demand | std of y over the last 13 weeks | yes | 0.0% |
| roll_mean_26 | recent demand | mean y over the last 26 weeks (post first sale) | yes | 0.0% |
| roll_std_26 | recent demand | std of y over the last 26 weeks | yes | 0.0% |
| roll_mean_52 | recent demand | mean y over the last 52 weeks (post first sale) | yes | 0.0% |
| roll_std_52 | recent demand | std of y over the last 52 weeks | yes | 0.0% |
| roll_max_13 | recent demand | max y over the last 13 weeks | yes | 0.0% |
| trend_4_13 | trend | mean4 / mean13 (NaN if mean13 = 0) |  | 23.5% |
| trend_13_52 | trend | mean13 / mean52 (NaN if mean52 = 0) |  | 8.9% |
| zero_share_13 | intermittency | share of zero weeks in the last 13 weeks |  | 0.0% |
| zero_share_52 | intermittency | share of zero weeks in the last 52 weeks |  | 0.0% |
| weeks_since_last_sale | intermittency | weeks from the last positive sale to T (0 = sold at T) |  | 0.0% |
| age_weeks | age | weeks from first sale to T |  | 0.0% |
| seas_lag_52 | seasonality | y at target week - 52 (NaN if before first sale) | yes | 27.3% |
| seas_mean_3 | seasonality | mean y over target week - 53 .. - 51 (available weeks only) | yes | 26.7% |
| tw_week_of_year | calendar | ISO week number of the target week |  | 0.0% |
| tw_month | calendar | month of the target week (Monday date) |  | 0.0% |
| tw_thanksgiving_week | calendar | target week contains US Thanksgiving (and Black Friday) |  | 0.0% |
| tw_cyber_week | calendar | target week starts on Cyber Monday |  | 0.0% |
| tw_christmas_week | calendar | target week contains 25 December |  | 0.0% |
| prod_mean_13 | hierarchy | same product summed over all its warehouses: mean weekly units, last 13 weeks | yes | 0.0% |
| client_trend_4_13 | hierarchy | client total units: mean4 / mean13 |  | 0.0% |
| client_share_13 | hierarchy | series units / client units over the last 13 weeks |  | 0.0% |
| relative_price | price | price at T / median valid price of the 13 weeks before (past only, reset at detected shifts) |  | 24.7% |
| price_observed_share_13 | price | share of the last 13 weeks with a trusted recorded price |  | 0.0% |
| client | categorical | client id (categorical) |  | 0.0% |
| warehouse | categorical | warehouse id (categorical) |  | 0.0% |
| demand_type | categorical | Syntetos-Boylan class from ADI/CV2 over the last 52 weeks at T |  | 0.0% |
| smoothed_share_13 | data quality | share of last 13 weeks in an identical run before the weekly switch |  | 0.0% |
| smoothed_share_52 | data quality | same over the last 52 weeks |  | 0.0% |
| stockout_share_13 | data quality | share of last 13 weeks flagged as suspected stockout (recomputed at T) |  | 0.0% |
| stockout_share_52 | data quality | same over the last 52 weeks |  | 0.0% |
| log_scale | scaling | log(scale): the series' volume level at T |  | 0.0% |
| h | horizon | weeks ahead (1-13) |  | 0.0% |

## Smoke test: LightGBM (tweedie, default params) vs moving_avg_13

Trained on the training origins, predicted at the validation origin, unscaled, scored with `vn1_score` on the last 13 weeks of Phase 0. Default parameters, 100 trees: a pipeline check, not a tuned model.

| model | score | accuracy | bias |
|---|---|---|---|
| LightGBM | **0.5127** | 0.4977 | 0.0150 |
| moving_avg_13 | 0.5004 | 0.4722 | 0.0282 |

1317 series first sold after the validation origin; both models forecast 0 for them (2.5% of validation units).

### Score by volume band (band from units in the 52 weeks up to the origin)

| volume band | units | lgbm score | ma13 score | lgbm signed bias | ma13 signed bias | lgbm error share |
|---|---|---|---|---|---|---|
| top 1% | 35.1% | 0.538 | 0.372 | +15.1% | +7.2% | 27.3% |
| top 1-10% | 42.2% | 0.504 | 0.525 | -4.7% | -5.6% | 38.7% |
| rest 90% | 20.2% | 0.745 | 0.741 | +3.2% | -2.5% | 29.0% |
| first sale after origin | 2.5% | 2.000 | 2.000 | -100.0% | -100.0% | 5.0% |

### Score by demand type at origin

| demand_type | lgbm | ma13 | units |
|---|---|---|---|
| erratic | 0.682 | 0.724 | 16.2% |
| first sale after origin | 2.0 | 2.0 | 2.5% |
| intermittent | 0.666 | 0.726 | 5.8% |
| lumpy | 1.091 | 1.237 | 4.2% |
| no_sales | 1.966 | 2.0 | 0.1% |
| smooth | 0.451 | 0.391 | 71.2% |

### Score by horizon

| h | lgbm | ma13 |
|---|---|---|
| 1 | 0.66 | 0.673 |
| 2 | 0.463 | 0.392 |
| 3 | 0.548 | 0.466 |
| 4 | 0.466 | 0.433 |
| 5 | 0.571 | 0.461 |
| 6 | 0.574 | 0.515 |
| 7 | 0.494 | 0.462 |
| 8 | 0.58 | 0.588 |
| 9 | 0.587 | 0.539 |
| 10 | 0.673 | 0.628 |
| 11 | 0.556 | 0.514 |
| 12 | 0.655 | 0.678 |
| 13 | 0.664 | 0.571 |

### Top 20 features (gain)

| feature | gain_share | splits |
|---|---|---|
| warehouse | 71.1% | 1277 |
| log_scale | 7.5% | 131 |
| roll_std_52 | 5.7% | 71 |
| client | 2.4% | 158 |
| seas_mean_3 | 1.8% | 81 |
| tw_week_of_year | 1.7% | 260 |
| roll_mean_26 | 1.5% | 44 |
| roll_mean_52 | 1.2% | 54 |
| client_trend_4_13 | 0.9% | 75 |
| client_share_13 | 0.8% | 55 |
| trend_13_52 | 0.8% | 63 |
| trend_4_13 | 0.7% | 17 |
| age_weeks | 0.5% | 122 |
| weeks_since_last_sale | 0.5% | 126 |
| lag_1 | 0.5% | 46 |
| h | 0.5% | 61 |
| roll_mean_4 | 0.5% | 31 |
| zero_share_52 | 0.2% | 62 |
| prod_mean_13 | 0.2% | 61 |
| seas_lag_52 | 0.1% | 24 |

### Why `warehouse` dominates: diagnostic variants (default params each)

Scaled-target distribution (train): median 0.00, 99th pct 20.0, 99.9th pct 660, max 8830. Rows with scaled target > 50 are 0.54% of training rows but hold 69% of the summed scaled target; 45% of them are warehouse 251. They come from series with a tiny scale (quiet for 13 weeks) that then sell again. Unweighted tweedie on the scaled target chases those rows, and a high-cardinality categorical is the easiest way to isolate them. This is a loss/scale mismatch, not a time leak (warehouse is constant per series).

| variant | score | bias | top feature |
|---|---|---|---|
| default (above) | 0.5127 | 0.015 | warehouse (71%) |
| without client/warehouse | 0.496 | 0.0245 | client_trend_4_13 (16%) |
| weight = scale | 0.4807 | 0.0195 | warehouse (33%) |
| weight = scale, without client/warehouse | 0.4791 | 0.0121 | lag_1 (16%) |
| scaled target clipped at 20 | 0.4785 | 0.0033 | warehouse (42%) |

### Leak screen: features most correlated with the scaled target (training rows)

| feature | |spearman| with scaled target |
|---|---|
| zero_share_13 | 0.556 |
| weeks_since_last_sale | 0.554 |
| price_observed_share_13 | 0.55 |
| zero_share_52 | 0.542 |
| lag_1 | 0.479 |
| log_scale | 0.476 |
| roll_mean_13 | 0.459 |
| lag_2 | 0.455 |
| roll_mean_4 | 0.435 |
| client_share_13 | 0.434 |

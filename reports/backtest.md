# Backtest: LightGBM vs baselines on Phase 0 folds

`python -m src.backtest`. Reads only `phase0_clean.parquet`; Phase 1 and 2 are not used. Every choice below (blend weight, best configuration) comes from these scores.

## Setup

| fold | origin | targets | training origins: full | training origins: weekly_only |
|---|---|---|---|---|
| A (holiday) | 2022-10-31 | 2022-11-07 → 2023-01-30 | 29 (2021-07-05 → 2022-08-01) | none |
| B | 2023-01-30 | 2023-02-06 → 2023-05-01 | 35 (2021-07-05 → 2022-10-24) | 5 (2022-09-05 → 2022-10-31) |
| C | 2023-04-03 | 2023-04-10 → 2023-07-03 | 40 (2021-07-05 → 2023-01-02) | 9 (2022-09-05 → 2022-12-26) |
| D | 2023-07-03 | 2023-07-10 → 2023-10-02 | 46 (2021-07-05 → 2023-03-27) | 16 (2022-09-05 → 2023-04-03) |

- LightGBM: tweedie, default params (100 trees), seed 0, rows weighted by scale. Grid: scale ['mean13', 'max13_52'] × client/warehouse {with, without} × training origins ['full', 'weekly_only'].
- Series that first sell after a fold's origin get 0 from every method.
- Run: 63 origins cached (68 s), total 607 s, peak memory 1.77 GB.
- Skipped: fold A, setting `weekly_only`: no training origin has all 13 targets <= 2022-10-31 (setting starts 2022-09-05).

## Scores (VN1, lower is better)

`mean all` = mean over folds A, B, C, D (– if a fold is missing); `mean non-holiday` = mean over B, C, D.

| method | A | B | C | D | mean all | mean non-holiday |
|---|---|---|---|---|---|---|
| lgbm_full_mean13_noids | 0.6419 | 0.5109 | 0.4458 | 0.4838 | 0.5206 | 0.4802 |
| lgbm_full_max13_52_noids | 0.6435 | 0.5109 | 0.4472 | 0.4857 | 0.5218 | 0.4813 |
| lgbm_full_max13_52_ids | 0.6250 | 0.5355 | 0.5416 | 0.4711 | 0.5433 | 0.5161 |
| lgbm_full_mean13_ids | 0.6347 | 0.5626 | 0.5574 | 0.4740 | 0.5572 | 0.5314 |
| blend_0.25 | 0.7893 | 0.7010 | 0.5097 | 0.5284 | 0.6321 | 0.5797 |
| moving_avg_13 | 0.8354 | 0.7645 | 0.4832 | 0.5004 | 0.6459 | 0.5827 |
| blend_0.5 | 0.7733 | 0.6650 | 0.5648 | 0.5903 | 0.6484 | 0.6067 |
| blend_0.75 | 0.7837 | 0.6555 | 0.6385 | 0.6761 | 0.6885 | 0.6567 |
| seasonal_naive | 0.8133 | 0.7257 | 0.7276 | 0.7742 | 0.7602 | 0.7425 |
| seasonal_index | 0.8237 | 0.6819 | 0.7179 | 0.8927 | 0.7790 | 0.7641 |
| lgbm_weekly_only_mean13_noids | – | 0.4849 | 0.4609 | 0.5007 | – | 0.4822 |
| lgbm_weekly_only_max13_52_noids | – | 0.4816 | 0.4884 | 0.4885 | – | 0.4862 |
| lgbm_weekly_only_mean13_ids | – | 2.3377 | 0.6531 | 0.4828 | – | 1.1579 |
| lgbm_weekly_only_max13_52_ids | – | 3.2219 | 0.6414 | 0.4801 | – | 1.4478 |

- Blend weight chosen by mean score over all folds: **blend_0.25** (w × seasonal naive + (1 − w) × MA13).
- Best baseline: **blend_0.25**. Best LightGBM (all folds): **lgbm_full_mean13_noids**.

## Does each LightGBM choice matter?

Score change when flipping one factor with the others held fixed, averaged over matching pairs (negative = the change helps).

| change | mean score change | pairs | better in | folds compared |
|---|---|---|---|---|
| setting: full → weekly_only | +0.3913 | 12 | 2/12 | B, C, D |
| scale: mean13 → max13_52 | +0.0594 | 14 | 9/14 | A, B, C, D |
| ids: True → False | -0.3674 | 14 | 8/14 | A, B, C, D |

## Accuracy, bias and signed bias per fold

| fold | method | score | accuracy | bias | signed_bias |
|---|---|---|---|---|---|
| A | lgbm_full_max13_52_ids | 0.6250 | 0.5591 | 0.0659 | -6.6% |
| A | lgbm_full_mean13_ids | 0.6347 | 0.5597 | 0.0749 | -7.5% |
| A | lgbm_full_mean13_noids | 0.6419 | 0.5676 | 0.0743 | -7.4% |
| A | lgbm_full_max13_52_noids | 0.6435 | 0.5598 | 0.0837 | -8.4% |
| A | blend_0.25 | 0.7893 | 0.5939 | 0.1954 | -19.5% |
| A | seasonal_naive | 0.8133 | 0.7200 | 0.0933 | -9.3% |
| A | seasonal_index | 0.8237 | 0.7194 | 0.1042 | +10.4% |
| A | moving_avg_13 | 0.8354 | 0.6060 | 0.2294 | -22.9% |
| B | lgbm_weekly_only_max13_52_noids | 0.4816 | 0.4767 | 0.0049 | -0.5% |
| B | lgbm_weekly_only_mean13_noids | 0.4849 | 0.4699 | 0.0150 | -1.5% |
| B | lgbm_full_max13_52_noids | 0.5109 | 0.4666 | 0.0442 | -4.4% |
| B | lgbm_full_mean13_noids | 0.5109 | 0.4611 | 0.0498 | -5.0% |
| B | lgbm_full_max13_52_ids | 0.5355 | 0.4985 | 0.0370 | +3.7% |
| B | lgbm_full_mean13_ids | 0.5626 | 0.5077 | 0.0550 | +5.5% |
| B | seasonal_index | 0.6819 | 0.5973 | 0.0845 | +8.5% |
| B | blend_0.25 | 0.7010 | 0.6109 | 0.0902 | +9.0% |
| B | seasonal_naive | 0.7257 | 0.6989 | 0.0268 | -2.7% |
| B | moving_avg_13 | 0.7645 | 0.6354 | 0.1292 | +12.9% |
| B | lgbm_weekly_only_mean13_ids | 2.3377 | 1.3745 | 0.9632 | +96.3% |
| B | lgbm_weekly_only_max13_52_ids | 3.2219 | 1.8211 | 1.4008 | +140.1% |
| C | lgbm_full_mean13_noids | 0.4458 | 0.4444 | 0.0014 | -0.1% |
| C | lgbm_full_max13_52_noids | 0.4472 | 0.4395 | 0.0077 | -0.8% |
| C | lgbm_weekly_only_mean13_noids | 0.4609 | 0.4574 | 0.0035 | -0.3% |
| C | moving_avg_13 | 0.4832 | 0.4773 | 0.0059 | -0.6% |
| C | lgbm_weekly_only_max13_52_noids | 0.4884 | 0.4670 | 0.0215 | +2.1% |
| C | blend_0.25 | 0.5097 | 0.4942 | 0.0154 | -1.5% |
| C | lgbm_full_max13_52_ids | 0.5416 | 0.4799 | 0.0617 | +6.2% |
| C | lgbm_full_mean13_ids | 0.5574 | 0.4862 | 0.0713 | +7.1% |
| C | lgbm_weekly_only_max13_52_ids | 0.6414 | 0.5298 | 0.1116 | +11.2% |
| C | lgbm_weekly_only_mean13_ids | 0.6531 | 0.5355 | 0.1175 | +11.8% |
| C | seasonal_index | 0.7179 | 0.5926 | 0.1253 | +12.5% |
| C | seasonal_naive | 0.7276 | 0.6835 | 0.0440 | -4.4% |
| D | lgbm_full_max13_52_ids | 0.4711 | 0.4659 | 0.0053 | -0.5% |
| D | lgbm_full_mean13_ids | 0.4740 | 0.4673 | 0.0067 | +0.7% |
| D | lgbm_weekly_only_max13_52_ids | 0.4801 | 0.4597 | 0.0204 | -2.0% |
| D | lgbm_weekly_only_mean13_ids | 0.4828 | 0.4596 | 0.0232 | -2.3% |
| D | lgbm_full_mean13_noids | 0.4838 | 0.4585 | 0.0253 | -2.5% |
| D | lgbm_full_max13_52_noids | 0.4857 | 0.4572 | 0.0285 | -2.9% |
| D | lgbm_weekly_only_max13_52_noids | 0.4885 | 0.4665 | 0.0220 | -2.2% |
| D | moving_avg_13 | 0.5004 | 0.4722 | 0.0282 | -2.8% |
| D | lgbm_weekly_only_mean13_noids | 0.5007 | 0.4619 | 0.0388 | -3.9% |
| D | blend_0.25 | 0.5284 | 0.4887 | 0.0397 | -4.0% |
| D | seasonal_naive | 0.7742 | 0.6998 | 0.0744 | -7.4% |
| D | seasonal_index | 0.8927 | 0.6591 | 0.2336 | +23.4% |

## Best LightGBM vs best baseline by volume band at origin (all folds pooled)

| volume_band | units | lgbm_full_mean13_noids | blend_0.25 | signed bias lgbm_full_mean13_noids | signed bias blend_0.25 | error share lgbm_full_mean13_noids |
|---|---|---|---|---|---|---|
| top 1% | 34.2% | 0.2992 | 0.3611 | +1.0% | -1.8% | 20.5% |
| top 1-10% | 39.2% | 0.4726 | 0.5779 | +0.1% | +3.3% | 38.2% |
| rest 90% | 23.5% | 0.7736 | 0.8767 | -5.0% | -8.8% | 35.1% |
| first sale after origin | 3.0% | 2.0000 | 2.0000 | -100.0% | -100.0% | 6.2% |

## Best LightGBM vs best baseline by demand type at origin (all folds pooled)

| demand_type | units | lgbm_full_mean13_noids | blend_0.25 | signed bias lgbm_full_mean13_noids | signed bias blend_0.25 | error share lgbm_full_mean13_noids |
|---|---|---|---|---|---|---|
| smooth | 68.6% | 0.3761 | 0.4087 | +1.4% | -0.6% | 51.3% |
| erratic | 16.1% | 0.6613 | 0.8810 | -1.3% | +8.1% | 21.5% |
| intermittent | 6.4% | 0.9551 | 1.1099 | -12.9% | -25.1% | 10.9% |
| lumpy | 5.9% | 0.9342 | 1.0545 | -11.9% | -10.4% | 9.9% |
| no_sales | 0.1% | 1.9723 | 2.0000 | -63.3% | -100.0% | 0.2% |
| first sale after origin | 3.0% | 2.0000 | 2.0000 | -100.0% | -100.0% | 6.2% |

## By horizon (all folds pooled)

| h | lgbm_full_mean13_noids | blend_0.25 |
|---|---|---|
| 1 | 0.4708 | 0.5446 |
| 2 | 0.4578 | 0.5389 |
| 3 | 0.4669 | 0.5752 |
| 4 | 0.5027 | 0.6137 |
| 5 | 0.5111 | 0.6112 |
| 6 | 0.4968 | 0.5992 |
| 7 | 0.5254 | 0.5769 |
| 8 | 0.5829 | 0.6244 |
| 9 | 0.5382 | 0.5885 |
| 10 | 0.5829 | 0.6620 |
| 11 | 0.5624 | 0.6004 |
| 12 | 0.6256 | 0.6538 |
| 13 | 0.5855 | 0.6251 |

## Fold A (holiday): total units per week

| week | actual | lgbm_full_mean13_noids | blend_0.25 | moving_avg_13 | seasonal_naive | lgbm_full_mean13_noids vs actual | blend_0.25 vs actual | moving_avg_13 vs actual | seasonal_naive vs actual |
|---|---|---|---|---|---|---|---|---|---|
| 2022-11-07 | 229,096 | 234,808 | 217,133 | 212,750 | 230,283 | +2.5% | -5.2% | -7.1% | +0.5% |
| 2022-11-14 | 241,898 | 252,657 | 220,931 | 212,750 | 245,473 | +4.4% | -8.7% | -12.0% | +1.5% |
| 2022-11-21 | 327,362 | 299,216 | 231,857 | 212,750 | 289,178 | -8.6% | -29.2% | -35.0% | -11.7% |
| 2022-11-28 | 362,628 | 305,143 | 232,839 | 212,750 | 293,105 | -15.9% | -35.8% | -41.3% | -19.2% |
| 2022-12-05 | 316,275 | 310,873 | 234,856 | 212,750 | 301,174 | -1.7% | -25.7% | -32.7% | -4.8% |
| 2022-12-12 | 360,707 | 318,620 | 240,063 | 212,750 | 322,003 | -11.7% | -33.4% | -41.0% | -10.7% |
| 2022-12-19 | 245,038 | 246,216 | 218,957 | 212,750 | 237,578 | +0.5% | -10.6% | -13.2% | -3.0% |
| 2022-12-26 | 254,236 | 238,176 | 217,616 | 212,750 | 232,215 | -6.3% | -14.4% | -16.3% | -8.7% |
| 2023-01-02 | 278,981 | 236,422 | 217,465 | 212,750 | 231,612 | -15.3% | -22.1% | -23.7% | -17.0% |
| 2023-01-09 | 223,578 | 221,992 | 214,442 | 212,750 | 219,517 | -0.7% | -4.1% | -4.8% | -1.8% |
| 2023-01-16 | 241,896 | 219,860 | 213,628 | 212,750 | 216,263 | -9.1% | -11.7% | -12.0% | -10.6% |
| 2023-01-23 | 239,884 | 218,000 | 212,779 | 212,750 | 212,868 | -9.1% | -11.3% | -11.3% | -11.3% |
| 2023-01-30 | 267,527 | 220,417 | 215,321 | 212,750 | 223,035 | -17.6% | -19.5% | -20.5% | -16.6% |

## Top 20 features of lgbm_full_mean13_noids (gain share, mean over folds)

| feature | gain_share |
|---|---|
| lag_1 | 17.9% |
| seas_mean_3 | 9.8% |
| roll_std_52 | 9.1% |
| roll_mean_52 | 8.5% |
| log_scale | 8.3% |
| tw_week_of_year | 7.3% |
| client_trend_4_13 | 4.1% |
| seas_lag_52 | 3.6% |
| prod_mean_13 | 3.2% |
| client_share_13 | 3.1% |
| roll_mean_26 | 2.3% |
| lag_2 | 2.1% |
| h | 1.9% |
| zero_share_13 | 1.8% |
| age_weeks | 1.7% |
| trend_4_13 | 1.7% |
| smoothed_share_52 | 1.4% |
| roll_mean_4 | 1.3% |
| zero_share_52 | 1.2% |
| relative_price | 1.0% |

## Training set sizes

| fold | train_rows | origins | fit_seconds |
|---|---|---|---|
| A | 2,959,866 | 29 | 17 |
| B | 3,741,387 | 35 | 13 |
| C | 4,457,427 | 40 | 17 |
| D | 5,388,266 | 46 | 26 |

(largest setting per fold)

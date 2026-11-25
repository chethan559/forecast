# Phase 1 result (run once with the frozen config)

Trained on Phase 0 only: 53 origins (2021-07-05 → 2023-07-03), 6,562,256 rows, 10,570 dropped for crossing a level shift. Forecast origin 2023-10-02, forecasts 2023-10-09 → 2024-01-01. Config: `reports/final_config.md`. Forecasts: `outputs/phase1/`. Phase 2 not read.

## Total

| model | score | accuracy | bias | signed bias |
|---|---|---|---|---|
| lgbm_final | 0.5712 | 0.5119 | 0.0593 | +5.9% |
| blend_0.25 | 0.5816 | 0.5327 | 0.0489 | -4.9% |

## Phase 1 vs the backtest

| model | backtest mean (A–D) | backtest fold A (holiday) | Phase 1 |
|---|---|---|---|
| lgbm_final | 0.5089 | 0.6235 | 0.5712 |
| blend_0.25 | 0.6321 | 0.7893 | 0.5816 |

## By volume band (at the forecast origin)

| volume_band | units | lgbm score | baseline score | lgbm signed bias | baseline signed bias | lgbm error share |
|---|---|---|---|---|---|---|
| top 1% | 35.4% | 0.476 | 0.456 | +7.6% | +4.4% | 27.6% |
| top 1-10% | 41.2% | 0.529 | 0.557 | +5.4% | -5.0% | 38.3% |
| rest 90% | 23.4% | 0.789 | 0.947 | +4.3% | -18.7% | 34.1% |

## By demand type (at the forecast origin)

| demand_type | units | lgbm score | baseline score | lgbm signed bias | baseline signed bias | lgbm error share |
|---|---|---|---|---|---|---|
| smooth | 72.6% | 0.445 | 0.446 | +4.4% | -2.6% | 57.0% |
| erratic | 20.4% | 0.715 | 0.801 | +5.3% | -8.3% | 26.4% |
| intermittent | 2.9% | 1.263 | 1.260 | +16.1% | -21.9% | 6.3% |
| lumpy | 4.0% | 1.630 | 1.376 | +33.7% | -12.8% | 10.2% |
| no_sales | 0.1% | 2.000 | 2.000 | -100.0% | -100.0% | 0.2% |
| no history | 0.0% | 2.000 | 2.000 | -100.0% | -100.0% | 0.1% |

## Week by week

| week | actual | lgbm_final | lgbm_final bias | blend_0.25 | blend_0.25 bias | holiday |
|---|---|---|---|---|---|---|
| 2023-10-09 | 329,137 | 266,660 | -19.0% | 275,196 | -16.4% |  |
| 2023-10-16 | 246,151 | 263,655 | +7.1% | 269,951 | +9.7% |  |
| 2023-10-23 | 271,351 | 266,378 | -1.8% | 267,942 | -1.3% |  |
| 2023-10-30 | 254,928 | 277,336 | +8.8% | 277,684 | +8.9% |  |
| 2023-11-06 | 272,795 | 283,691 | +4.0% | 275,864 | +1.1% |  |
| 2023-11-13 | 276,448 | 295,953 | +7.1% | 280,471 | +1.5% |  |
| 2023-11-20 | 408,746 | 385,460 | -5.7% | 301,683 | -26.2% | Thanksgiving |
| 2023-11-27 | 368,713 | 405,518 | +10.0% | 309,259 | -16.1% | Cyber week |
| 2023-12-04 | 334,951 | 397,830 | +18.8% | 298,130 | -11.0% |  |
| 2023-12-11 | 375,764 | 388,381 | +3.4% | 308,697 | -17.8% |  |
| 2023-12-18 | 278,600 | 313,795 | +12.6% | 280,448 | +0.7% |  |
| 2023-12-25 | 240,185 | 286,655 | +19.3% | 281,964 | +17.4% | Christmas |
| 2024-01-01 | 248,037 | 306,117 | +23.4% | 287,584 | +15.9% |  |

## 10 series with the largest absolute error (lgbm_final)

| unique_id | band | demand_type | mean_13wk_before | max_52wk_before | same_weeks_last_year_mean | forecast_per_week | baseline_per_week | actual_per_week | actual_max_week | error_share | level_shift_date | group_level_shift_date | has_suspected_stockout |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0_3_1154 | top 1% | erratic | 1,637 | 13,681 | 1,323 | 926 | 1,558 | 2,003 | 15,236 | 1.1% | – | – | False |
| 41_228_2866 | top 1% | smooth | 3,978 | 10,475 | 6,900 | 3,970 | 4,708 | 5,594 | 11,409 | 1.1% | – | – | False |
| 21_135_7572 | top 1% | lumpy | 715 | 4,800 | 682 | 1,476 | 715 | 131 | 500 | 0.9% | 2023-05-08 | – | True |
| 38_63_7889 | top 1% | erratic | 2,793 | 3,691 | – | 4,294 | 2,793 | 3,040 | 5,324 | 0.8% | – | – | False |
| 44_319_3445 | top 1% | erratic | 1,982 | 4,336 | – | 3,288 | 1,982 | 2,304 | 3,755 | 0.8% | – | – | False |
| 38_68_4902 | top 1% | smooth | 3,358 | 5,931 | 2,793 | 3,300 | 3,217 | 2,496 | 4,533 | 0.8% | – | – | False |
| 38_63_12903 | top 1% | smooth | 3,502 | 6,878 | 2,877 | 3,684 | 3,346 | 2,670 | 5,044 | 0.8% | – | – | False |
| 38_68_7768 | top 1% | smooth | 2,791 | 3,897 | 1,891 | 2,979 | 2,566 | 1,909 | 3,803 | 0.7% | – | – | False |
| 38_68_9805 | top 1% | smooth | 2,843 | 3,301 | 1,730 | 2,885 | 2,565 | 2,004 | 3,843 | 0.7% | – | – | False |
| 22_265_6181 | top 1% | smooth | 1,697 | 6,104 | 2,695 | 2,118 | 1,947 | 2,676 | 5,893 | 0.6% | – | – | False |

`same_weeks_last_year_mean` = mean of 2022-10-10 → 2023-01-02 (the Phase 1 weeks one year earlier).
## Reading (written after scoring; nothing was changed in the model or config)

- **Headline:** frozen LightGBM 0.571 vs blend_0.25 0.582. LightGBM wins by 0.010, far less than the
  backtest gap (0.12 on average, 0.17 on fold A). Accuracy part is better (0.512 vs 0.533); bias part is
  worse (0.059 vs 0.049), and in the opposite direction: LightGBM over-forecasts (+5.9%), the baseline
  under-forecasts (−4.9%).
- **Holiday:** LightGBM catches the peak's arrival (Thanksgiving week −5.7%, baseline −26%), then stays high
  for too long: +19% in early December, +19% Christmas week, +23% New Year week. The baseline misses the
  peak but is closer on ordinary weeks.
- **Where LightGBM loses:** the top 1% band (0.476 vs 0.456, +7.6% bias) and lumpy series (1.63 vs 1.38,
  +34%). It wins clearly on the rest-90% band (0.789 vs 0.947) and on erratic series (0.715 vs 0.801).
- **Phase 1 vs backtest:** harder than the backtest average for LightGBM (0.571 vs 0.509) but easier than
  the holiday fold (0.624). Much easier than the backtest for the baseline (0.582 vs 0.632 average, 0.789 on
  fold A). Likely reason: in fold A, the baseline's seasonal-naive part copied 2021, a smoothed (monthly
  spread) holiday, which made it look weak. In Phase 1 it copies 2022, a true weekly holiday. The backtest
  overstated LightGBM's edge.
- **Top-10 series:** error is spread thin (each 0.6–1.1%, together ~8%). Three patterns:
  1. *Client 38 over-forecast* (38_68_4902, 38_63_12903, 38_68_7768, 38_68_9805): forecasts follow the recent
     13-week level (2.8–3.5k/week), but last year these weeks were well below it (1.7–2.9k) and so is Phase 1
     (1.9–2.7k). The model did not apply last year's Q4 dip for this client.
  2. *Under-forecast of spiky / strongly seasonal series* (0_3_1154 with a 15k spike week; 41_228_2866, whose
     same weeks last year averaged 6.9k: forecast 4.0k, actual 5.6k).
  3. *Young series extrapolated up* (38_63_7889, 44_319_3445: no last year, forecast ~50% above their
     13-week mean, actual close to it), and a lumpy series after a down level shift plus stockouts
     (21_135_7572: forecast 1,476/week, actual 131).

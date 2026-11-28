# M5 data quality (training weeks only)

`python -m src.experiment_m5 --stage prepare`. Adapter: daily units summed into complete Monday weeks, series = item × store, price = day-weighted average of the Walmart-week prices (M5 calendar). Cleaning: `configs/m5.yaml` = the VN1 thresholds. The last 13 weeks (2016-02-22 .. 2016-05-16 (13)) are sealed in `data/processed/m5/sealed/` and were not cleaned or looked at.

## Size

| item | value |
|---|---|
| series (item × store) | 30,490 |
| sold at least once in training | 30,490 |
| stores / states / departments / items | 10 / 3 / 7 / 3,049 |
| training weeks | 2011-01-31 .. 2016-02-15 (264) |
| series-weeks after first sale | 6,294,639 |
| units in training | 62,994,893 |
| zero weeks (after first sale) | 24.4% |
| series per store | 3,049 – 3,049 |

## Demand types (Syntetos–Boylan on the whole trimmed history, VN1 thresholds ADI 1.32 / CV² 0.49)

| demand_type | series | units | series % | units % | sample series % |
|---|---|---|---|---|---|
| smooth | 15,741 | 46,677,156 | 51.6% | 74.1% | 51.2% |
| erratic | 1,843 | 6,215,003 | 6.0% | 9.9% | 5.8% |
| intermittent | 10,390 | 7,795,510 | 34.1% | 12.4% | 34.4% |
| lumpy | 2,516 | 2,307,224 | 8.3% | 3.7% | 8.6% |

## Smoothed periods (monthly totals spread over weeks)

The VN1 detector (changepoint on the weekly repeat rate: share of positive weeks equal to the previous week) puts the switch at **2014-03-17**, repeat rate 10.9% before vs 12.0% after (VN1: 20.0% vs 10.1%). By year: 2011: 10.7%, 2012: 10.9%, 2013: 11.2%, 2014: 11.7%, 2015: 12.1%, 2016: 12.0%.
Weeks flagged `is_smoothed` (identical run ≥ 3 before that date): 121,864 (1.94% of series-weeks).

## Level shifts (VN1 rule: 4-week mean > 10× or < 1/10 of the previous 13 weeks, level ≥ 5)

- series events: 29,796 in 13,245 series (43.4% of selling series)
- group events (> 50% of a group's active series in the same week, groups: state_id, store_id, cat_id, dept_id): 0

## Other flags

- suspected stockout weeks (zero run 4–13 weeks after ≥ 2 units/week): 201,753 (3.21% of series-weeks)
- series with a price level shift: 86

## Modelling sample

Modelling uses a fixed random sample of 7,000 series (seed 0), because all 30,490 series under the VN1 training-origin rule would need ~33M training rows and do not fit in 8 GB; the sample gives about VN1's training size. The demand-type mix of the sample is in the table above.

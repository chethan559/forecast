# M5 backtest (VN1 default engine, untouched)

`python -m src.experiment_m5 --stage backtest`. 7,000 sampled item × store series, training weeks only (≤ 2016-02-15). Automatic rolling folds (`src/folds.py`): origins every 13 weeks back from the last training week; a holiday fold is kept only if the same window a year earlier is clean weekly data. Each fold trains on targets ≤ its origin and forecasts the next 13 weeks.

Metrics: `score` = VN1 score (WAPE + |bias|), `wape` = Σ|F−A|/ΣA, `bias` = Σ(F−A)/ΣA (> 0 = over-forecast), `coverage_80` = share of series-weeks inside the engine's [p10, p90] (conformal; baselines have no interval).

## Fold selection

| origin | window | holiday | previous_year_clean | kept |
|---|---|---|---|---|
| 2015-11-16 | 2015-11-23 .. 2016-02-15 | True | True | True |
| 2015-08-17 | 2015-08-24 .. 2015-11-16 | False | True | True |
| 2015-05-18 | 2015-05-25 .. 2015-08-17 | False | True | True |
| 2015-02-16 | 2015-02-23 .. 2015-05-18 | False | False | True |

| fold | origin | train_rows | n_train_origins | seconds |
|---|---|---|---|---|
| F1 | 2015-02-16 | 5,345,652 | 74 | 149 |
| F2 | 2015-05-18 | 5,877,300 | 80 | 85 |
| F3 | 2015-08-17 | 6,504,914 | 87 | 93 |
| F4 | 2015-11-16 | 7,048,340 | 93 | 102 |

## VN1 score by fold

| fold | origin | blend_0.25 | engine | moving_avg_13 | seasonal_naive |
|---|---|---|---|---|---|
| F1 | 2015-02-16 | 0.5468 | 0.5112 | 0.5726 | 0.7212 |
| F2 | 2015-05-18 | 0.5568 | 0.5712 | 0.5722 | 0.6939 |
| F3 | 2015-08-17 | 0.4969 | 0.4886 | 0.4882 | 0.7053 |
| F4 | 2015-11-16 | 0.4857 | 0.5209 | 0.5274 | 0.7177 |

## Mean over folds

| method | score | wape | bias | coverage_80 | relative_width |
|---|---|---|---|---|---|
| engine | 0.5230 | 0.4651 | -5.8% | 80.9% | 1.78 |
| moving_avg_13 | 0.5401 | 0.5031 | -2.4% | – | – |
| seasonal_naive | 0.7095 | 0.6549 | -3.9% | – | – |
| blend_0.25 | 0.5216 | 0.4940 | -2.7% | – | – |

Folds the engine beats each baseline on: moving_avg_13 3/4, seasonal_naive 4/4, blend_0.25 2/4.

## By volume band (mean over folds; band at each origin by units in the previous 52 weeks)

| band | method | score | wape | bias | coverage_80 | units |
|---|---|---|---|---|---|---|
| all | engine | 0.5230 | 0.4651 | -5.8% | 80.9% | 811,410 |
| all | moving_avg_13 | 0.5401 | 0.5031 | -2.4% | – | 811,410 |
| all | seasonal_naive | 0.7095 | 0.6549 | -3.9% | – | 811,410 |
| all | blend_0.25 | 0.5216 | 0.4940 | -2.7% | – | 811,410 |
| top 1% | engine | 0.3897 | 0.3207 | -4.8% | 97.6% | 122,574 |
| top 1% | moving_avg_13 | 0.4655 | 0.3705 | +2.5% | – | 122,574 |
| top 1% | seasonal_naive | 0.4809 | 0.4196 | +5.3% | – | 122,574 |
| top 1% | blend_0.25 | 0.4250 | 0.3450 | +3.2% | – | 122,574 |
| top 1-10% | engine | 0.4514 | 0.4048 | -4.7% | 79.6% | 281,334 |
| top 1-10% | moving_avg_13 | 0.4834 | 0.4416 | +1.5% | – | 281,334 |
| top 1-10% | seasonal_naive | 0.6556 | 0.5955 | +0.4% | – | 281,334 |
| top 1-10% | blend_0.25 | 0.4605 | 0.4317 | +1.2% | – | 281,334 |
| rest 90% | engine | 0.6114 | 0.5482 | -6.3% | 81.1% | 405,482 |
| rest 90% | moving_avg_13 | 0.6425 | 0.5836 | -5.9% | – | 405,482 |
| rest 90% | seasonal_naive | 0.8563 | 0.7653 | -9.1% | – | 405,482 |
| rest 90% | blend_0.25 | 0.6468 | 0.5800 | -6.7% | – | 405,482 |
| first sale after origin | engine | 2.0000 | 1.0000 | -100.0% | 13.6% | 2,020 |
| first sale after origin | moving_avg_13 | 2.0000 | 1.0000 | -100.0% | – | 2,020 |
| first sale after origin | seasonal_naive | 2.0000 | 1.0000 | -100.0% | – | 2,020 |
| first sale after origin | blend_0.25 | 2.0000 | 1.0000 | -100.0% | – | 2,020 |

## Every fold × method × band

| fold | method | band | series | units | score | wape | bias | coverage_80 | relative_width |
|---|---|---|---|---|---|---|---|---|---|
| F1 | engine | all | 6,915 | 766,268 | 0.5112 | 0.4871 | -2.4% | 77.2% | 1.72 |
| F1 | engine | top 1% | 68 | 111,971 | 0.3590 | 0.3493 | +1.0% | 96.7% | 2.09 |
| F1 | engine | top 1-10% | 612 | 265,464 | 0.4380 | 0.4228 | -1.5% | 75.6% | 1.22 |
| F1 | engine | rest 90% | 6,192 | 387,255 | 0.6050 | 0.5689 | -3.6% | 77.5% | 1.96 |
| F1 | engine | first sale after origin | 43 | 1,578 | 2.0000 | 1.0000 | -100.0% | 12.6% | 0.00 |
| F1 | moving_avg_13 | all | 6,915 | 766,268 | 0.5726 | 0.5319 | -4.1% | – | – |
| F1 | moving_avg_13 | top 1% | 68 | 111,971 | 0.4005 | 0.3947 | +0.6% | – | – |
| F1 | moving_avg_13 | top 1-10% | 612 | 265,464 | 0.4805 | 0.4742 | +0.6% | – | – |
| F1 | moving_avg_13 | rest 90% | 6,192 | 387,255 | 0.6917 | 0.6092 | -8.2% | – | – |
| F1 | moving_avg_13 | first sale after origin | 43 | 1,578 | 2.0000 | 1.0000 | -100.0% | – | – |
| F1 | seasonal_naive | all | 6,915 | 766,268 | 0.7212 | 0.6890 | +3.2% | – | – |
| F1 | seasonal_naive | top 1% | 68 | 111,971 | 0.6475 | 0.4707 | +17.7% | – | – |
| F1 | seasonal_naive | top 1-10% | 612 | 265,464 | 0.7357 | 0.6364 | +9.9% | – | – |
| F1 | seasonal_naive | rest 90% | 6,192 | 387,255 | 0.8386 | 0.7870 | -5.2% | – | – |
| F1 | seasonal_naive | first sale after origin | 43 | 1,578 | 2.0000 | 1.0000 | -100.0% | – | – |
| F1 | blend_0.25 | all | 6,915 | 766,268 | 0.5468 | 0.5243 | -2.2% | – | – |
| F1 | blend_0.25 | top 1% | 68 | 111,971 | 0.4302 | 0.3817 | +4.9% | – | – |
| F1 | blend_0.25 | top 1-10% | 612 | 265,464 | 0.4946 | 0.4650 | +3.0% | – | – |
| F1 | blend_0.25 | rest 90% | 6,192 | 387,255 | 0.6791 | 0.6043 | -7.5% | – | – |
| F1 | blend_0.25 | first sale after origin | 43 | 1,578 | 2.0000 | 1.0000 | -100.0% | – | – |
| F2 | engine | all | 6,971 | 831,818 | 0.5712 | 0.4747 | -9.7% | 82.0% | 1.77 |
| F2 | engine | top 1% | 69 | 135,603 | 0.5059 | 0.3373 | -16.9% | 95.8% | 1.88 |
| F2 | engine | top 1-10% | 617 | 290,046 | 0.5036 | 0.4153 | -8.8% | 78.4% | 1.25 |
| F2 | engine | rest 90% | 6,229 | 401,807 | 0.6265 | 0.5582 | -6.8% | 82.7% | 2.12 |
| F2 | engine | first sale after origin | 56 | 4,362 | 2.0000 | 1.0000 | -100.0% | 14.9% | 0.00 |
| F2 | moving_avg_13 | all | 6,971 | 831,818 | 0.5722 | 0.4948 | -7.7% | – | – |
| F2 | moving_avg_13 | top 1% | 69 | 135,603 | 0.5090 | 0.3688 | -14.0% | – | – |
| F2 | moving_avg_13 | top 1-10% | 617 | 290,046 | 0.4782 | 0.4246 | -5.4% | – | – |
| F2 | moving_avg_13 | rest 90% | 6,229 | 401,807 | 0.6459 | 0.5824 | -6.4% | – | – |
| F2 | moving_avg_13 | first sale after origin | 56 | 4,362 | 2.0000 | 1.0000 | -100.0% | – | – |
| F2 | seasonal_naive | all | 6,971 | 831,818 | 0.6939 | 0.6570 | -3.7% | – | – |
| F2 | seasonal_naive | top 1% | 69 | 135,603 | 0.4836 | 0.4483 | +3.5% | – | – |
| F2 | seasonal_naive | top 1-10% | 617 | 290,046 | 0.6272 | 0.5978 | +2.9% | – | – |
| F2 | seasonal_naive | rest 90% | 6,229 | 401,807 | 0.8652 | 0.7664 | -9.9% | – | – |
| F2 | seasonal_naive | first sale after origin | 56 | 4,362 | 2.0000 | 1.0000 | -100.0% | – | – |
| F2 | blend_0.25 | all | 6,971 | 831,818 | 0.5568 | 0.4895 | -6.7% | – | – |
| F2 | blend_0.25 | top 1% | 69 | 135,603 | 0.4432 | 0.3469 | -9.6% | – | – |
| F2 | blend_0.25 | top 1-10% | 617 | 290,046 | 0.4534 | 0.4205 | -3.3% | – | – |
| F2 | blend_0.25 | rest 90% | 6,229 | 401,807 | 0.6542 | 0.5818 | -7.2% | – | – |
| F2 | blend_0.25 | first sale after origin | 56 | 4,362 | 2.0000 | 1.0000 | -100.0% | – | – |
| F3 | engine | all | 6,989 | 834,404 | 0.4886 | 0.4466 | -4.2% | 82.4% | 1.84 |
| F3 | engine | top 1% | 70 | 127,533 | 0.2959 | 0.2637 | +3.2% | 99.5% | 2.19 |
| F3 | engine | top 1-10% | 622 | 286,721 | 0.4174 | 0.3945 | -2.3% | 83.7% | 1.39 |
| F3 | engine | rest 90% | 6,279 | 418,257 | 0.6091 | 0.5355 | -7.4% | 82.2% | 2.06 |
| F3 | engine | first sale after origin | 18 | 1,893 | 2.0000 | 1.0000 | -100.0% | 8.5% | 0.00 |
| F3 | moving_avg_13 | all | 6,989 | 834,404 | 0.4882 | 0.4852 | -0.3% | – | – |
| F3 | moving_avg_13 | top 1% | 70 | 127,533 | 0.4219 | 0.3130 | +10.9% | – | – |
| F3 | moving_avg_13 | top 1-10% | 622 | 286,721 | 0.4843 | 0.4398 | +4.5% | – | – |
| F3 | moving_avg_13 | rest 90% | 6,279 | 418,257 | 0.6316 | 0.5666 | -6.5% | – | – |
| F3 | moving_avg_13 | first sale after origin | 18 | 1,893 | 2.0000 | 1.0000 | -100.0% | – | – |
| F3 | seasonal_naive | all | 6,989 | 834,404 | 0.7053 | 0.6344 | -7.1% | – | – |
| F3 | seasonal_naive | top 1% | 70 | 127,533 | 0.3706 | 0.3539 | -1.7% | – | – |
| F3 | seasonal_naive | top 1-10% | 622 | 286,721 | 0.6307 | 0.5750 | -5.6% | – | – |
| F3 | seasonal_naive | rest 90% | 6,279 | 418,257 | 0.8526 | 0.7590 | -9.4% | – | – |
| F3 | seasonal_naive | first sale after origin | 18 | 1,893 | 2.0000 | 1.0000 | -100.0% | – | – |
| F3 | blend_0.25 | all | 6,989 | 834,404 | 0.4969 | 0.4770 | -2.0% | – | – |
| F3 | blend_0.25 | top 1% | 70 | 127,533 | 0.3668 | 0.2894 | +7.7% | – | – |
| F3 | blend_0.25 | top 1-10% | 622 | 286,721 | 0.4455 | 0.4261 | +1.9% | – | – |
| F3 | blend_0.25 | rest 90% | 6,279 | 418,257 | 0.6390 | 0.5668 | -7.2% | – | – |
| F3 | blend_0.25 | first sale after origin | 18 | 1,893 | 2.0000 | 1.0000 | -100.0% | – | – |
| F4 | engine | all | 7,000 | 813,152 | 0.5209 | 0.4522 | -6.9% | 81.9% | 1.80 |
| F4 | engine | top 1% | 70 | 115,191 | 0.3980 | 0.3325 | -6.6% | 98.5% | 2.18 |
| F4 | engine | top 1-10% | 626 | 283,103 | 0.4464 | 0.3866 | -6.0% | 80.8% | 1.24 |
| F4 | engine | rest 90% | 6,293 | 414,609 | 0.6051 | 0.5300 | -7.5% | 81.9% | 2.07 |
| F4 | engine | first sale after origin | 11 | 249 | 2.0000 | 1.0000 | -100.0% | 18.5% | 0.00 |
| F4 | moving_avg_13 | all | 7,000 | 813,152 | 0.5274 | 0.5005 | +2.7% | – | – |
| F4 | moving_avg_13 | top 1% | 70 | 115,191 | 0.5305 | 0.4053 | +12.5% | – | – |
| F4 | moving_avg_13 | top 1-10% | 626 | 283,103 | 0.4907 | 0.4278 | +6.3% | – | – |
| F4 | moving_avg_13 | rest 90% | 6,293 | 414,609 | 0.6006 | 0.5762 | -2.4% | – | – |
| F4 | moving_avg_13 | first sale after origin | 11 | 249 | 2.0000 | 1.0000 | -100.0% | – | – |
| F4 | seasonal_naive | all | 7,000 | 813,152 | 0.7177 | 0.6390 | -7.9% | – | – |
| F4 | seasonal_naive | top 1% | 70 | 115,191 | 0.4218 | 0.4055 | +1.6% | – | – |
| F4 | seasonal_naive | top 1-10% | 626 | 283,103 | 0.6288 | 0.5728 | -5.6% | – | – |
| F4 | seasonal_naive | rest 90% | 6,293 | 414,609 | 0.8689 | 0.7490 | -12.0% | – | – |
| F4 | seasonal_naive | first sale after origin | 11 | 249 | 2.0000 | 1.0000 | -100.0% | – | – |
| F4 | blend_0.25 | all | 7,000 | 813,152 | 0.4857 | 0.4851 | +0.1% | – | – |
| F4 | blend_0.25 | top 1% | 70 | 115,191 | 0.4598 | 0.3619 | +9.8% | – | – |
| F4 | blend_0.25 | top 1-10% | 626 | 283,103 | 0.4483 | 0.4151 | +3.3% | – | – |
| F4 | blend_0.25 | rest 90% | 6,293 | 414,609 | 0.6152 | 0.5669 | -4.8% | – | – |
| F4 | blend_0.25 | first sale after origin | 11 | 249 | 2.0000 | 1.0000 | -100.0% | – | – |

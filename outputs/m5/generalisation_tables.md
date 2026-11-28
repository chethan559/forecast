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

# Round 3: level-shift restart and bias multiplier

`python -m src.experiment_round3`. Phase 0 folds A–D only. Config: default (full origins, max13_52, no ids, uplift features), cap k=3. Runtime 363 s.

## 1. restart_after_shift

On: features use only weeks from the latest warehouse/client level-shift event; training rows whose (origin, target] window contains an event are dropped. Cell = score (signed bias).

| restart_after_shift | A | B | C | D | average | training rows dropped |
|---|---|---|---|---|---|---|
| off | 0.6251 (-7.0%) | 0.5153 (-5.9%) | 0.4465 (-1.1%) | 0.4761 (-2.0%) | 0.5158 | 0 / 0 / 0 / 0 |
| on | 0.6235 (-7.0%) | 0.4904 (-3.9%) | 0.4478 (-1.4%) | 0.4739 (-1.8%) | 0.5089 | 882 / 8,882 / 8,882 / 9,067 |

Decision: **on** (rule: on only if the average is not worse; off 0.5158, on 0.5089).

## 2. Bias multiplier (leave-one-fold-out)

For each fold: fit the multiplier on the other three (grid 0.80–1.20, step 0.01, minimising their mean score), apply it to the held-out fold. Forecast × multiplier, then cap.

| held_out_fold | multiplier_fit_on_others | held_out_score_with | held_out_score_without |
|---|---|---|---|
| A | 1.02 | 0.6100 | 0.6235 |
| B | 1.02 | 0.4744 | 0.4904 |
| C | 1.04 | 0.4675 | 0.4478 |
| D | 1.04 | 0.4875 | 0.4739 |

Mean held-out score: with 0.5098, without 0.5089. Decision: **do not adopt** (multiplier stays 1.0; the all-fold optimum would be 1.02).

Score by multiplier (all folds):

| multiplier | A | B | C | D | mean |
|---|---|---|---|---|---|
| 0.90 | 0.7002 | 0.5846 | 0.5397 | 0.5598 | 0.5961 |
| 0.95 | 0.6600 | 0.5347 | 0.4906 | 0.5141 | 0.5499 |
| 0.97 | 0.6449 | 0.5163 | 0.4728 | 0.4974 | 0.5329 |
| 0.98 | 0.6376 | 0.5074 | 0.4642 | 0.4894 | 0.5247 |
| 0.99 | 0.6305 | 0.4988 | 0.4559 | 0.4816 | 0.5167 |
| 1.00 | 0.6235 | 0.4904 | 0.4478 | 0.4739 | 0.5089 |
| 1.01 | 0.6167 | 0.4823 | 0.4399 | 0.4665 | 0.5013 |
| 1.02 | 0.6100 | 0.4744 | 0.4430 | 0.4620 | 0.4973 |
| 1.03 | 0.6034 | 0.4667 | 0.4552 | 0.4747 | 0.5000 |
| 1.05 | 0.5905 | 0.4709 | 0.4801 | 0.5005 | 0.5105 |
| 1.10 | 0.6071 | 0.5337 | 0.5459 | 0.5683 | 0.5637 |

## Final backtest score of the frozen setup

A: 0.6235 | B: 0.4904 | C: 0.4478 | D: 0.4739 | **mean 0.5089**

# Phase 2 result (sealed test, run once)

Cleaning re-run on Phase 0 + Phase 1 combined; trained on 59 origins (2021-07-05 → 2023-09-25), 7,688,773 rows (10,750 dropped for crossing a level shift). Forecast origin 2024-01-01, forecasts 2024-01-08 → 2024-04-01. Config: `reports/final_config.md`. Forecasts: `outputs/phase2/`.

## Total

| method | score | accuracy | bias | signed bias | vs leaderboard winner |
|---|---|---|---|---|---|
| lgbm (chosen) | 0.4879 | 0.4446 | 0.0433 | -4.3% | +0.0242 |
| blend_0.25 | 0.7844 | 0.5951 | 0.1894 | +18.9% | +0.3207 |

VN1 Phase 2 leaderboard winner: **0.4637**. Not a like-for-like comparison: the official scoring removed stockout weeks from the actuals, and we have no stockout data, so every zero week counts here. Our scores are therefore harsher than the leaderboard's would be for the same forecasts.

## By volume band (at the forecast origin)

| volume_band | units | lgbm score | lgbm bias | blend_0.25 score | blend_0.25 bias |
|---|---|---|---|---|---|
| top 1% | 38.5% | 0.454 | -9.8% | 0.643 | +14.3% |
| top 1-10% | 42.7% | 0.435 | -2.8% | 0.734 | +19.0% |
| rest 90% | 18.8% | 0.744 | +3.4% | 1.190 | +28.3% |

## By demand type (at the forecast origin)

| demand_type | units | lgbm score | lgbm bias | blend_0.25 score | blend_0.25 bias |
|---|---|---|---|---|---|
| smooth | 75.7% | 0.396 | -3.2% | 0.538 | +9.8% |
| erratic | 18.3% | 0.634 | -8.3% | 1.291 | +41.9% |
| intermittent | 2.1% | 1.470 | +23.2% | 1.438 | +17.1% |
| lumpy | 3.2% | 1.155 | -4.5% | 2.258 | +53.2% |
| no_sales | 0.0% | 2.000 | -100.0% | 2.000 | -100.0% |
| no history | 0.7% | 2.000 | -100.0% | 5.790 | +268.9% |

## Week by week (total units)

| week | actual | lgbm | blend_0.25 | lgbm bias | blend_0.25 bias |
|---|---|---|---|---|---|
| 2024-01-08 | 256,499 | 239,082 | 292,696 | -6.8% | +14.1% |
| 2024-01-15 | 240,900 | 236,024 | 296,493 | -2.0% | +23.1% |
| 2024-01-22 | 243,019 | 241,690 | 296,122 | -0.5% | +21.9% |
| 2024-01-29 | 248,486 | 249,529 | 302,858 | +0.4% | +21.9% |
| 2024-02-05 | 236,531 | 248,097 | 303,116 | +4.9% | +28.2% |
| 2024-02-12 | 261,467 | 233,500 | 290,633 | -10.7% | +11.2% |
| 2024-02-19 | 234,517 | 233,282 | 296,603 | -0.5% | +26.5% |
| 2024-02-26 | 246,484 | 238,413 | 296,832 | -3.3% | +20.4% |
| 2024-03-04 | 256,865 | 238,119 | 300,255 | -7.3% | +16.9% |
| 2024-03-11 | 227,597 | 236,283 | 292,931 | +3.8% | +28.7% |
| 2024-03-18 | 250,503 | 235,492 | 294,950 | -6.0% | +17.7% |
| 2024-03-25 | 268,816 | 238,984 | 299,373 | -11.1% | +11.4% |
| 2024-04-01 | 271,055 | 233,860 | 293,936 | -13.7% | +8.4% |

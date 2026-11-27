# Prediction intervals (p10 / p90) on VN1

`python -m src.experiment_intervals`. Folds B, C, D (Phase 0 backtest) and E = Phase 1. Target: 80% of series-weeks inside [p10, p90]. `relative_width` = Σ(p90 − p10) / Σ actual units (lower = narrower).

- **quantile_lgbm**: two extra LightGBM models, quantile objective (alpha 0.1 / 0.9), same training rows.
- **conformal**: the model trained 13 weeks before the origin, scored on those 13 weeks; 10th/90th percentiles of its scaled residuals by volume band × horizon (falls back to horizon, then overall, below 200 rows).

## Coverage by fold

| fold | method | p50_score | coverage | below_p10 | above_p90 | relative_width |
|---|---|---|---|---|---|---|
| B | conformal | 0.4904 | 82.8% | 7.3% | 9.9% | 2.15 |
| B | quantile_lgbm | 0.4904 | 85.6% | 3.1% | 11.4% | 1.32 |
| C | conformal | 0.4478 | 73.8% | 14.1% | 12.1% | 1.58 |
| C | quantile_lgbm | 0.4478 | 86.4% | 2.5% | 11.1% | 1.34 |
| D | conformal | 0.4739 | 80.9% | 7.4% | 11.7% | 1.55 |
| D | quantile_lgbm | 0.4739 | 87.1% | 2.7% | 10.2% | 1.31 |
| E | conformal | 0.5712 | 71.6% | 17.4% | 11.1% | 1.43 |
| E | quantile_lgbm | 0.5712 | 88.9% | 2.7% | 8.4% | 1.53 |

## Mean over folds B–E

| method | coverage | below_p10 | above_p90 | relative_width |
|---|---|---|---|---|
| conformal | 77.3% | 11.5% | 11.2% | 1.67 |
| quantile_lgbm | 87.0% | 2.7% | 10.3% | 1.37 |

**Chosen: `conformal`** (closest to 80% coverage). Rule: closest to 80% coverage; if both are within 3% of it, the narrower one.

## Coverage by volume band (mean over folds)

| method | band | coverage | relative_width |
|---|---|---|---|
| conformal | first sale after origin | 45.5% | 0.00 |
| quantile_lgbm | first sale after origin | 45.5% | 0.00 |
| conformal | rest 90% | 78.4% | 2.11 |
| quantile_lgbm | rest 90% | 89.2% | 1.85 |
| conformal | top 1% | 96.2% | 2.00 |
| quantile_lgbm | top 1% | 84.9% | 1.12 |
| conformal | top 1-10% | 79.0% | 1.24 |
| quantile_lgbm | top 1-10% | 82.9% | 1.41 |

## Point forecast check

The engine's p50 must equal the frozen model's reported scores:

| fold | engine p50 score | reported score | match |
|---|---|---|---|
| B | 0.4904291261 | 0.4904291261 | True |
| C | 0.4478167420 | 0.4478167420 | True |
| D | 0.4739087885 | 0.4739087885 | True |
| E | 0.5711712613 | 0.5711712613 | True |

# zova-forecast

Weekly demand forecasting pipeline for Zova (inventory planning for DTC ecommerce brands), built and
tested on the VN1 forecasting competition data: cleaning, baselines, a global LightGBM, a backtest harness,
and one-off scored runs on Phase 1 and the sealed Phase 2.

**Result:** VN1 score 0.571 on Phase 1 and **0.488 on Phase 2** (leaderboard winner 0.4637, scored without
stockout weeks, so not like-for-like). Summary: `reports/forecast_layer_summary.md`; frozen config and
evidence: `reports/final_config.md`; full log: `experiments.md`.

The raw VN1 CSVs are not in the repo: put them in `data/raw/vn1/` and run `python -m src.run_cleaning`.

## Run any dataset (one command)

```bash
.venv/bin/python -m src.run --config configs/<name>.yaml --stage clean|backtest|fit|predict
```

| stage | what it does | writes |
|---|---|---|
| `clean` | adapter → schema check → cleaning flags | `<processed_dir>/standard.parquet`, `clean.parquet`, `<reports_dir>/data_summary.md` |
| `backtest` | trains at each origin in `backtest.folds`, scores the next 13 weeks (p50 VN1 score, 80% interval coverage) | `<reports_dir>/engine_backtest.md` |
| `fit` | trains on all data | model artifact `<artifacts_dir>/model_<last week>/` |
| `predict` | newest artifact (or `--model`), `--origin`, `--horizon` | `<outputs_dir>/forecast_<origin>.csv`: `unique_id, ds, p50, p10, p90` |

Configs: `configs/defaults.yaml` (generic defaults) is merged with a dataset file:

| config | adapter | data (not in the repo) | status |
|---|---|---|---|
| `configs/vn1.yaml` | `vn1` (wide CSVs) | `data/raw/vn1/` | full pipeline, frozen and scored |
| `configs/m5.yaml` | `m5` (daily → Monday weeks) | Kaggle M5 files in `data/raw/m5/` | loads and validates; no modelling yet |
| `configs/shopify_example.yaml` | `shopify_orders` (orders export → weekly units per SKU, refunds/cancellations excluded) | `data/raw/shopify_example/orders_export.csv` | example; set your folds |

A new dataset needs: an adapter (or `long_csv` with a column mapping) that produces the standard schema
(`src/schema.py`: `unique_id, ds` (Monday), `y`, optional `price`, `in_stock`, hierarchy columns), and a
YAML with its hierarchy, calendar (`us` or `au`), paths and backtest folds. History needed: one year
before the first training origin, and for conformal intervals another 26 weeks before a forecast origin.

In Python:

```python
from src.config import load_config
from src import engine
cfg = load_config("configs/shopify_example.yaml")
model_dir = engine.fit(df, cfg)                        # df in the standard schema
fc = engine.predict(model_dir, df, origin=None, horizon=13)   # unique_id, ds, p50, p10, p90
res = engine.backtest(df, cfg)                         # folds from the config
```

`p50` is the point forecast the VN1 model was optimised for (tweedie, closer to a mean than a strict
median); `p10` / `p90` bound an 80% interval (method chosen in `reports/intervals.md`).

## Setup

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

## Run everything

```bash
.venv/bin/python -m src.run_cleaning
```

This loads the raw CSVs in `data/raw/vn1/` (copies of the originals, never modified), cleans
Phase 0 and Phase 0+1, writes parquet files to `data/processed/` and the report to
`reports/data_quality.md` (charts in `reports/figures/`). Takes about 30 seconds.
Add `--no-report` to skip the charts.

## Baselines

```bash
.venv/bin/python -m src.run_baselines
```

Trains on `phase0_clean.parquet`, forecasts the 13 Phase 1 weeks with every baseline in
`src/models/baselines.py`, scores them against `phase1_actuals.parquet` and writes forecasts to
`outputs/baselines/` and the results to `reports/baselines.md`. Never reads Phase 2.

## Features (global direct multi-horizon model)

```bash
.venv/bin/python -m src.run_features
```

Reads only `data/processed/phase0_clean.parquet`. Builds one row per (series, origin, horizon 1–13) for the
inner-validation origins (`src/splits.py`), writes `data/processed/features_inner.parquet`, and runs a
LightGBM smoke test against moving_avg_13 on the last 13 weeks of Phase 0. Feature list, missing rates and
smoke-test results: `reports/features.md`.

## Backtest

```bash
.venv/bin/python -m src.backtest
```

Scores LightGBM (a grid of scale method × client/warehouse × training-origin setting) and the baselines
(MA13, seasonal naive, blends, seasonal index) on four Phase 0 validation origins (config `backtest.folds`,
fold A covers the 2022 holidays). Features are cached per origin in `data/processed/features_cache/`.
Results: `reports/backtest.md`, scores and forecasts in `outputs/backtest/`. Reads only `phase0_clean.parquet`.
Takes roughly 20–30 minutes and up to ~5 GB of RAM.

## Final selection and one-off scored runs

```bash
.venv/bin/python -m src.experiment_final   # candidates on folds B-E (E = Phase 1) -> reports/final_round.md
.venv/bin/python -m src.run_phase1         # frozen config, scored once on Phase 1 -> reports/phase1.md
.venv/bin/python -m src.run_phase2         # frozen config, scored once on Phase 2 -> reports/phase2.md
```

Both one-off runs check a frozen snapshot of the config and source (`outputs/phase*/frozen_config.json`)
and refuse to run twice.

## Tests

```bash
.venv/bin/python -m pytest -q            # ~110 tests, ~4 min
.venv/bin/python -m pytest -m slow -q    # VN1 regression: Phase 1 0.5712 and Phase 2 0.4879 must reproduce (~6 min)
```

The old one-off runners (`src/run_phase1.py`, `src/run_phase2.py`) refuse to run now: their frozen
snapshots match the code as it was when they were scored. The slow regression test is the guarantee
that the refactored engine still reproduces those scores.

## Layout

| path | what |
|---|---|
| `src/config.py` | loads `configs/defaults.yaml` + `configs/<dataset>.yaml` (`get_config()` = VN1) |
| `src/adapters/vn1.py` | VN1 wide CSV → long `unique_id, client, warehouse, product, ds, y/price` with shape checks |
| `src/cleaning.py` | steps a–g as separate functions + `run_cleaning(sales, price, config)` |
| `src/metrics.py` | `vn1_score` (score, accuracy part, bias part), `vn1_score_by_group` |
| `src/models/baselines.py` | zero, moving_avg, seasonal_naive, moving_avg_inactive_zero |
| `src/run_baselines.py` | baseline runner + `reports/baselines.md` |
| `src/features.py` | `make_features(weekly_clean, origins, horizons, config)`: point-in-time features |
| `src/splits.py` | training / validation origins (inner validation, Phase 1 run) |
| `src/run_features.py` | feature build + LightGBM smoke test + `reports/features.md` |
| `src/models/lgbm.py` | LightGBM fit / predict / importance (tweedie, weighted by scale) |
| `src/evaluation.py` | Phase 0 loader and scoring helpers shared by the smoke test and backtest |
| `src/feature_cache.py` | per-origin cache of unscaled features |
| `src/backtest.py` | backtest folds, LightGBM grid + baselines, `reports/backtest.md` |
| `src/run.py` | one-command CLI: `--config ... --stage clean|backtest|fit|predict` |
| `src/engine.py` | `fit` / `predict` / `backtest` / `save` / `load`; p50 + p10/p90 (conformal or quantile LightGBM) |
| `src/schema.py` | standard input schema and validation with clear errors |
| `src/calendar.py` | US and AU holiday calendars (Monday weeks) |
| `src/adapters/m5.py`, `shopify_orders.py`, `long_csv.py` | other data sources → standard schema |
| `configs/*.yaml` | per-dataset config (all thresholds, dates, rules); `defaults.yaml` for the rest |
| `src/experiment_intervals.py` | interval method comparison on VN1 (`reports/intervals.md`) |
| `src/report.py` | data quality report and charts |
| `src/run_cleaning.py` | one-command entry point |
| `notebooks/01_eda.ipynb` | exploration on top of the processed files |
| `experiments.md` | dated log of what was done and key numbers |

## Outputs (`data/processed/`)

| file | content |
|---|---|
| `phase0_clean.parquet` | training weeks (after leading-zero trim) with all flags |
| `phase01_clean.parquet` | Phase 0 + Phase 1 cleaned the same way (for the final model) |
| `phase1_actuals.parquet`, `phase2_actuals.parquet` | long actuals, untouched |
| `series_table.parquet` | one row per series from Phase 0: every per-series stat and flag |
| `series_table_phase01.parquet` | same, from Phase 0 + 1 |
| `price_shift_events.parquet` | every detected price level shift (Phase 0) |

### Weekly columns

| column | meaning |
|---|---|
| `y` | units sold, never changed |
| `identical_run_len`, `identical_run` | length of the run of identical values; run ≥ 3 and positive |
| `is_smoothed_era`, `is_smoothed` | week before the estimated weekly-data switch; identical run inside that era |
| `price_raw` | original price (NaN when nothing sold) |
| `price_observed`, `price_zero`, `price_filled` | price recorded; price = 0 with sales > 0 (treated as missing); `price_clean` carried forward |
| `price_clean` | last valid price, forward-filled within the series (never backfilled) |
| `price_shift` | back-dated week of a structural price jump (descriptive, uses a few later weeks) |
| `price_shift_detected`, `price_regime` | week the past-only rule detected a shift; counter of detected shifts |
| `price_baseline`, `relative_price` | median valid price of the past 13 weeks in the current regime; `price_clean / price_baseline` |
| `zero_run_len`, `is_suspected_stockout` | length of the zero run; inferred stockout week (flag only) |

## Rules this code follows

- Phase 2 (held-out test) is only loaded for the shape check and the actuals export. `run_cleaning`
  refuses any date after `max_allowed_date` (2024-01-01), and `tests/test_no_leakage.py` checks that
  scrambling Phase 2 changes nothing the cleaning writes.
- Flag, don't overwrite: raw values stay next to adjusted ones.
- Every window is trailing (past only); `tests/test_cleaning.py` checks that adding future rows never
  changes earlier price features.
- New data sources need a new adapter that produces the same long schema; `cleaning.py` does not know about VN1.

# M5 frozen config (no M5-specific tuning)

Frozen by `python -m src.experiment_m5 --stage freeze` → `outputs/m5/test/frozen_config.json` (config + sha256 of every `src/*.py`). The test stage refuses to run if either changes, and runs once.

## Model (identical to the frozen VN1 model)

| setting | value |
|---|---|
| `features.scale_method` | `max13_52` |
| `features.optional_features` | `['uplift']` |
| `features.lags` | `[1, 2, 4, 8, 13]` |
| `features.roll_windows` | `[4, 8, 13, 26, 52]` |
| `lgbm.params` | `{objective: tweedie, verbosity: -1, seed: 0, deterministic: true, force_col_wise: true}` |
| `lgbm.weight_by_scale` | `True` |
| `lgbm.use_ids` | `False` |
| `lgbm.forecast_cap_k` | `3.0` |
| `lgbm.bias_multiplier` | `1.0` |
| `level_shift.restart_after_shift` | `True` |
| `intervals.method` | `conformal` |
| `backtest.step_weeks` | `2` |

Global direct multi-horizon LightGBM (tweedie, rows weighted by scale, 100 rounds, default leaves / rate), point-in-time features recomputed at every origin, training origins every 2 weeks from data start + 52 weeks, rows crossing a level shift dropped, forecasts capped at 3× the series' best week of the last 52, conformal 80% intervals by volume band × horizon.

## Every difference from configs/vn1.yaml

Settings (thresholds, features, model) that differ: **4** – listed below.

| key | VN1 | M5 | kind |
|---|---|---|---|
| backtest.folds.A | 2022-10-31 | – | data / bookkeeping |
| backtest.folds.B | 2023-01-30 | – | data / bookkeeping |
| backtest.folds.C | 2023-04-03 | – | data / bookkeeping |
| backtest.folds.D | 2023-07-03 | – | data / bookkeeping |
| backtest.holiday_folds | ['A'] | [] | data / bookkeeping |
| backtest.lgbm_grid.ids | [True, False] | – | data / bookkeeping |
| backtest.lgbm_grid.origins | ['full', 'weekly_only'] | – | data / bookkeeping |
| backtest.lgbm_grid.scale | ['mean13', 'max13_52'] | – | data / bookkeeping |
| backtest.origin_settings.full.start | 2021-07-05 | – | setting |
| backtest.origin_settings.weekly_only.start | 2022-09-05 | – | data / bookkeeping |
| backtest.output_dir | outputs/backtest | outputs/m5/backt | data / bookkeeping |
| baselines.output_dir | outputs/baseline | outputs/m5/basel | data / bookkeeping |
| config_path | configs/vn1.yaml | configs/m5.yaml | data / bookkeeping |
| dataset.adapter | vn1 | m5 | data / bookkeeping |
| dataset.adapter_args.calendar_file | – | calendar.csv | data / bookkeeping |
| dataset.adapter_args.prices_file | – | sell_prices.csv | data / bookkeeping |
| dataset.adapter_args.sales_file | – | sales_train_evaluation.csv | data / bookkeeping |
| dataset.group_level | client | store_id | data / bookkeeping |
| dataset.hierarchy | ['client', 'warehouse'] | ['state_id', 'store_id', 'cat_id', 'dept_id'] | data / bookkeeping |
| dataset.item_column | product | item_id | data / bookkeeping |
| dataset.name | vn1 | m5 | data / bookkeeping |
| experiment.sample_seed | – | 0 | data / bookkeeping |
| experiment.sample_series | – | 7000 | data / bookkeeping |
| final.min_fold_wins | 3 | – | data / bookkeeping |
| final.routing_blend_w | 0.5 | – | data / bookkeeping |
| final.routing_segments.demand_type | ['lumpy'] | – | data / bookkeeping |
| final.routing_segments.volume_band | ['top 1%'] | – | data / bookkeeping |
| final.selection_folds | ['B', 'C', 'D', 'E'] | – | data / bookkeeping |
| level_shift.group_keys | ['warehouse', 'client'] | ['state_id', 'store_id', 'cat_id', 'dept_id'] | setting |
| lgbm_smoke.objective | tweedie | – | data / bookkeeping |
| lgbm_smoke.seed | 0 | – | data / bookkeeping |
| lgbm_smoke.verbosity | -1 | – | data / bookkeeping |
| max_allowed_date | 2024-01-01 | 2016-02-15 | data / bookkeeping |
| paths.artifacts_dir | artifacts/vn1 | artifacts/m5 | data / bookkeeping |
| paths.files.phase0_price | Phase 0 - Price.csv | – | data / bookkeeping |
| paths.files.phase0_sales | Phase 0 - Sales.csv | – | data / bookkeeping |
| paths.files.phase1_price | Phase 1 - Price.csv | – | data / bookkeeping |
| paths.files.phase1_sales | Phase 1 - Sales.csv | – | data / bookkeeping |
| paths.files.phase2_sales | Phase 2 - Sales.csv | – | data / bookkeeping |
| paths.outputs_dir | outputs | outputs/m5 | data / bookkeeping |
| paths.processed_dir | data/processed | data/processed/m | data / bookkeeping |
| paths.raw_dir | data/raw/vn1 | data/raw/m5 | data / bookkeeping |
| paths.reports_dir | reports | reports/m5 | data / bookkeeping |
| price_rules.ignore | [{'column': 'client', 'value': 0, 'before': '2022-08-29'}] | [] | setting |
| seal.first_week | – | 2016-02-22 | data / bookkeeping |
| seal.last_week | – | 2016-05-16 | data / bookkeeping |
| seal.weeks | – | 13 | data / bookkeeping |
| smoothing.switch_date | 2022-08-29 | 2014-03-17 | setting |
| splits.inner_valid_origin | 2023-07-03 | – | data / bookkeeping |
| splits.origin_start | 2022-09-05 | – | data / bookkeeping |
| splits.origin_step_weeks | 2 | – | data / bookkeeping |
| splits.phase1_forecast_origin | 2023-10-02 | – | data / bookkeeping |

Notes:
- `smoothing.switch_date`: VN1 pinned the date its detector found; M5 follows the same procedure (estimated in the prepare step, then pinned). It is data, not a tuned value.
- `level_shift.group_keys`: the generic default (every hierarchy level), as VN1 (which lists both of its levels).
- `price_rules.ignore`: the VN1 rule is about one VN1 client; M5 has none.
- `backtest.origin_settings.full.start`: null = data start + 52 weeks, the same rule that gave VN1's date.

## Test setup

- sealed weeks: 2016-02-22 .. 2016-05-16; train on every week ≤ 2016-02-15, forecast origin 2016-02-15
- series: the same 7,000-series sample (seed 0)
- backtest folds: F1 2015-02-16, F2 2015-05-18, F3 2015-08-17, F4 2015-11-16
- best baseline (lowest mean backtest VN1 score): **blend_0.25** (moving_avg_13 0.5401, seasonal_naive 0.7095, blend_0.25 0.5216; engine 0.5230)

# Final config for Phase 1 (frozen 2026-11-25)

Frozen before Phase 1 was scored. Snapshot of the full config + sha256 of every `src/*.py`:
`outputs/phase1/frozen_config.json`; `python -m src.run_phase1` refuses to run if anything differs.
All evidence comes from Phase 0 backtest folds (A 2022-10-31 holiday, B 2023-01-30, C 2023-04-03,
D 2023-07-03); Phase 1 and Phase 2 were never used for a choice.

## The model

Global direct multi-horizon LightGBM: one row per (series, origin, horizon 1–13), target = units at origin + h.

| choice | value | evidence |
|---|---|---|
| Data | Phase 0 cleaned (`phase0_clean.parquet`); features recomputed from data ≤ origin at every origin | leak tests: scrambling the future / corrupting stored flags changes no feature |
| Training origins | every 2 weeks from 2021-07-05, all 13 targets ≤ forecast origin (Phase 1 run: 2021-07-05 → 2023-07-03, 53 origins) | backtest: `full` beats `weekly_only` (paired +0.39 worse for weekly_only incl. blow-ups; 0.482 vs 0.481 on B–D without ids, and full is the only setting that covers a holiday fold) |
| Scale | `max13_52` = max(mean last 13 weeks, mean last 52 weeks since first sale), floor 0.1 | equal to `mean13` (0.522 vs 0.521); chosen because it stops quiet series inflating scaled targets |
| Target / loss | scaled target, tweedie, rows weighted by scale | smoke test: unweighted 0.513 vs weighted 0.481 (inner fold) |
| Model params | LightGBM defaults (100 trees), seed 0, deterministic | not tuned (step 4 not done) |
| Client / warehouse features | **off** | with them: fold B blow-ups 2.34 / 3.22 (warehouse 251 jump memorised); paired effect −0.37 for removing them |
| Optional features | `uplift` on (`ly_uplift_series`, `ly_uplift_client`); `holiday_distance` off | round 2: average 0.516 vs 0.522 default, fold A 0.625 vs 0.644; holiday distance 0.523 (neutral) |
| Level-shift restart | **on** (`restart_after_shift`) | round 3: 0.5089 vs 0.5158; fold B 0.490 vs 0.515 |
| Bias multiplier | **1.0 (off)** | round 3 leave-one-fold-out: 0.5098 with vs 0.5089 without |
| Forecast cap | forecast ≤ 3 × series max weekly sales in last 52 weeks | neutral on the default (0.522 both); fixes blow-ups (3.22 → 1.00) |
| Post-processing order | model → × multiplier (1.0) → cap | |

Features (45): lags 1/2/4/8/13; rolling mean/std 4/8/13/26/52; max 13; trends 4/13 and 13/52; zero share 13/52;
weeks since last sale; age; seasonal lag 52 and 3-week mean; target-week ISO week, month, Thanksgiving / Cyber /
Christmas flags; product mean 13; client trend 4/13; client share 13; relative price; trusted-price share 13;
demand type at origin; smoothed and stockout shares 13/52; log scale; h; uplift ratios (series, client).
Definitions: `reports/features.md`.

## Cleaning choices that feed the model

| choice | value |
|---|---|
| Leading zeros | trimmed (weeks before first sale are not demand) |
| Smoothed era | before 2022-08-29 (changepoint on repeat rate), used as a feature, not removed |
| Suspected stockouts | ≥ 4 and ≤ 13 zero weeks after ≥ 2 units/week; flag only |
| Price shift | 8-week medians, 2.5×, ≥ 4 observed prices; client 0 prices before 2022-08-29 ignored |
| Level shift | 4-week vs previous 13-week mean, 10×, level ≥ 5; group event > 50% of ≥ 5 active series |

## Comparator

`blend_0.25` = 0.25 × seasonal naive + 0.75 × moving average 13 (weight chosen on the 4-fold mean).

## Expected performance (backtest, frozen setup)

| | A (holiday) | B | C | D | mean |
|---|---|---|---|---|---|
| frozen LightGBM | 0.624 | 0.490 | 0.448 | 0.474 | **0.509** |
| blend_0.25 | 0.789 | 0.701 | 0.510 | 0.528 | 0.632 |

Phase 1 (2023-10-09 → 2024-01-01) contains Thanksgiving and Christmas, so fold A is the closest rehearsal.

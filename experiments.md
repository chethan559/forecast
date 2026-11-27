# Experiments log

## 2026-11-25: Data cleaning stage (VN1)

**What:** built the VN1 adapter, VN1 metric, cleaning steps a–g, parquet outputs, data quality report and
tests (27 passing). Config: `src/config.py` defaults. Command: `python -m src.run_cleaning`.

**Data checks**
- 15,053 series (46 clients, 328 warehouses, 11,171 products), Phase 0 = 170 weeks, Phase 1 = 13, Phase 2 = 13.
  Same series in every file; no missing or negative sales; all sales are integers.
- Price is NaN exactly when sales = 0 (1,836,437 weeks, 0 exceptions). 4,755 weeks have price = 0 with sales > 0.

**Cleaning results (Phase 0)**
- Leading zeros: 12,829 series (85.2%), median 93 weeks; 1,139,262 raw rows (44.5%) trimmed. 0 series never sold.
  2,559,010 → 1,419,748 rows.
- Weekly-data switch: **2022-08-29**, from a one-changepoint fit on the weekly repeat rate (20.0% before → 10.1% after).
  Same date on Phase 0+1. Before it, new-month weeks repeat less (16.1%) than mid-month weeks (21.1%): monthly totals spread over weeks.
- `is_smoothed`: 63,230 weeks (4.5%), 3,623 series. `identical_run` after the switch (chance floor): 2.1% of weeks.
- Price shifts (8-wk medians, 2.5×, ≥4 obs per window): 367 series, 896 events, 302 series are client 0. Client-0 shifts
  flip up/down through the smoothed era and stop at the switch. `min_periods=1` gives 767 series (≈ the ~758 seen earlier).
- Suspected stockouts: 67,054 weeks (4.7%), 3,399 series, 5,557 runs. 37% of flagged weeks are in runs > 26 weeks (likely seasonal/dormant).
- Demand types (full history): intermittent 48.1%, smooth 19.3%, lumpy 19.3%, erratic 13.3%. Smooth + erratic = 89.7% of units.
- is_new 4,825 (32.1%); is_inactive 3,817 (25.4%).
- Volume: top 1% of series = 43.9% of units, top 10% = 84.9%.

**Validation-set observations (Phase 1, allowed; Phase 2 not used)**
- 763 of the 3,817 inactive series sell again in Phase 1, but together they are only 0.4% of Phase 1 units.
- Series that are `is_new` in Phase 0 make up 22.6% of Phase 1 units: young series matter a lot for the score.

**Bug caught by tests:** the first version reset `relative_price` at the back-dated shift week, which leaks up to
15 weeks of future prices. Now it resets at the (causal) detection week; the prefix-invariance test guards it.

**Open decisions:** training window (all vs weekly era), stockout handling (cap run length? mask?), price-shift
threshold (`min_periods` 4 vs 1). See the end of the data quality report.

## 2026-11-25: Config decisions applied, cleaning re-run

- `stockout.max_zero_run = 13`: suspected stockouts drop from 67,054 to **27,440 weeks (1.9%)**, 2,810 series, 4,233 runs.
- `price_shift.min_periods = 4` kept (367 series). Added `price_shift.ignore_prices_before = {0: "2022-08-29"}`
  (client 0's prices before the weekly switch are not trusted). Not applied anywhere yet: no model uses price.
- Other cleaning numbers unchanged (switch date 2022-08-29, trimming, demand types).

## 2026-11-25: Baselines (train Phase 0 → score Phase 1)

**What:** `src/models/baselines.py` (zero, moving_avg 4/8/13/26, seasonal_naive with MA13 fallback,
MA13 with inactive → 0), runner `python -m src.run_baselines`. Forecasts in `outputs/baselines/*.parquet`,
scores in `outputs/baselines/scores_*.csv`, report `reports/baselines.md`. 33 tests pass. Phase 2 not read.

| method | score | accuracy | bias | signed bias |
|---|---|---|---|---|
| moving_avg_13 | **0.6339** | 0.5476 | 0.0863 | −8.6% |
| moving_avg_13_inactive_zero | 0.6339 | 0.5476 | 0.0863 | −8.6% |
| moving_avg_8 | 0.6369 | 0.5404 | 0.0965 | −9.6% |
| moving_avg_4 | 0.6400 | 0.5440 | 0.0960 | −9.6% |
| moving_avg_26 | 0.6587 | 0.5717 | 0.0870 | −8.7% |
| seasonal_naive | 0.7599 | 0.6965 | 0.0635 | +6.3% |
| zero | 2.0000 | 1.0000 | 1.0000 | −100% (sanity check passes) |

**Key findings (MA13)**
- `inactive_zero` is identical to MA13 by construction (13-week mean of a series with no sales in 13 weeks is already 0).
- Error by group: smooth 46% of abs error (58% of units), erratic 34% (28%), lumpy 13% (8%), intermittent 7% (6%).
  Top 10% of series by volume = 66% of abs error. Worst 1% of series = 36% of abs error, worst 10% = 79%.
- Bias is low overall (−8.6%) but driven by the holiday peak: weeks 2023-11-20 → 12-11 hold 38% of Phase 1 units,
  MA13 under-forecasts them by 26%, and is +2% on the other 9 weeks.
- Seasonal naive tracks the peak (weekly totals within ~10%) but is noisy per series (accuracy 0.70) and
  explodes on inactive series (score 11.4: last year's sales copied onto series that stopped).
- Lumpy and low-volume series are under-forecast by 22–32%.
- Exploratory only, weight picked on Phase 1 itself so optimistic: 0.5×MA13 + 0.5×seasonal_naive scores 0.572
  (bias +1.1%). Needs a proper backtest on Phase 0 origins before it counts.

## 2026-11-25: Feature engineering for a global direct multi-horizon model

**What:** `src/features.py` (`make_features`), `src/splits.py` (inner validation + Phase 1 origins),
`src/run_features.py` (`python -m src.run_features`), LightGBM smoke test (`src/models/smoke_lgbm.py`).
Output `data/processed/features_inner.parquet`, report `reports/features.md`. 47 tests pass. Added `lightgbm==4.7.0`.

**Data rules enforced**
- Reads only `phase0_clean.parquet`. Test: every other processed file (incl. Phase 1/2) is chmod 000 and the
  whole build + smoke test still runs; plus a source scan for Phase 1/2 file names.
- Features use weeks ≤ origin only. Test: scramble y and price after T, re-run cleaning, rebuild: all features identical.
- Flags in the cleaned file and series_table were computed on all of Phase 0, so `make_features` ignores them
  and recomputes smoothed/stockout/price-regime/demand-type at each origin (test corrupts those columns: no change).
  Example of why: a zero run crossing T is "closed" in the cleaned file but still open at T.
- Switch date 2022-08-29 used as a fixed calendar constant for `smoothed_share_*` (it was found on Phase 0).

**Dataset:** 16 train origins (every 2 weeks, 2022-09-05 → 2023-04-03) + validation origin 2023-07-03.
2,499,211 rows (2,320,643 train, 178,568 valid) × 70 columns, 43 model features; 764 MB in memory (float32),
40 MB parquet. Build ≈ 20 s; build + smoke test (5 LightGBM fits) ≈ 1.5 min.

**Smoke test (inner validation, targets 2023-07-10 → 2023-10-02, scored on last 13 weeks of Phase 0)**

| model | score | bias |
|---|---|---|
| moving_avg_13 | 0.5004 | 0.0282 |
| LightGBM tweedie, default (spec) | 0.5127 | 0.0150 |
| … without client/warehouse | 0.4960 | 0.0245 |
| … weight = scale | 0.4807 | 0.0195 |
| … weight = scale, without client/warehouse | 0.4791 | 0.0121 |
| … scaled target clipped at 20 | 0.4785 | 0.0033 |

- Default LightGBM: `warehouse` = 71% of gain. Not a time leak. 0.54% of training rows (scaled target > 50,
  series quiet for 13 weeks so scale ≈ 0.1–0.15, then selling again) hold 69% of the summed scaled target;
  45% of them are warehouse 251. Unweighted tweedie chases them. Weighting by scale or clipping fixes it.
- Default model loses badly on the top 1% band (0.538 vs MA13 0.372, +15% bias), wins on erratic/intermittent/lumpy.
- Leak screen: highest |spearman| with scaled target is 0.56 (zero_share_13); nothing suspicious.
- 1,317 series first sell after the validation origin (2.5% of validation units): forecast 0 by every model.
- Inner window is easier than Phase 1 (MA13 0.500 here vs 0.634 on Phase 1, which contains the holiday peak);
  the training origins (Sep 2022 → Apr 2023) include only one holiday season.

## 2026-11-25: Backtest harness (steps 1–3)

**What:** `src/backtest.py` (`python -m src.backtest`), per-origin feature cache (`src/feature_cache.py`),
LightGBM moved to `src/models/lgbm.py` (weighted by scale by default), new scale `max13_52`
(max of 13- and 52-week means, floor 0.1), baselines blend and seasonal index. Report `reports/backtest.md`.
Folds (origin → 13 weeks): A 2022-10-31 (holiday), B 2023-01-30, C 2023-04-03, D 2023-07-03.
Training origins every 2 weeks, all targets ≤ fold origin; settings `full` (from 2021-07-05, 29–46 origins,
3.0–5.4M rows) and `weekly_only` (from 2022-09-05, 0–16 origins; fold A skipped). Run ≈ 25 min, peak 3.7 GB.

| method | A | B | C | D | mean |
|---|---|---|---|---|---|
| lgbm full mean13 noids | 0.642 | 0.511 | 0.446 | 0.484 | **0.521** |
| lgbm full max13_52 noids | 0.644 | 0.511 | 0.447 | 0.486 | 0.522 |
| lgbm full max13_52 ids | 0.625 | 0.536 | 0.542 | 0.471 | 0.543 |
| lgbm full mean13 ids | 0.635 | 0.563 | 0.557 | 0.474 | 0.557 |
| blend_0.25 (best baseline; w picked on fold mean) | 0.789 | 0.701 | 0.510 | 0.528 | 0.632 |
| moving_avg_13 | 0.835 | 0.765 | 0.483 | 0.500 | 0.646 |
| seasonal_naive | 0.813 | 0.726 | 0.728 | 0.774 | 0.760 |
| seasonal_index | 0.824 | 0.682 | 0.718 | 0.893 | 0.779 |
| lgbm weekly_only noids (mean13 / max13_52), B–D mean | – | | | | 0.482 / 0.486 |
| lgbm weekly_only ids (mean13 / max13_52), fold B | – | **2.338 / 3.222** | | | |

- LightGBM beats the best baseline by ~0.11 on average, and on every fold except C (0.446 vs MA13 0.483: still ahead).
- Scale: mean13 vs max13_52 makes no real difference (0.521 vs 0.522).

## 2026-11-25: Round 2: locked default, fold B blow-ups, cap, fold A, seasonality, blend

**Decision (locked in config `lgbm` / `features`):** training origins `full`, scale `max13_52`, NO client/warehouse.
Command `python -m src.experiment_round2`, report `reports/experiments_round2.md`, 7 min, peak 3.9 GB. Phase 0 only.

| config | A | B | C | D | average (signed bias) |
|---|---|---|---|---|---|
| blend_0.25 (best baseline) | 0.789 | 0.701 | 0.510 | 0.528 | 0.632 (−4.0%) |
| lgbm_default | 0.644 | 0.511 | 0.447 | 0.486 | 0.522 (−4.1%) |
| lgbm_default + cap k=2 | 0.644 | 0.511 | 0.447 | 0.486 | 0.522 (−4.2%) |
| **lgbm + uplift** | **0.625** | 0.515 | **0.446** | **0.476** | **0.516** (−3.9%) |
| lgbm + holiday_distance | 0.635 | 0.519 | 0.450 | 0.486 | 0.523 (−4.0%) |
| 0.5 × (lgbm + uplift) + 0.5 × blend_0.25 | 0.692 | 0.526 | 0.466 | 0.491 | 0.544 (−4.0%) |

- **Fold B blow-ups:** warehouse 251 (client 2) jumped ~533× on 2022-11-07 (98% of its series > 20×;
  65 new series that week). weekly_only fold B training = pre-jump scales + post-jump targets → model learned
  "warehouse 251 → ×hundreds" and applied it on top of an already post-jump scale. Warehouse 251 = 68% of
  that config's error; forecast 7.8M vs 3.3M actual. Cleaning has no sales level-shift detector: to add.
- **Cap (k × max weekly sales last 52 weeks):** neutral on the default (0.522 either way; ~9% of forecasts
  touched, mostly series with no sales in 52 weeks forced to 0; 0.1–0.2% of units removed). Fixes the
  blow-ups: 3.22 → 1.00, 2.34 → 1.05. Not adopted by the "better average" rule (0.0001 worse), but a cheap safety net.
- **Fold A:** LightGBM under-forecasts the peak (Thanksgiving → pre-Christmas −10.9%, baseline −31.3%),
  Christmas → New Year −8%, January −11%. Not the head: top 1% band −0.4%, top-10 series +5% at the peak.
  The gap is the tail: rest-90% band −21%, intermittent −49%, and 3.4% of units from series first selling
  after the origin (−100%).
- **Seasonality:** uplift ratios (series + client) help: average 0.522 → 0.516, fold A 0.644 → 0.625,
  fold D 0.486 → 0.476; fold B slightly worse (0.511 → 0.515). Each uplift feature ≈ 6% of gain.
  Holiday distance helps fold A only (0.635), neutral on average.
- **Blend with best baseline hurts** (0.544): the baseline is much weaker and drags LightGBM's holiday forecast down.

## 2026-11-25: Final decisions for Phase 1 (round 3 + freeze)

**Decisions taken from round 2 (config updated):**
- `features.optional_features = ["uplift"]`: best 4-fold average (0.516 vs 0.522) and the biggest gain on the
  holiday fold (0.644 → 0.625). Fold A probably *underrates* it: the only earlier holiday available to fold A
  is 2021, which is in the smoothed (monthly-spread) era, so last year's weekly peak shape is blurred. For
  Phase 1, last year's holiday (2022) is true weekly data.
- `lgbm.forecast_cap_k = 3`: a safety net, not a score lever. Neutral on normal models (removes ~0.1% of units)
  but turns the fold B blow-ups (3.22 / 2.34) into ~1.0.
- The "first sold after origin" gap (3.4% of fold A units, forecast 0) is **not pursued**: every series sold in
  Phase 0, so no series is new at the Phase 1 origin.

**Level-shift detection** (`src/cleaning.py` step h, config `level_shift`): series event when the 4-week mean is
> 10× or < 1/10 of the previous 13-week mean (higher level ≥ 5 units/week, 13-week cooldown); group event when
> 50% of a warehouse's/client's active series (≥ 5) shift in the same week. Phase 0: 5,638 series events in
3,667 series; 8 group events: warehouse 251 + client 2 on 2022-11-07 (100/163 series, up), plus 6 small
warehouses
(25, 116, 196, 204, 208, 324; 5–16 active series each). Stored as `level_shift*` columns, `level_shift_date` /
`group_level_shift_date` in series_table, and `level_shift_events.parquet`. Past-only (prefix-invariance test).

**Round 3** (`python -m src.experiment_round3`, `reports/experiments_round3.md`, 6 min):

| setting | A | B | C | D | average |
|---|---|---|---|---|---|
| restart_after_shift off | 0.625 | 0.515 | 0.447 | 0.476 | 0.5158 |
| **restart_after_shift on** | **0.624** | **0.490** | 0.448 | **0.474** | **0.5089** |

- Restart on: features restart at the latest group event; training rows crossing an event dropped
  (882–9,067 rows per fold, out of 3–5.4M). Kept **on** (average better; mainly fold B, the warehouse 251 fold).
- Bias multiplier, leave-one-fold-out (fit on 3 folds, grid 0.80–1.20): fitted factors 1.02 / 1.02 / 1.04 / 1.04;
  held-out mean 0.5098 with vs 0.5089 without → **not adopted** (stays 1.0). The all-fold optimum (1.02,
  0.497 in-sample) does not survive leave-one-out: it helps A and B, hurts C and D.

**Frozen config** (`reports/final_config.md`, snapshot + source hashes in `outputs/phase1/frozen_config.json`):
backtest mean **0.5089** (A 0.624, B 0.490, C 0.448, D 0.474). Best baseline blend_0.25: 0.632 (A 0.789).
No changes after this point.

## 2026-11-25: Phase 1 (run once, frozen config)

`python -m src.run_phase1` (60 s, peak 4.2 GB). 53 training origins (2021-07-05 → 2023-07-03), forecast origin
2023-10-02, Phase 1 weeks 2023-10-09 → 2024-01-01. Report `reports/phase1.md`, forecasts `outputs/phase1/`.

| model | score | accuracy | bias | signed bias |
|---|---|---|---|---|
| **frozen LightGBM** | **0.5712** | 0.5119 | 0.0593 | +5.9% |
| blend_0.25 | 0.5816 | 0.5327 | 0.0489 | −4.9% |
| (earlier, baselines stage) moving_avg_13 | 0.6339 | 0.5476 | 0.0863 | −8.6% |

- LightGBM wins by 0.010 (backtest predicted ~0.12). It is harder than the backtest mean (0.509) but easier than
  the holiday fold (0.624). The baseline did much better than its backtest (0.582 vs 0.632 / fold A 0.789):
  its seasonal-naive part copied a smoothed 2021 holiday in fold A, but a true weekly 2022 holiday in Phase 1.
- LightGBM catches the Thanksgiving jump (−5.7% vs baseline −26%) but stays too high afterwards
  (+19% early December, +19% Christmas, +23% New Year): net +5.9%.
- Loses on top 1% (0.476 vs 0.456) and lumpy (1.63 vs 1.38); wins on rest-90% (0.789 vs 0.947), erratic, top 1–10%.
- Top-10 error series ≈ 8% of error: client 38 over-forecast (ignored last year's Q4 dip), spiky/seasonal
  under-forecast (0_3_1154, 41_228_2866), young series extrapolated upward.
- Nothing changed after scoring. Phase 2 untouched.

## 2026-11-25: Final selection round (folds B–E) and Phase 2

**Selection** (`python -m src.experiment_final`, `reports/final_round.md`). Folds B, C, D + E = Phase 1; fold A for
reference only (smoothed prior-year holiday). Rule: best mean over B–E and beats frozen LightGBM on ≥ 3 of 4.

| candidate | A (ref.) | B | C | D | E | mean B–E | wins |
|---|---|---|---|---|---|---|---|
| **frozen LightGBM** | 0.624 | 0.490 | 0.448 | 0.474 | 0.571 | 0.496 | – |
| 0.7 LightGBM + 0.3 blend_0.25 | 0.660 | 0.480 | 0.455 | 0.481 | 0.529 | 0.486 | 2/4 |
| routing (top 1% + lumpy → 0.5 blend) | 0.649 | 0.497 | 0.464 | 0.479 | 0.553 | 0.498 | 1/4 |
| 0.5 LightGBM + 0.5 blend_0.25 | 0.691 | 0.536 | 0.467 | 0.490 | 0.509 | 0.500 | 1/4 |

Chosen: **frozen LightGBM** (the 0.7 blend's better mean comes from Phase 1 alone; fails 3-of-4).
Phase 1 refit reproduced the one-off forecast exactly (max abs diff 0.0).

**Phase 2, run once** (`python -m src.run_phase2`, `reports/phase2.md`, 3 min, peak 4.8 GB). Cleaning re-run on
Phase 0 + 1; 59 training origins 2021-07-05 → 2023-10-02; origin 2024-01-01.

| method | score | accuracy | bias | signed bias |
|---|---|---|---|---|
| **frozen LightGBM** | **0.4879** | 0.4446 | 0.0433 | −4.3% |
| blend_0.25 | 0.7844 | 0.5951 | 0.1894 | +18.9% |

- VN1 leaderboard winner 0.4637 (official scoring removed stockout weeks; ours can't, so ours is harsher).
- Baseline +19%: its MA13 carried the holiday level (~300k/week Oct–Dec) into a ~250k/week quarter.
- Warehouse 68 / client 38 collapsed from ~20k to ~20 units/week in mid-December 2023: group level-shift event
  (25/25 series, down) at the origin → restart → forecast 0, right for 10 of 13 weeks; it reopened mid-March
  (15–26k/week), which explains most of the −6% to −14% in the last three weeks.
- Segments: top 1% 0.454 (−9.8%), top 1–10% 0.435, rest 0.744; smooth 0.396, erratic 0.634, lumpy 1.155.
- Summary of the whole forecast layer: `reports/forecast_layer_summary.md`. Nothing changed after scoring.

## 2026-11-25: Comparison with the VN1 leaderboard

Official Phase 2 results (Nixtla nixtlar vignette, places only): 1st 0.4637, 2nd 0.4657, 3rd 0.4758, 4th 0.4774,
5th 0.4808; TimeGPT zero-shot 0.4651 (not an official entry). Ours 0.4879: just outside the top 5 if scoring is
identical (lower places unpublished). Stockout-week handling of the official metric not verified; our actuals
have no missing weeks. Added to `reports/forecast_layer_summary.md`.

## 2026-11-27: Reusable forecasting engine (no new modelling)

**What was built**
- `configs/defaults.yaml` + `configs/vn1.yaml` (all VN1 thresholds, dates, the client 0 price rule, the pinned
  smoothed-era date, folds, decisions); `src/config.py` is now a YAML loader (`get_config()` = VN1). Added PyYAML.
- `src/schema.py`: standard input (`unique_id, ds` Monday, `y`; optional `price`, `in_stock`, hierarchy columns)
  with clear validation errors (missing columns, non-Mondays, negatives, duplicates, gaps, changing hierarchy).
- Generalised code: hierarchy levels, item column and group level come from config (`dataset.*`); group features
  are named after the group level (VN1: `client_trend_4_13`, ... unchanged); price-trust rules are a config list;
  smoothed-era detection can be switched off or run automatically (VN1 pins the detected 2022-08-29);
  `in_stock` replaces inferred stockouts when present; the seasonal-index baseline uses the group level.
- `src/calendar.py`: US (Thanksgiving, Cyber week, Christmas) and AU (Black Friday, Cyber Monday, Click Frenzy
  = 2nd Tuesday of November, Christmas, Boxing Day, EOFY = weeks touching June, Mother's Day, Father's Day).
- `src/engine.py`: `fit` → artifact (LightGBM text models + meta.json with version, source hash, features,
  categories, config; conformal table); `predict` → `unique_id, ds, p50, p10, p90`; `backtest` → report.
- Adapters: `m5.py` (daily → Monday weeks, partial weeks dropped, day-weighted Walmart-week prices),
  `shopify_orders.py` (line items → weekly units per SKU; refunded / partially refunded / voided / cancelled
  orders excluded; gaps filled with 0; partial last week dropped), `long_csv.py` (column mapping).
  Configs `configs/m5.yaml`, `configs/shopify_example.yaml` (AU calendar). M5 checked on a synthetic M5-format
  fixture only (real M5 files are not in the repo); no M5 modelling.
- `src/run.py`: `python -m src.run --config configs/<name>.yaml --stage clean|backtest|fit|predict`.
- Tests: 109 (+2 slow). Slow regression (`pytest -m slow`): VN1 Phase 1 = 0.5711712613 and Phase 2 =
  0.4878877058 reproduce exactly (plus blend_0.25 0.5816 / 0.7844).

**Prediction intervals** (`python -m src.experiment_intervals`, `reports/intervals.md`, 13 min, peak 4.4 GB).
Folds B, C, D + E (Phase 1). Engine p50 reproduced every frozen fold score exactly.

| method | B | C | D | E | mean coverage | mean relative width |
|---|---|---|---|---|---|---|
| **conformal** (13-week calibration window, band × horizon) | 82.8% | 73.8% | 80.9% | 71.6% | **77.3%** | 1.67 |
| quantile LightGBM (alpha 0.1 / 0.9) | 85.6% | 86.4% | 87.1% | 88.9% | 87.0% | **1.37** |

- Rule (closest to 80%; narrower if both within 3 points): **conformal** (2.7 points off vs 7.0).
  `intervals.method: conformal` in the configs.
- Caveats: conformal under-covers the holiday quarter (71.6%: calibrated on Jul–Sep) and gives the top 1%
  too-wide intervals (96% coverage): with ~150 series it falls below the 200-row minimum per group and uses
  horizon-level residuals dominated by small series. Quantile LightGBM is narrower and steadier but
  over-covers, mostly at the bottom (2.7% below p10). Candidate next steps: lower the conformal group minimum,
  calibrate on the same season last year, or a conformal correction on top of the quantile models.

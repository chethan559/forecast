"""All tunable thresholds and paths live here, so the same cleaning code can run on
VN1 now and on Shopify exports later with a different config (and adapter).

Keep this a plain dict: easy to print, diff, copy into experiments.md, or dump to YAML.
"""
from copy import deepcopy
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

CONFIG = {
    "paths": {
        "raw_dir": PROJECT_ROOT / "data" / "raw" / "vn1",
        "processed_dir": PROJECT_ROOT / "data" / "processed",
        "reports_dir": PROJECT_ROOT / "reports",
        "files": {
            "phase0_sales": "Phase 0 - Sales.csv",
            "phase0_price": "Phase 0 - Price.csv",
            "phase1_sales": "Phase 1 - Sales.csv",
            "phase1_price": "Phase 1 - Price.csv",
            "phase2_sales": "Phase 2 - Sales.csv",  # held-out test: actuals export + shape check ONLY
        },
    },
    # Leakage guard: run_cleaning refuses any data dated after this (last Phase 1 week).
    # Phase 2 starts 2024-01-08, so cleaning can never see it.
    "max_allowed_date": "2024-01-01",
    "freq": "W-MON",

    # (b) smoothed-era detection
    "smoothing": {
        "min_run_length": 3,          # >=3 identical consecutive positive weeks = suspicious run
        "switch_date": None,          # None = estimate from data; or force e.g. "2022-08-29"
        "switch_min_segment_weeks": 13,  # changepoint must leave >=13 weeks on each side
    },

    # (d) price level shifts
    "price_shift": {
        "window": 8,                  # rolling median window (weeks)
        "ratio_threshold": 2.5,       # flag if new/old median > 2.5 or < 1/2.5
        "min_periods": 4,             # decision 2026-11-25: need >=4 observed prices in each window
        # decision 2026-11-25: client 0 prices before the weekly switch flip ~3x up and down (looks like an
        # aggregation artefact), so they should not be used. Not applied yet: baselines don't use price.
        "ignore_prices_before": {0: "2022-08-29"},   # {client: first date whose prices are trusted}
    },

    # (e) relative price
    "relative_price": {
        "window": 13,                 # past 13 weeks, current week excluded
        "min_periods": 1,
    },

    # (f) suspected stockouts
    "stockout": {
        "min_zero_run": 4,            # >=4 consecutive zero weeks
        "max_zero_run": 13,           # decision 2026-11-25: runs longer than 13 weeks look more like an
                                      # off-season or dormancy than a stockout, so they are not flagged.
                                      # None = no cap.
        "pre_window": 13,             # look-back window before the run
        "min_pre_mean": 2.0,          # trailing mean before the run must be >= 2 units/week
        "min_pre_periods": 4,         # need >=4 weeks of history before the run to judge
        "include_trailing_run": False,  # a zero run still open at the end could be a delisting,
                                        # so by default only runs that end with a sale are flagged
    },

    # (h) sales level shifts (e.g. warehouse 251 jumped ~500x on 2022-11-07)
    "level_shift": {
        "short_window": 4,            # recent level = mean of the last 4 weeks (incl. this week)
        "long_window": 13,            # previous level = mean of the 13 weeks before those 4
        "ratio": 10.0,                # shift if recent > 10 x previous or recent < previous / 10
        "min_level": 5.0,             # the higher of the two levels must be >= 5 units/week (ignores 0 -> 1 noise)
        "cooldown_weeks": 13,         # firings within 13 weeks of an event belong to that event
        "group_keys": ["warehouse", "client"],
        "group_share": 0.5,           # group event: > 50% of the group's active series shift in the same week
        "min_group_series": 5,        # ... and the group has >= 5 active series that week
        # On: (1) features at an origin use only weeks from the latest group event of the series'
        # warehouse/client, (2) training rows whose (origin, target] window contains such an event are dropped.
        # decision 2026-11-25 (round 3): ON. 4-fold average 0.5089 vs 0.5158 off (fold B 0.515 -> 0.490).
        "restart_after_shift": True,
    },

    # (g) series classification (Syntetos-Boylan)
    "classification": {
        "adi_threshold": 1.32,
        "cv2_threshold": 0.49,
        "new_series_weeks": 52,       # < 52 weeks of history after trim = new
        "inactive_weeks": 13,         # no sales in the last 13 weeks = inactive
        "from_date": None,            # None = full trimmed history; e.g. switch date for weekly-era only
    },
    # Baselines (src/models/baselines.py, src/run_baselines.py)
    "baselines": {
        "h": 13,
        "moving_avg_windows": [4, 8, 13, 26],
        "season_length": 52,          # seasonal naive looks back 52 weeks (364 days)
        "fallback_window": 13,        # seasonal naive falls back to moving_avg(13) without 52 weeks of history
        "inactive_weeks": 13,         # "inactive" = no sales in the last 13 training weeks
        "volume_bands": [0.01, 0.10], # top 1%, next 9% (top 10%), rest 90%
        "output_dir": PROJECT_ROOT / "outputs" / "baselines",
    },
    # Features for the global direct multi-horizon model (src/features.py, src/splits.py)
    "features": {
        "horizons": list(range(1, 14)),
        "lags": [1, 2, 4, 8, 13],              # lag k = y at origin - (k - 1): lag 1 is the origin week itself
        "roll_windows": [4, 8, 13, 26, 52],    # rolling mean/std over the last k weeks up to the origin
        "max_window": 13,
        "season_length": 52,
        "scale_window": 13,                    # weeks in the "mean13" part of the scale
        # decision 2026-11-25 (backtest step 1): "max13_52" = max(mean13, mean52 since first sale).
        # "mean13" (the original) is kept for comparison.
        "scale_method": "max13_52",
        # Optional feature groups (see features.feature_specs): "uplift", "holiday_distance". Off by default.
        # decision 2026-11-25 (final for Phase 1): "uplift" on. Best 4-fold average (0.516 vs 0.522) and
        # the biggest gain on the holiday fold (0.644 -> 0.625).
        "optional_features": ["uplift"],
        "scale_floor": 0.1,                    # units/week; stops near-zero scales blowing up scaled values
        "demand_type_window": 52,              # ADI / CV2 at origin use the last 52 weeks only
        # The weekly-data switch is a fixed property of the VN1 dataset (found in the cleaning EDA),
        # used as a calendar constant: is_smoothed = identical run AND week < this date.
        "switch_date": "2022-08-29",
    },
    "splits": {
        "origin_start": "2022-09-05",          # first training origin (first full week of true weekly data)
        "origin_step_weeks": 2,
        "inner_valid_origin": "2023-07-03",    # inner validation: targets 2023-07-10 .. 2023-10-02 (end of Phase 0)
        "phase1_forecast_origin": "2023-10-02",  # last Phase 0 week; forecasts Phase 1 (not scored here)
    },
    # LightGBM (src/models/lgbm.py): default params + fixed seed/determinism so reruns give the same scores.
    "lgbm": {
        "params": {"objective": "tweedie", "verbosity": -1, "seed": 0, "deterministic": True,
                   "force_col_wise": True},
        "weight_by_scale": True,       # rows weighted by scale: the loss counts units like the VN1 score
        # decision 2026-11-25 (after the backtest): default = full-history origins, max13_52 scale,
        # NO client/warehouse features. With them, warehouse 251's ~500x volume jump on 2022-11-07 was
        # memorised as a warehouse multiplier (fold B scores 2.34 / 3.22).
        "use_ids": False,
        "origins_setting": "full",     # key of backtest.origin_settings used for the default model
        # decision 2026-11-25: cap at 3 x max weekly sales of the last 52 weeks. Neutral on normal models,
        # but turns the fold B blow-ups (3.22 / 2.34) into ~1.0. A safety net, not a score lever.
        "forecast_cap_k": 3.0,
        # forecasts x this factor before the cap. decision 2026-11-25 (round 3): NOT adopted (1.0).
        # Leave-one-fold-out mean 0.5098 with vs 0.5089 without; the all-fold optimum (1.02) does not generalise.
        "bias_multiplier": 1.0,
    },
    "lgbm_smoke": {"objective": "tweedie", "verbosity": -1, "seed": 0},

    # Final forecast layer (src/final_model.py). "method" is chosen in the final selection round
    # (src/experiment_final.py) on folds B, C, D, E (E = Phase 1); fold A is reported but not used.
    "final": {
        "method": "lgbm",                      # one of final_model.CANDIDATES
        "baseline": "blend_0.25",
        "routing_blend_w": 0.5,                # routed series get w x LightGBM + (1 - w) x baseline
        "routing_segments": {"volume_band": ["top 1%"], "demand_type": ["lumpy"]},
        "selection_folds": ["B", "C", "D", "E"],
        "min_fold_wins": 3,                    # a candidate must beat frozen LightGBM on >= 3 of the 4 folds
    },

    # Backtest harness (src/backtest.py). Phase 0 only.
    "backtest": {
        "folds": {"A": "2022-10-31", "B": "2023-01-30", "C": "2023-04-03", "D": "2023-07-03"},
        "holiday_folds": ["A"],
        "origin_settings": {
            "full": {"start": "2021-07-05"},          # one year after data starts; includes the smoothed era
            "weekly_only": {"start": "2022-09-05"},   # true weekly data only
        },
        "step_weeks": 2,
        "step_weeks_before_switch": 2,  # raise to 4 if memory is tight (thins smoothed-era origins)
        "lgbm_grid": {"scale": ["mean13", "max13_52"], "ids": [True, False], "origins": ["full", "weekly_only"]},
        "blend_weights": [0.25, 0.5, 0.75],
        "seasonal_index_clip": [0.25, 4.0],          # cap the client seasonal ratio (small clients are noisy)
        "output_dir": PROJECT_ROOT / "outputs" / "backtest",
    },
}


def get_config(overrides: dict | None = None) -> dict:
    """Return a deep copy of CONFIG with nested overrides applied (so tests can't mutate the default)."""
    cfg = deepcopy(CONFIG)

    def _merge(base, upd):
        for k, v in upd.items():
            if isinstance(v, dict) and isinstance(base.get(k), dict):
                _merge(base[k], v)
            else:
                base[k] = v

    if overrides:
        _merge(cfg, overrides)
    return cfg

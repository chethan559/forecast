import os
import re
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.adapters.vn1 import load_phase
from src.cleaning import run_cleaning
from src.config import get_config
from src.features import (BASE_COLUMNS, TARGET_COLUMNS, make_features, model_features, scaled_features,
                          us_holiday_weeks)
from src.run_cleaning import build_outputs
from src.splits import inner_validation_split, phase1_split
from tests.conftest import ROOT, make_panel

CFG = get_config()
H = CFG["features"]["horizons"]
ORIGINS = [pd.Timestamp("2022-09-05"), pd.Timestamp("2023-04-03")]


@pytest.fixture(scope="module")
def raw_phase0(small_raw):
    return load_phase(small_raw, "Phase 0 - Sales.csv", "Phase 0 - Price.csv")


def _clean(raw):
    return run_cleaning(raw, None, CFG)[0]


# ---------------------------------------------------------------- shift the future
@pytest.mark.parametrize("T", ORIGINS)
def test_changing_the_future_does_not_change_features(raw_phase0, T):
    """Scramble y and price after T, re-run cleaning (so precomputed flags change too), rebuild.
    Every feature at origin T must be identical; only the target may differ."""
    rng = np.random.default_rng(0)
    fut = raw_phase0["ds"] > T
    alt = raw_phase0.copy()
    alt.loc[fut, "y"] = rng.integers(0, 500, fut.sum()).astype(float)
    alt.loc[fut, "price"] = rng.uniform(0.5, 200, fut.sum())
    a = make_features(_clean(raw_phase0), [T], H, CFG)
    b = make_features(_clean(alt), [T], H, CFG)
    feature_cols = [c for c in a.columns if c not in TARGET_COLUMNS]
    pd.testing.assert_frame_equal(a[feature_cols], b[feature_cols])
    assert not np.allclose(a["target_raw"].fillna(-1), b["target_raw"].fillna(-1))  # the test did change something


def test_precomputed_flag_columns_are_ignored(raw_phase0):
    """Corrupting the full-history flags in the cleaned file must not change any feature."""
    clean = _clean(raw_phase0)
    bad = clean.copy()
    for c in ["is_smoothed", "is_suspected_stockout", "identical_run", "price_shift", "price_shift_detected"]:
        bad[c] = ~bad[c]
    bad["relative_price"] = 999.0
    bad["price_regime"] = 7
    pd.testing.assert_frame_equal(make_features(clean, ORIGINS, H, CFG), make_features(bad, ORIGINS, H, CFG))


# ---------------------------------------------------------------- row counts and inclusion
def test_row_counts_and_inclusion(raw_phase0):
    clean = _clean(raw_phase0)
    X = make_features(clean, ORIGINS, H, CFG)
    first_sale = clean[clean["y"] > 0].groupby("unique_id")["ds"].min()
    for T in ORIGINS:
        at = X[X["origin"] == T]
        expected = set(first_sale.index[first_sale <= T])
        assert set(at["unique_id"]) == expected  # only series that had sold by T
        assert len(at) == len(expected) * len(H)
    assert not X.duplicated(["unique_id", "origin", "h"]).any()
    assert (X["target_date"] == X["origin"] + pd.to_timedelta(X["h"].astype(int) * 7, unit="D")).all()


def test_target_and_scaling(raw_phase0):
    clean = _clean(raw_phase0)
    X = make_features(clean, ORIGINS[:1], H, CFG)
    y = clean.set_index(["unique_id", "ds"])["y"]
    lookup = y.reindex(pd.MultiIndex.from_arrays([X["unique_id"], X["target_date"]])).to_numpy()
    np.testing.assert_allclose(X["target_raw"], lookup, rtol=1e-6)
    np.testing.assert_allclose(X["target"], X["target_raw"] / X["scale"], rtol=1e-5)
    for c in scaled_features(CFG):
        np.testing.assert_allclose(X[c], X[f"{c}_raw"] / X["scale"], rtol=1e-5, equal_nan=True)
    assert (X["scale"] >= CFG["features"]["scale_floor"]).all()


def test_no_target_leakage_columns():
    feats = model_features(CFG)
    forbidden = {"target", "target_raw", "target_date", "origin", "unique_id", "y", "scale"}
    assert not forbidden & set(feats)
    assert not [f for f in feats if f.endswith("_raw") or "target" in f]
    assert len(feats) == len(set(feats))


# ---------------------------------------------------------------- no Phase 1 / Phase 2 file is opened
# File names / config keys of Phase 1+2 data (function names like phase1_split are fine).
FORBIDDEN = re.compile(r"phase0?[12]_(clean|actuals|sales|price)|series_table_phase01|Phase [12] - ")
CODE = ["src/features.py", "src/splits.py", "src/run_features.py", "src/models/lgbm.py", "src/evaluation.py",
        "src/backtest.py", "src/feature_cache.py"]


def test_feature_code_names_no_phase1_or_phase2_file():
    for f in CODE:
        assert not FORBIDDEN.search((ROOT / f).read_text()), f"{f} mentions a Phase 1/2 file"


def test_pipeline_runs_with_phase1_and_phase2_files_unreadable(small_raw, tmp_path):
    from src.run_features import build
    processed = tmp_path / "processed"
    build_outputs(CFG, raw_dir=small_raw, out_dir=processed, make_report=False)
    locked = [p for p in processed.iterdir() if p.name != "phase0_clean.parquet"]
    assert any("phase1" in p.name for p in locked) and any("phase2" in p.name for p in locked)
    for p in locked:
        os.chmod(p, 0)  # any attempt to open them now raises PermissionError
    try:
        out = build(CFG, processed_dir=processed, reports_dir=tmp_path / "reports", run_smoke=True)
    finally:
        for p in locked:
            os.chmod(p, 0o644)
    assert out["smoke"]["lgbm"]["score"] > 0
    assert (processed / "features_inner.parquet").exists()


# ---------------------------------------------------------------- splits and calendar
def test_inner_split_targets_stay_before_validation():
    s = inner_validation_split(CFG)
    last_train_target = max(s["train_origins"]) + pd.Timedelta(weeks=max(H))
    assert last_train_target <= s["valid_origin"]
    assert s["valid_target_end"] == pd.Timestamp("2023-10-02")  # last week of Phase 0
    assert len(s["train_origins"]) == 16


def test_phase1_split():
    s = phase1_split(CFG)
    assert max(s["train_origins"]) <= pd.Timestamp("2023-07-03")
    assert max(s["train_origins"]) + pd.Timedelta(weeks=13) <= s["forecast_origin"]
    assert s["forecast_origin"] == pd.Timestamp("2023-10-02")


def test_holiday_weeks():
    h = us_holiday_weeks([2022, 2023])
    assert pd.Timestamp("2023-11-20") in h["thanksgiving"]  # Thanksgiving 2023-11-23
    assert pd.Timestamp("2023-11-27") in h["cyber"]
    assert pd.Timestamp("2023-12-25") in h["christmas"]
    assert pd.Timestamp("2022-11-21") in h["thanksgiving"]  # Thanksgiving 2022-11-24
    assert pd.Timestamp("2022-12-19") in h["christmas"]     # 25 Dec 2022 is a Sunday


# ---------------------------------------------------------------- synthetic feature logic
def _synthetic(series, start="2022-01-03", client_of=None):
    df = make_panel(series, start=start).rename(columns={"price": "price_raw"})
    if client_of:
        df["client"] = df["unique_id"].map(client_of)
    return df


def test_lags_windows_and_missing_before_first_sale():
    ys = [0] * 10 + list(range(1, 61))  # first sale at week 10
    df = _synthetic({"1_1_1": ys})
    T = df["ds"].iloc[-1]
    X = make_features(df, [T], [1, 2], CFG)
    r = X.iloc[0]
    assert r["lag_1_raw"] == 60 and r["lag_2_raw"] == 59 and r["lag_13_raw"] == 48
    assert r["roll_mean_4_raw"] == pytest.approx(np.mean([57, 58, 59, 60]))
    assert r["scale"] == pytest.approx(np.mean(range(48, 61)))
    assert r["age_weeks"] == 59 and r["weeks_since_last_sale"] == 0
    assert r["zero_share_52"] == 0  # pre-first-sale zeros are not counted as demand
    # seasonal lag for h=1: target week - 52 = T - 51 -> value 60 - 51 = 9
    assert r["seas_lag_52_raw"] == 9


def test_client0_prices_before_cutoff_are_ignored():
    n = 30
    df = _synthetic({"0_1_1": [5] * n, "1_1_1": [5] * n}, start="2022-04-04")
    T = pd.Timestamp("2022-08-22")  # every week <= T is before 2022-08-29
    X = make_features(df, [T], [1], CFG).set_index("unique_id")
    assert X.loc["0_1_1", "price_observed_share_13"] == 0 and np.isnan(X.loc["0_1_1", "relative_price"])
    assert X.loc["1_1_1", "price_observed_share_13"] == 1 and X.loc["1_1_1", "relative_price"] == pytest.approx(1)


def test_requires_raw_columns():
    with pytest.raises(ValueError, match="missing input columns"):
        make_features(pd.DataFrame(columns=["unique_id", "ds", "y"]), ORIGINS, H, CFG)


# ---------------------------------------------------------------- optional seasonality features
def test_optional_features_follow_config():
    assert "ly_uplift_series" in model_features(CFG)              # final default: uplift on
    assert "tw_weeks_to_christmas" not in model_features(CFG)     # holiday_distance off
    assert "ly_uplift_series" not in model_features(get_config({"features": {"optional_features": []}}))
    cfg = get_config({"features": {"optional_features": ["uplift", "holiday_distance"]}})
    f = model_features(cfg)
    assert {"ly_uplift_series", "ly_uplift_client", "tw_weeks_to_thanksgiving", "tw_weeks_to_christmas"} <= set(f)


def test_last_year_uplift_and_holiday_distance():
    n = 80
    ys = np.full(n, 4.0)
    T_idx = n - 1
    ys[T_idx - 52 + 1 - 1: T_idx - 52 + 1 + 2] = 12.0  # last year's weeks around target h=1 are 3x the level
    df = _synthetic({"1_1_1": list(ys), "1_1_2": list(np.full(n, 4.0))})
    T = df["ds"].max()
    X = make_features(df, [T], [1], CFG).set_index("unique_id")
    # Last year's level window (T-64..T-52) shares week T-52 with the h=1 target window (T-52..T-50).
    assert X.loc["1_1_1", "ly_uplift_series"] == pytest.approx(12 / ((12 * 4 + 12) / 13))
    assert X.loc["1_1_2", "ly_uplift_series"] == pytest.approx(1.0)
    assert X.loc["1_1_1", "ly_uplift_client"] == pytest.approx(16 / ((12 * 8 + 16) / 13))  # client = both series
    h = make_features(_synthetic({"1_1_1": [1.0] * 60}, start="2022-09-05"), [pd.Timestamp("2022-11-07")],
                      [1, 2, 3, 6], CFG).set_index("h")
    # targets 11-14, 11-21 (Thanksgiving week 2022), 11-28, 12-19 (Christmas week 2022)
    assert h["tw_weeks_to_thanksgiving"].tolist() == [1, 0, -1, -4]
    assert h["tw_weeks_to_christmas"].tolist() == [5, 4, 3, 0]


def test_restart_after_shift_cuts_history_and_stays_past_only(raw_phase0):
    """With restart on, features still don't change when the future is scrambled."""
    cfg = get_config({"level_shift": {"restart_after_shift": True}})
    T = ORIGINS[1]
    rng = np.random.default_rng(1)
    alt = raw_phase0.copy()
    fut = alt["ds"] > T
    alt.loc[fut, "y"] = rng.integers(0, 500, fut.sum()).astype(float)
    a = make_features(_clean(raw_phase0), [T], H, cfg)
    b = make_features(_clean(alt), [T], H, cfg)
    cols = [c for c in a.columns if c not in TARGET_COLUMNS]
    pd.testing.assert_frame_equal(a[cols], b[cols])


def test_restart_and_row_drop_on_synthetic_group_jump():
    from src.features import drop_rows_crossing_shift, group_shift_dates_upto
    cfg = get_config({"level_shift": {"restart_after_shift": True}})
    base = [1.0, 2.0, 1.0, 0.0] * 6
    series = {f"2_9_{i}": base + [500.0] * 10 for i in range(6)}
    df = _synthetic(series, start="2022-01-03")
    jump = df["ds"].unique()[24]
    T = df["ds"].max()
    X = make_features(df, [T], [1], cfg)
    assert (X["age_weeks"] == 9).all()  # history restarts at the jump week
    assert X["roll_mean_52_raw"].eq(500).all()
    dates = group_shift_dates_upto(df, T, cfg)
    assert set(dates["shift_ds"]) == {jump}
    rows = pd.DataFrame({"unique_id": ["2_9_0"] * 3, "origin": [jump - 2 * WEEK_TD, jump, jump - 8 * WEEK_TD],
                         "target_date": [jump + WEEK_TD, jump + 3 * WEEK_TD, jump - 2 * WEEK_TD]})
    kept = drop_rows_crossing_shift(rows, dates)
    assert len(kept) == 2 and kept["origin"].tolist() == [jump, jump - 8 * WEEK_TD]


WEEK_TD = pd.Timedelta(weeks=1)

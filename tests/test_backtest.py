import os

import numpy as np
import pandas as pd
import pytest

from src.backtest import fold_plan, run_backtest, training_origins
from src.config import get_config
from src.features import make_raw_features
from src.models.baselines import blend, moving_avg, seasonal_index
from src.run_cleaning import build_outputs
from tests.conftest import make_panel

CFG = get_config()
WEEK = pd.Timedelta(weeks=1)


# ---------------------------------------------------------------- folds
def test_every_training_target_is_before_its_fold_origin():
    for p in fold_plan(CFG):
        if p["train_origins"]:
            assert max(p["train_origins"]) + 13 * WEEK <= p["origin"], p
            assert min(p["train_origins"]) >= pd.Timestamp(CFG["backtest"]["origin_settings"][p["setting"]]["start"])


def test_weekly_only_has_no_origins_for_holiday_fold():
    plan = {(p["fold"], p["setting"]): p["train_origins"] for p in fold_plan(CFG)}
    assert plan[("A", "weekly_only")] == []
    assert len(plan[("A", "full")]) == 29 and len(plan[("D", "weekly_only")]) == 16


def test_wider_step_before_switch_thins_only_old_origins():
    cfg = get_config({"backtest": {"step_weeks_before_switch": 4}})
    o = training_origins("2021-07-05", "2023-07-03", cfg)
    switch = pd.Timestamp(cfg["features"]["switch_date"])
    old, new = [t for t in o if t < switch], [t for t in o if t >= switch]
    assert all(b - a == 4 * WEEK for a, b in zip(old, old[1:]))
    assert all(b - a == 2 * WEEK for a, b in zip(new, new[1:]))


# ---------------------------------------------------------------- baselines and scale
def test_blend():
    a = pd.DataFrame({"unique_id": ["x", "x"], "ds": pd.to_datetime(["2024-01-01", "2024-01-08"]), "forecast": [10.0, 0.0]})
    b = a.assign(forecast=[0.0, 4.0])
    assert blend(a, b, 0.25)["forecast"].tolist() == [2.5, 3.0]


def test_seasonal_index_applies_last_years_client_shape():
    # 2 series of client 1; last year the target weeks were 2x the 13 weeks before last year's origin.
    n = 80
    ys = np.full(n, 5.0)
    origin_idx = n - 1
    ly_targets = range(origin_idx - 52 + 1, origin_idx - 52 + 14)
    for i in ly_targets:
        ys[i] = 10.0
    df = make_panel({"1_1_1": list(ys), "1_1_2": list(ys)})
    fc = seasonal_index(df, 13).set_index(["unique_id", "ds"])["forecast"]
    ma = moving_avg(df, 13, 13).set_index(["unique_id", "ds"])["forecast"]
    np.testing.assert_allclose(fc.to_numpy(), 2 * ma.to_numpy())


def test_max13_52_scale_keeps_quiet_series_realistic():
    ys = [10.0] * 47 + [0.0] * 13  # sold 10/week, then quiet for 13 weeks
    df = make_panel({"1_1_1": ys}).rename(columns={"price": "price_raw"})
    X = make_raw_features(df, [df["ds"].max()], [1], CFG).iloc[0]
    assert X["scale_max13_52"] == pytest.approx(390 / 52)   # 39 weeks of 10 in the last 52
    assert X["scale_mean13"] == pytest.approx(470 / 60)     # mean13 = 0 -> mean since first sale


# ---------------------------------------------------------------- end to end
SMALL = {"backtest": {"lgbm_grid": {"scale": ["mean13", "max13_52"], "ids": [True, False],
                                    "origins": ["full", "weekly_only"]}}}


def test_backtest_phase0_only_and_deterministic(small_raw, tmp_path):
    cfg = get_config(SMALL)
    processed = tmp_path / "processed"
    build_outputs(cfg, raw_dir=small_raw, out_dir=processed, make_report=False)
    locked = [p for p in processed.iterdir() if p.is_file() and p.name != "phase0_clean.parquet"]
    assert any("phase1" in p.name for p in locked) and any("phase2" in p.name for p in locked)
    for p in locked:
        os.chmod(p, 0)  # any read of Phase 1/2 (or anything but phase0_clean) now fails
    try:
        r1 = run_backtest(cfg, processed_dir=processed, out_dir=tmp_path / "o1", reports_dir=tmp_path)
        r2 = run_backtest(cfg, processed_dir=processed, out_dir=tmp_path / "o2", reports_dir=tmp_path, write_report=False)
    finally:
        for p in locked:
            os.chmod(p, 0o644)
    pd.testing.assert_frame_equal(r1["scores"], r2["scores"])  # same inputs -> same scores
    sc = r1["scores"]
    assert not ((sc["fold"] == "A") & (sc["method"].str.contains("weekly_only"))).any()  # skipped, not faked
    assert any("weekly_only" in s for s in r1["run_info"]["skipped"])
    assert (sc.loc[sc["family"] == "lgbm", "score"] > 0).all()
    assert (tmp_path / "backtest.md").exists()


def test_cache_key_ignores_apply_time_settings(tmp_path):
    from src.feature_cache import cache_key
    (tmp_path / "phase0_clean.parquet").write_bytes(b"x")
    base = cache_key(CFG, tmp_path)
    assert cache_key(get_config({"features": {"scale_method": "mean13", "optional_features": ["uplift"]}}), tmp_path) == base
    assert cache_key(get_config({"features": {"scale_floor": 0.5}}), tmp_path) != base


def test_cap_forecasts():
    from src.models.postprocess import cap_forecasts
    weekly = make_panel({"1_1_1": [1, 5, 2], "1_1_2": [0, 0, 0]})
    T = weekly["ds"].max()
    fc = pd.DataFrame({"unique_id": ["1_1_1", "1_1_1", "1_1_2"], "ds": [T + WEEK, T + 2 * WEEK, T + WEEK],
                       "forecast": [12.0, 3.0, 4.0]})
    out, st = cap_forecasts(fc, weekly, T, k=2)
    assert out["forecast"].tolist() == [10.0, 3.0, 0.0]  # 2 x max(1,5,2) = 10; no sales -> 0
    assert st["n_capped"] == 2 and st["series_capped"] == 2


def test_routing_uses_blend_only_for_routed_segments():
    from src.final_model import routed_ids
    seg = pd.DataFrame({"volume_band": ["top 1%", "rest 90%", "rest 90%"], "demand_type": ["smooth", "lumpy", "smooth"]},
                       index=["a", "b", "c"])
    assert list(routed_ids(seg, CFG)) == ["a", "b"]

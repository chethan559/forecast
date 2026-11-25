import numpy as np
import pandas as pd
import pytest

from src.adapters.vn1 import load_phase
from src.cleaning import run_cleaning
from src.config import get_config
from src.models.baselines import (moving_avg, moving_avg_inactive_zero, run_all, seasonal_naive, zero)
from src.run_baselines import check_forecast, evaluate
from src.run_cleaning import build_outputs
from tests.conftest import make_panel

H = 13


# ---------------------------------------------------------------- contract on real data (small sample)
@pytest.fixture(scope="module")
def real_forecasts(small_raw):
    cfg = get_config()
    train, _ = run_cleaning(load_phase(small_raw, "Phase 0 - Sales.csv", "Phase 0 - Price.csv"), None, cfg)
    p1 = load_phase(small_raw, "Phase 1 - Sales.csv", "Phase 1 - Price.csv")
    return run_all(train, cfg), p1


def test_every_series_gets_13_forecasts_on_phase1_dates(real_forecasts):
    forecasts, p1 = real_forecasts
    ids, dates = pd.Index(p1["unique_id"].unique()), pd.DatetimeIndex(sorted(p1["ds"].unique()))
    assert len(forecasts) == 7
    for name, fc in forecasts.items():
        assert list(fc.columns) == ["unique_id", "ds", "forecast"], name
        assert fc.groupby("unique_id").size().eq(H).all(), name
        assert fc["forecast"].notna().all(), name
        assert (fc["forecast"] >= 0).all(), name
        assert (np.sort(fc["ds"].unique()) == dates.to_numpy()).all(), name
        check_forecast(fc, ids, dates, H)  # same checks the runner applies


def test_evaluate_runs_without_phase2(small_raw, tmp_path):
    """The baseline stage must not need Phase 2 at all: delete it and everything still runs."""
    build_outputs(get_config(), raw_dir=small_raw, out_dir=tmp_path / "processed", make_report=False)
    (tmp_path / "processed" / "phase2_actuals.parquet").unlink()
    totals, _ = evaluate(get_config(), processed_dir=tmp_path / "processed", out_dir=tmp_path / "out",
                         reports_dir=tmp_path, write_report=True)
    assert totals.set_index("method").loc["zero", "score"] == pytest.approx(2.0)
    assert (tmp_path / "baselines.md").exists()


# ---------------------------------------------------------------- method logic on synthetic data
def test_moving_avg_uses_only_weeks_after_first_sale():
    untrimmed = make_panel({"0_1_1": [0, 0, 0, 0, 4, 2], "0_1_2": [1, 2, 3, 4, 5, 6]})
    fc = moving_avg(untrimmed, H, window=4).groupby("unique_id")["forecast"].first()
    assert fc["0_1_1"] == pytest.approx(3.0)  # mean(4, 2), not mean(0, 0, 4, 2) = 1.5
    assert fc["0_1_2"] == pytest.approx(4.5)  # mean of the last 4 weeks
    trimmed = untrimmed[~((untrimmed.unique_id == "0_1_1") & (untrimmed.y.cumsum() == 0))]
    pd.testing.assert_frame_equal(moving_avg(trimmed, H, 4), moving_avg(untrimmed, H, 4))


def test_forecast_dates_follow_panel_end():
    df = make_panel({"0_1_1": [1, 2, 3]}, start="2023-01-02")
    fc = zero(df, H)
    assert fc["ds"].iloc[0] == pd.Timestamp("2023-01-23") and fc["ds"].nunique() == H
    assert (fc["forecast"] == 0).all()


def test_seasonal_naive_copies_last_year_and_falls_back():
    long_ys = list(range(1, 71))  # 70 weeks: forecast week t = value 52 weeks earlier
    short_ys = [5] * 30           # < 52 weeks of history -> moving_avg(13) = 5
    long_df = make_panel({"0_1_1": long_ys}, start="2023-01-02")
    short_start = long_df["ds"].max() - pd.Timedelta(weeks=29)  # both series end on the same week
    df = pd.concat([long_df, make_panel({"0_1_2": short_ys}, start=short_start)], ignore_index=True)
    fc = seasonal_naive(df, H).set_index("unique_id")["forecast"]
    # horizon weeks are 71..83, so the copies are weeks 19..31, i.e. values 19..31
    assert fc["0_1_1"].tolist() == list(range(19, 32))
    assert (fc["0_1_2"] == 5).all()


def test_inactive_zero_differs_only_with_longer_window():
    df = make_panel({"0_1_1": [10] * 20 + [0] * 13, "0_1_2": [3] * 33})
    a = moving_avg(df, H, 13).groupby("unique_id")["forecast"].first()
    b = moving_avg_inactive_zero(df, H, 13).groupby("unique_id")["forecast"].first()
    assert a["0_1_1"] == 0 and b["0_1_1"] == 0  # identical at window = 13
    c = moving_avg_inactive_zero(df, H, window=26).groupby("unique_id")["forecast"].first()
    assert c["0_1_1"] == 0 and moving_avg(df, H, 26).groupby("unique_id")["forecast"].first()["0_1_1"] > 0
    assert c["0_1_2"] == 3

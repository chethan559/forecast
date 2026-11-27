import numpy as np
import pandas as pd
import pytest

from src.adapters.vn1 import load_phase
from src.cleaning import (add_relative_price, clean_price, classify_series, flag_price_level_shift,
                          flag_smoothed_weeks, flag_suspected_stockouts, run_cleaning, trim_leading_zeros)
from src.config import get_config
from tests.conftest import make_panel

CFG = get_config({"smoothing": {"switch_date": "2023-03-06"}})


@pytest.fixture(scope="module")
def real_small(small_raw):
    return load_phase(small_raw, "Phase 0 - Sales.csv", "Phase 0 - Price.csv")


# ---------------------------------------------------------------- (a) trim
def test_trim_every_series_starts_positive_on_real_data(real_small):
    clean, series = run_cleaning(real_small, None, get_config())
    first = clean.sort_values("ds").groupby("unique_id")["y"].first()
    assert (first > 0).all()
    assert (clean.groupby("unique_id")["ds"].min() == series.set_index("unique_id")["first_sale_date"]).all()


def test_trim_records_leading_weeks_and_keeps_later_zeros():
    df = make_panel({"0_1_1": [0, 0, 0, 5, 0, 3], "0_1_2": [2, 0, 1, 0, 0, 0]})
    out, info = trim_leading_zeros(df)
    assert info.loc["0_1_1", "n_leading_zero_weeks"] == 3
    assert info.loc["0_1_2", "n_leading_zero_weeks"] == 0
    assert out.loc[out.unique_id == "0_1_1", "y"].tolist() == [5, 0, 3]  # inner zero kept
    assert len(out[out.unique_id == "0_1_2"]) == 6


def test_never_sold_series_is_reported_not_lost():
    out, info = trim_leading_zeros(make_panel({"0_1_1": [0, 0, 0], "0_1_2": [1, 1, 1]}))
    assert bool(info.loc["0_1_1", "never_sold"]) and "0_1_1" not in set(out.unique_id)


# ---------------------------------------------------------------- (b) smoothing
def test_identical_runs_flagged_and_values_unchanged():
    ys = [7, 7, 7, 7, 3, 3, 1, 1, 1, 0, 0, 0, 0, 4, 4, 4]
    df = make_panel({"0_1_1": ys})
    out, info = flag_smoothed_weeks(df, CFG)
    assert out["y"].tolist() == ys  # never overwrites
    assert out["identical_run"].tolist() == [True] * 4 + [False] * 2 + [True] * 3 + [False] * 4 + [True] * 3
    # switch forced at 2023-03-06 (week 9): only runs before it count as is_smoothed
    assert out["is_smoothed"].tolist() == [True] * 4 + [False] * 2 + [True] * 3 + [False] * 7


def test_switch_date_estimated_from_repeat_rate():
    rng = np.random.default_rng(1)
    series = {}
    for i in range(40):  # 40 weeks of monthly-constant values, then 40 weeks of random weekly counts
        monthly = np.repeat(rng.integers(5, 50, 10), 4)
        weekly = rng.poisson(20, 40) + 1
        series[f"0_1_{i}"] = list(monthly) + list(weekly)
    df = make_panel(series)
    out, info = flag_smoothed_weeks(df, get_config({"smoothing": {"switch_date": None}}))  # unpinned: estimate
    assert info["switch_date"] == pd.Timestamp("2023-01-02") + pd.Timedelta(weeks=40)


# ---------------------------------------------------------------- (c) price
def test_price_flags_and_zero_price_treated_as_missing():
    df = make_panel({"0_1_1": [0, 2, 0, 3, 1]}, price={"0_1_1": [np.nan, 10.0, np.nan, 0.0, 12.0]})
    out = clean_price(df)
    assert out["price_raw"].isna().tolist() == [True, False, True, False, False]  # raw kept
    assert out["price_zero"].tolist() == [False, False, False, True, False]
    assert out["price_clean"].tolist()[1:] == [10.0, 10.0, 10.0, 12.0]
    assert np.isnan(out["price_clean"].iloc[0])  # no backfill into the past


def test_forward_fill_never_uses_future_values():
    rng = np.random.default_rng(2)
    ys = rng.poisson(1.0, 60).astype(float)
    ys[0] = 1
    price = [float(rng.uniform(5, 15)) if y > 0 else np.nan for y in ys]
    df = make_panel({"0_1_1": list(ys)}, price={"0_1_1": price})
    full = add_relative_price(flag_price_level_shift(clean_price(df), CFG)[0], CFG)
    for cut in [10, 25, 40, 59]:
        part = df[df["ds"] <= df["ds"].iloc[cut]]
        p = add_relative_price(flag_price_level_shift(clean_price(part), CFG)[0], CFG)
        for col in ["price_clean", "relative_price", "price_regime"]:
            np.testing.assert_array_equal(p[col].to_numpy(), full[col].iloc[:cut + 1].to_numpy(),
                                          err_msg=f"{col} changed when future rows were added (cut={cut})")
    # every ffilled value equals the most recent valid price at or before that week
    last_valid = full["price_raw"].where(full["price_raw"] > 0).ffill()
    np.testing.assert_array_equal(full["price_clean"].to_numpy(), last_valid.to_numpy())


def test_price_columns_prefix_invariant_on_real_data(real_small):
    cfg = get_config()
    full, _ = run_cleaning(real_small, None, cfg)
    cut = pd.Timestamp("2022-06-27")
    part, _ = run_cleaning(real_small[real_small["ds"] <= cut], None, cfg)
    a = full[full["ds"] <= cut].set_index(["unique_id", "ds"]).sort_index()
    b = part.set_index(["unique_id", "ds"]).sort_index()
    # price_shift itself is back-dated (descriptive) and is deliberately excluded here
    for col in ["price_clean", "price_baseline", "relative_price", "price_regime", "price_shift_detected"]:
        pd.testing.assert_series_equal(a[col], b[col], check_names=False)


# ---------------------------------------------------------------- (d) + (e) price shift and relative price
def test_price_shift_detected_and_relative_price_resets():
    ys = [5] * 40
    price = [10.0] * 20 + [30.0] * 20  # 3x jump at week 20
    df = make_panel({"0_1_1": ys}, price={"0_1_1": price})
    out, ev = flag_price_level_shift(clean_price(df), CFG)
    assert len(ev) == 1 and ev["shift_date"].iloc[0] == df["ds"].iloc[20]
    assert ev["detected_date"].iloc[0] > ev["shift_date"].iloc[0]  # detection lags, location is back-dated
    det = int(np.flatnonzero(out["price_shift_detected"])[0])
    out = add_relative_price(out, CFG)
    assert np.allclose(out["relative_price"].iloc[1:20], 1.0)
    assert (out["relative_price"].iloc[20:det] > 2.5).all()  # live view: jump not yet confirmed
    assert np.isnan(out["relative_price"].iloc[det])  # window reset: no past price in the new regime
    assert np.allclose(out["relative_price"].iloc[det + 1:], 1.0)  # not 3.0


def test_promo_week_is_not_a_shift_but_shows_in_relative_price():
    price = [10.0] * 15 + [4.0] + [10.0] * 15
    df = make_panel({"0_1_1": [5] * 31}, price={"0_1_1": price})
    out, ev = flag_price_level_shift(clean_price(df), CFG)
    assert ev.empty
    out = add_relative_price(out, CFG)
    assert out["relative_price"].iloc[15] == pytest.approx(0.4)  # current week excluded from its baseline


# ---------------------------------------------------------------- (f) stockouts
def test_stockout_rule():
    busy = [5] * 13 + [0] * 4 + [5] * 3       # busy then 4 zeros -> flagged
    slow = [1] * 13 + [0] * 4 + [1] * 3       # mean 1 < 2 -> not flagged
    short = [5] * 13 + [0] * 3 + [5] * 3      # only 3 zeros -> not flagged
    trailing = [5] * 13 + [0] * 6             # open at the end -> skipped by default
    df = make_panel({"0_1_1": busy, "0_1_2": slow, "0_1_3": short, "0_1_4": trailing})
    out = flag_suspected_stockouts(df, CFG).set_index("unique_id")
    assert out.loc["0_1_1", "is_suspected_stockout"].sum() == 4
    assert out.loc["0_1_1", "y"].tolist() == busy  # flag only
    for uid in ["0_1_2", "0_1_3", "0_1_4"]:
        assert out.loc[uid, "is_suspected_stockout"].sum() == 0
    cfg = get_config({"stockout": {"include_trailing_run": True}})
    assert flag_suspected_stockouts(df, cfg).query("unique_id == '0_1_4'")["is_suspected_stockout"].sum() == 6


# ---------------------------------------------------------------- (g) classification
def test_classification():
    df = make_panel({
        "0_1_1": [10, 11, 9, 10] * 15,                 # smooth
        "0_1_2": [1, 0, 0, 1, 0, 0] * 10,              # intermittent (ADI 3, CV2 0)
        "0_1_3": [1, 0, 0, 40, 0, 0] * 10,             # lumpy
        "0_1_4": [1, 50, 2, 60] * 15,                  # erratic
        "0_1_5": [3] * 20 + [0] * 20,                  # inactive, new (40 weeks), ADI 2
    })
    t = classify_series(df, CFG)
    assert t["demand_type"].to_dict() == {"0_1_1": "smooth", "0_1_2": "intermittent", "0_1_3": "lumpy",
                                          "0_1_4": "erratic", "0_1_5": "intermittent"}
    assert t.loc["0_1_2", "adi"] == pytest.approx(3.0)
    assert t["is_inactive"].tolist() == [False, False, False, False, True]
    assert t["is_new"].tolist() == [False, False, False, False, True]
    assert t.loc["0_1_4", "volume_rank"] == 1


def test_guard_rejects_held_out_dates():
    df = make_panel({"0_1_1": [1, 2, 3]}, start="2023-12-25")  # runs into 2024-01-08 (Phase 2)
    with pytest.raises(ValueError, match="held-out"):
        run_cleaning(df, None, get_config())


# ---------------------------------------------------------------- (h) level shifts
def test_level_shift_series_and_group():
    from src.cleaning import detect_level_shifts, group_shift_dates, level_shift_group_events
    cfg = get_config()
    base = [1.0, 2.0, 1.0, 0.0] * 6                       # 24 weeks around 1/week
    jump = {f"2_9_{i}": base + [500.0 + i] * 8 for i in range(4)}        # 4 of 6 jump at week 24
    flat = {f"2_9_{i}": base + [1.0] * 8 for i in range(4, 6)}
    other = {"3_8_1": [10.0] * 32}
    df = make_panel({**jump, **flat, **other})
    out = detect_level_shifts(df, cfg)
    ev_week = df["ds"].unique()[24]
    shifted = out[out["level_shift"]]
    assert set(shifted["unique_id"]) == set(jump) and (shifted["ds"] == ev_week).all()  # one event each, at the jump
    assert shifted["level_shift_up"].all()
    assert out["y"].tolist() == df.sort_values(["unique_id", "ds"])["y"].tolist()  # flag only
    ev = level_shift_group_events(out, cfg)
    assert len(ev) == 2 and set(ev["group_type"]) == {"warehouse", "client"}  # 4/6 > 50% in warehouse 9 / client 2
    assert (ev["ds"] == ev_week).all() and (ev["direction"] == "up").all()
    dates = group_shift_dates(out, ev, cfg)
    assert set(dates["unique_id"]) == set(jump) | set(flat)  # the whole group, incl. series that didn't cross
    assert "3_8_1" not in set(dates["unique_id"])


def test_level_shift_is_past_only(real_small):
    from src.cleaning import detect_level_shifts
    cfg = get_config()
    full = detect_level_shifts(real_small[["unique_id", "client", "warehouse", "ds", "y"]].copy(), cfg)
    cut = pd.Timestamp("2022-06-27")
    part = detect_level_shifts(real_small.loc[real_small["ds"] <= cut, ["unique_id", "client", "warehouse", "ds", "y"]].copy(), cfg)
    a = full[full["ds"] <= cut].set_index(["unique_id", "ds"]).sort_index()
    b = part.set_index(["unique_id", "ds"]).sort_index()
    for col in ["level_shift", "level_regime", "level_active"]:
        pd.testing.assert_series_equal(a[col], b[col], check_names=False)


def test_warehouse_251_event_detected(raw_dir):
    """The real event that broke fold B: warehouse 251, 2022-11-07."""
    from src.cleaning import detect_level_shifts, level_shift_group_events
    p0 = load_phase(raw_dir, "Phase 0 - Sales.csv", "Phase 0 - Price.csv")
    sub = p0[p0["client"].eq(2)]
    out = detect_level_shifts(trim_leading_zeros(sub)[0], get_config())
    ev = level_shift_group_events(out, get_config())
    hit = ev[(ev["group_type"] == "warehouse") & (ev["group_id"] == 251)]
    assert pd.Timestamp("2022-11-07") in set(hit["ds"])
    assert (hit.loc[hit["ds"] == "2022-11-07", "share"] > 0.5).all()

"""Generalisation stage: automatic folds, sealed test weeks, point metrics, M5 config."""
import pandas as pd
import pytest

from src.folds import auto_folds
from src.metrics import point_metrics
from src.sealed import SealedError, read_sealed, sealed_path, split_sealed, write_sealed
from tests.conftest import ROOT, synthetic_engine_config, synthetic_weekly

WEEK = pd.Timedelta(weeks=1)


def _weekly(start="2020-01-06", n_weeks=200):
    return synthetic_weekly(n_series=5, n_weeks=n_weeks, start=start)


# ----------------------------------------------------------------------------- automatic folds
def test_auto_folds_roll_back_from_the_end(tmp_path):
    cfg = synthetic_engine_config(tmp_path, backtest={"folds": {}, "auto_folds": {"n": 4, "step_weeks": 13}})
    w = _weekly()
    folds, log = auto_folds(w, cfg)
    last = w["ds"].max()
    assert list(folds) == ["F1", "F2", "F3", "F4"]
    assert folds["F4"] == last - 13 * WEEK
    assert all(folds[f"F{i + 1}"] - folds[f"F{i}"] == 13 * WEEK for i in range(1, 4))
    assert sum(x.get("holiday", False) for x in log) >= 1  # four 13-week windows cover a year


def test_holiday_fold_needs_clean_previous_year(tmp_path):
    cfg = synthetic_engine_config(tmp_path, backtest={"folds": {}, "auto_folds": {"n": 4, "step_weeks": 13}})
    w = _weekly()
    clean, _ = auto_folds(w, cfg, switch_date=None)
    # weekly data only from late in the series: last year's holiday window is "smoothed"
    late_switch = w["ds"].max() - 40 * WEEK
    folds, log = auto_folds(w, cfg, switch_date=late_switch)
    skipped = [x for x in log if x.get("holiday") and not x["kept"]]
    assert skipped and all(not x["previous_year_clean"] for x in skipped)
    assert len(folds) == 4 and set(folds.values()) != set(clean.values())
    for T in folds.values():  # no kept fold covers a holiday whose previous year is before the switch
        rec = next(x for x in log if x.get("origin") == T.date())
        assert rec["kept"] and (not rec["holiday"] or rec["previous_year_clean"])


def test_vn1_like_switch_drops_the_2022_holiday_fold(tmp_path):
    # VN1: weekly data from 2022-08-29, so a 2022 holiday fold reads smoothed 2021 values (why fold A was dropped)
    cfg = synthetic_engine_config(tmp_path, backtest={"folds": {}, "auto_folds": {"n": 6, "step_weeks": 13}})
    w = _weekly(start="2020-07-06", n_weeks=170)  # ends 2023-09-25
    folds, log = auto_folds(w, cfg, switch_date="2022-08-29")
    hol = [x for x in log if x.get("holiday")]
    assert any(str(x["origin"]).startswith("2022-1") and not x["kept"] for x in hol)
    assert all(not (T.year == 2022 and T.month >= 9) for T in folds.values())


def test_auto_folds_respect_minimum_history(tmp_path):
    cfg = synthetic_engine_config(tmp_path, backtest={"folds": {}, "auto_folds": {"n": 10, "step_weeks": 13}})
    w = _weekly(n_weeks=110)
    folds, log = auto_folds(w, cfg)
    first = w["ds"].min()
    assert all(T >= first + (52 + 26) * WEEK for T in folds.values())
    assert len(folds) < 10 and "note" in log[-1]


# ----------------------------------------------------------------------------- sealed weeks
def _seal_cfg(tmp_path, w, **seal):
    last = w["ds"].max()
    return synthetic_engine_config(
        tmp_path, max_allowed_date=str((last - 13 * WEEK).date()),
        seal={"weeks": 13, "first_week": str((last - 12 * WEEK).date()), "last_week": str(last.date()), **seal},
        backtest={"folds": {}, "auto_folds": {"n": 1, "step_weeks": 13}},
        experiment={"sample_series": 5, "sample_seed": 0})


def test_split_sealed_takes_the_last_weeks(tmp_path):
    w = _weekly()
    cfg = _seal_cfg(tmp_path, w)
    train, sealed = split_sealed(w, cfg)
    assert sealed["ds"].nunique() == 13 and train["ds"].max() < sealed["ds"].min()
    assert len(train) + len(sealed) == len(w)


def test_split_sealed_checks_pinned_dates(tmp_path):
    w = _weekly()
    cfg = _seal_cfg(tmp_path, w, first_week="2000-01-03")
    with pytest.raises(SealedError, match="first_week"):
        split_sealed(w, cfg)


def test_read_sealed_refuses_before_freeze(tmp_path):
    w = _weekly()
    cfg = _seal_cfg(tmp_path, w)
    write_sealed(split_sealed(w, cfg)[1], cfg)
    with pytest.raises(SealedError, match="frozen"):
        read_sealed(cfg, tmp_path / "no_freeze.json")
    (tmp_path / "freeze.json").write_text("{}")
    assert read_sealed(cfg, tmp_path / "freeze.json")["ds"].nunique() == 13


def test_engine_refuses_sealed_weeks(tmp_path):
    from src import engine
    w = _weekly()
    cfg = _seal_cfg(tmp_path, w)
    with pytest.raises(ValueError, match="max_allowed_date"):
        engine.backtest(w, cfg, folds={"X": "2022-06-27"})


def test_backtest_stage_never_opens_the_sealed_file(tmp_path, monkeypatch):
    from src import experiment_m5 as x
    w = _weekly()
    cfg = _seal_cfg(tmp_path, w)
    train, sealed = split_sealed(w, cfg)
    write_sealed(sealed, cfg)
    proc = cfg["paths"]["processed_dir"]
    proc.mkdir(parents=True, exist_ok=True)
    train.to_parquet(proc / x.MODEL_INPUT, index=False)

    opened, real = [], pd.read_parquet

    def spy(path, *a, **k):
        opened.append(str(path))
        if str(path) == str(sealed_path(cfg)):
            raise AssertionError("backtest opened the sealed test weeks")
        return real(path, *a, **k)

    monkeypatch.setattr(pd, "read_parquet", spy)
    res = x.stage_backtest(cfg)
    assert str(sealed_path(cfg)) not in opened
    assert pd.Timestamp(res["origin"].max()) + 13 * WEEK <= pd.Timestamp(cfg["max_allowed_date"])
    assert set(res["method"]) == {"engine", *x.BASELINES}


def test_m5_config_seal_matches_m5_calendar():
    cal = ROOT / "data" / "raw" / "m5" / "calendar.csv"
    if not cal.exists():
        pytest.skip("M5 files not present")
    from src.config import load_config
    cfg = load_config(ROOT / "configs" / "m5.yaml")
    d = pd.to_datetime(pd.read_csv(cal, usecols=["date"])["date"]).iloc[:1941]  # sales_train_evaluation days
    monday = d - pd.to_timedelta(d.dt.day_of_week, unit="D")
    full = monday.value_counts().loc[lambda s: s == 7].index.sort_values()
    assert pd.Timestamp(cfg["seal"]["last_week"]) == full[-1]
    assert pd.Timestamp(cfg["seal"]["first_week"]) == full[-13]
    assert pd.Timestamp(cfg["max_allowed_date"]) == full[-14]


def test_m5_config_keeps_vn1_model_settings():
    from src.config import get_config, load_config
    m5, vn1 = load_config(ROOT / "configs" / "m5.yaml"), get_config()
    for section in ("features", "lgbm", "intervals", "stockout", "price_shift", "relative_price", "classification"):
        assert m5[section] == vn1[section], section
    lv = {k: v for k, v in m5["level_shift"].items() if k != "group_keys"}
    assert lv == {k: v for k, v in vn1["level_shift"].items() if k != "group_keys"}
    assert m5["backtest"]["step_weeks"] == vn1["backtest"]["step_weeks"]


# ----------------------------------------------------------------------------- metrics
def test_point_metrics():
    a = pd.DataFrame({"unique_id": ["a", "a", "b"], "ds": pd.to_datetime(["2024-01-01", "2024-01-08", "2024-01-01"]),
                      "y": [10.0, 0.0, 10.0]})
    f = a.assign(forecast=[12.0, 1.0, 5.0])[["unique_id", "ds", "forecast"]]
    m = point_metrics(f, a)
    assert m["wape"] == pytest.approx(8 / 20)
    assert m["bias"] == pytest.approx(-2 / 20)
    assert m["score"] == pytest.approx(8 / 20 + 2 / 20)

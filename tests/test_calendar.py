import pandas as pd
import pytest

from src import calendar as cal
from src.config import get_config
from src.features import make_features, model_features
from tests.conftest import synthetic_weekly

M = pd.Timestamp


def test_us_weeks_match_the_original_definition():
    w = cal.event_weeks("us", [2022, 2023])
    assert w["thanksgiving"] == {M("2022-11-21"), M("2023-11-20")}   # 24 Nov 2022, 23 Nov 2023
    assert w["cyber"] == {M("2022-11-28"), M("2023-11-27")}
    assert w["christmas"] == {M("2022-12-19"), M("2023-12-25")}      # 25 Dec 2022 is a Sunday


def test_au_events():
    w = cal.event_weeks("au", [2023])
    assert w["black_friday"] == {M("2023-11-20")}       # Fri 24 Nov 2023
    assert w["cyber_monday"] == {M("2023-11-27")}
    assert w["click_frenzy"] == {M("2023-11-13")}       # Main Event Tue 14 Nov 2023
    assert w["boxing_day"] == {M("2023-12-25")}         # Tue 26 Dec 2023
    assert w["mothers_day"] == {M("2023-05-08")}        # Sun 14 May 2023
    assert w["fathers_day"] == {M("2023-08-28")}        # Sun 3 Sep 2023
    assert w["eofy"] == {M("2023-05-29"), M("2023-06-05"), M("2023-06-12"), M("2023-06-19"), M("2023-06-26")}
    assert all(d.day_of_week == 0 for ws in w.values() for d in ws)


def test_unknown_calendar():
    with pytest.raises(ValueError, match="unknown calendar"):
        cal.event_weeks("xx", [2023])


def test_features_use_the_configured_calendar(tmp_path):
    from tests.conftest import synthetic_engine_config
    cfg = synthetic_engine_config(tmp_path, dataset={"calendar": "au"})
    df = synthetic_weekly(3, 120).rename(columns={"price": "price_raw"})
    X = make_features(df, [pd.Timestamp("2023-10-30")], range(1, 14), cfg)
    assert "tw_black_friday_week" in X and "tw_thanksgiving_week" not in X
    assert X.loc[X["target_date"] == "2023-11-13", "tw_click_frenzy_week"].eq(1).all()
    assert "tw_eofy_week" in model_features(cfg)
    us = model_features(get_config())
    assert us[us.index("tw_month") + 1: us.index("tw_month") + 4] == ["tw_thanksgiving_week", "tw_cyber_week",
                                                                      "tw_christmas_week"]

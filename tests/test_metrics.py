import numpy as np
import pandas as pd
import pytest

from src.metrics import vn1_score


@pytest.fixture
def actual():
    rng = np.random.default_rng(0)
    return pd.DataFrame({"unique_id": np.repeat(["a", "b", "c"], 4),
                         "ds": np.tile(pd.date_range("2024-01-08", periods=4, freq="W-MON"), 3),
                         "y": rng.integers(0, 20, 12).astype(float) + 1})


def fc(actual, values):
    return actual[["unique_id", "ds"]].assign(y_hat=values)


def test_perfect_forecast_is_zero(actual):
    r = vn1_score(fc(actual, actual["y"]), actual)
    assert r["score"] == 0 and r["accuracy"] == 0 and r["bias"] == 0


def test_all_zero_forecast_is_two(actual):
    r = vn1_score(fc(actual, 0.0), actual)
    assert r["score"] == pytest.approx(2.0)
    assert r["accuracy"] == pytest.approx(1.0) and r["bias"] == pytest.approx(1.0)


def test_double_forecast_is_two(actual):
    assert vn1_score(fc(actual, 2 * actual["y"]), actual)["score"] == pytest.approx(2.0)


def test_parts_and_cancelling_errors():
    a = pd.DataFrame({"unique_id": ["x", "x"], "ds": pd.to_datetime(["2024-01-08", "2024-01-15"]), "y": [10.0, 10.0]})
    f = a[["unique_id", "ds"]].assign(y_hat=[12.0, 8.0])  # errors +2 and -2 cancel in the bias term
    r = vn1_score(f, a)
    assert r["accuracy"] == pytest.approx(0.2) and r["bias"] == pytest.approx(0.0)


def test_nan_actuals_ignored(actual):
    a = actual.copy()
    a.loc[0, "y"] = np.nan
    f = fc(actual, actual["y"]).copy()
    f.loc[0, "y_hat"] = 1e6  # would explode the score if the NaN row were used
    r = vn1_score(f, a)
    assert r["score"] == 0 and r["n"] == len(a) - 1


def test_missing_forecast_raises(actual):
    with pytest.raises(ValueError):
        vn1_score(fc(actual, actual["y"]).iloc[1:], actual)

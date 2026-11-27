import numpy as np
import pandas as pd
import pytest

from src import engine
from src.features import model_features
from tests.conftest import synthetic_engine_config, synthetic_weekly

WEEK = pd.Timedelta(weeks=1)


@pytest.fixture(scope="module")
def setup(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("engine")
    cfg = synthetic_engine_config(tmp)
    df = synthetic_weekly(30, 150)
    art = engine.fit(df, cfg, methods=["conformal", "quantile_lgbm"])
    return cfg, df, art


def test_artifact_contents(setup):
    cfg, df, art = setup
    names = {p.name for p in art.iterdir()}
    assert {"meta.json", "model_p50.txt", "model_q_lo.txt", "model_q_hi.txt", "conformal.csv"} <= names
    m = engine.load(art)
    assert m.meta["version"] == engine.VERSION and len(m.meta["source_sha256"]) == 64
    assert m.meta["train_end"] == str(df["ds"].max().date()) and m.features
    assert "store_trend_4_13" in m.features and "store" not in m.features  # ids off by default


@pytest.mark.parametrize("method", ["conformal", "quantile_lgbm"])
def test_predict_contract(setup, method):
    cfg, df, art = setup
    fc = engine.predict(art, df, horizon=6, method=method)
    assert list(fc.columns) == ["unique_id", "ds", "p50", "p10", "p90"]
    assert len(fc) == df["unique_id"].nunique() * 6 and fc.notna().all().all()
    assert (fc["p10"] <= fc["p50"]).all() and (fc["p50"] <= fc["p90"]).all() and (fc["p10"] >= 0).all()
    assert fc["ds"].min() == df["ds"].max() + WEEK


def test_saved_and_in_memory_model_agree(setup):
    cfg, df, art = setup
    weekly = engine.prepare(df, cfg)
    model, _ = engine.train_at(weekly, weekly["ds"].max(), cfg, methods=["conformal", "quantile_lgbm"])
    a = engine.predict(model, df, cfg=cfg)
    b = engine.predict(art, df)
    pd.testing.assert_frame_equal(a, b)


def test_predict_uses_only_data_up_to_origin(setup):
    cfg, df, art = setup
    origin = df["ds"].max() - 10 * WEEK
    alt = df.copy()
    alt.loc[alt["ds"] > origin, "y"] = 999.0
    pd.testing.assert_frame_equal(engine.predict(art, df, origin=origin), engine.predict(art, alt, origin=origin))


def test_backtest_writes_report(setup, tmp_path):
    cfg, df, _ = setup
    res = engine.backtest(df, cfg, report_path=tmp_path / "bt.md", methods=["conformal", "quantile_lgbm"])
    t = res["table"]
    assert set(t["interval_method"]) == {"conformal", "quantile_lgbm"}
    assert t["coverage_80"].between(0, 1).all() and (t["score"] > 0).all()
    assert (tmp_path / "bt.md").read_text().startswith("# Backtest")


def test_backtest_needs_actuals_after_each_fold(setup):
    cfg, df, _ = setup
    with pytest.raises(ValueError, match="need 13 weeks"):
        engine.backtest(df, cfg, folds={"late": str((df["ds"].max() - 5 * WEEK).date())})


def test_no_hierarchy_dataset(tmp_path):
    cfg = synthetic_engine_config(tmp_path, dataset={"hierarchy": [], "group_level": None},
                                  level_shift={"group_keys": []})
    df = synthetic_weekly(12, 130).drop(columns="store")
    assert "all_trend_4_13" in model_features(cfg)
    fc = engine.predict(engine.fit(df, cfg), df, horizon=3)
    assert len(fc) == 36 and fc["p50"].notna().all()


def test_in_stock_column_is_used(tmp_path):
    cfg = synthetic_engine_config(tmp_path)
    df = synthetic_weekly(10, 130, with_in_stock=True)
    assert engine.predict(engine.fit(df, cfg), df, horizon=2)["p50"].notna().all()


def test_conformal_lookup_falls_back():
    from src.engine import _conformal_lookup, _conformal_table
    rng = np.random.default_rng(0)
    resid = pd.DataFrame({"band": ["top 1%"] * 10 + ["rest 90%"] * 400, "h": [1] * 410, "r": rng.normal(size=410)})
    t = _conformal_table(resid, 0.1, 0.9)
    q = _conformal_lookup(t, pd.DataFrame({"band": ["top 1%", "rest 90%", "new band"], "h": [1, 1, 2]}))
    by_h = t[t["level"] == "h"].iloc[0]
    full = t[(t["level"] == "band+h") & (t["band"] == "rest 90%")].iloc[0]
    assert q.loc[0, "q_lo"] == pytest.approx(by_h["q_lo"])   # 10 rows < 200: horizon-level fallback
    assert q.loc[1, "q_lo"] == pytest.approx(full["q_lo"])   # enough rows: own group
    overall = t[t["level"] == "all"].iloc[0]
    assert q.loc[2, "q_hi"] == pytest.approx(overall["q_hi"])  # unseen horizon: overall


def test_max_allowed_date_guard(tmp_path):
    cfg = synthetic_engine_config(tmp_path, max_allowed_date="2022-01-03")
    with pytest.raises(ValueError, match="max_allowed_date"):
        engine.prepare(synthetic_weekly(3, 60), cfg)

"""Final forecast layer: frozen LightGBM, the blend_0.25 baseline, and the candidate combinations.

forecast_candidates(weekly, T, ...) fits the frozen LightGBM once on data <= T and returns every
candidate's forecast for the 13 weeks after T, for every series in `weekly`:
  lgbm       : frozen LightGBM (uplift features, level-shift restart, cap k=3)
  lgbm_w0.5  : 0.5 x lgbm + 0.5 x blend_0.25
  lgbm_w0.7  : 0.7 x lgbm + 0.3 x blend_0.25
  routing    : lgbm, except top-1% volume and lumpy series (segments from data <= T) get the 0.5 blend
  blend_0.25 : the baseline (0.25 x seasonal naive + 0.75 x MA13)
"""
import numpy as np
import pandas as pd

from src.backtest import baseline_forecasts, lgbm_forecasts, training_origins
from src.evaluation import ALLOWED_INPUT, complete_with_zeros, volume_band_at_origin
from src.feature_cache import build_cache, load_origins
from src.models.baselines import blend
from src.models.postprocess import cap_forecasts

WEEK = pd.Timedelta(weeks=1)
CANDIDATES = ["lgbm", "lgbm_w0.5", "lgbm_w0.7", "routing"]


def segments_at_origin(weekly: pd.DataFrame, T, cache, ids) -> pd.DataFrame:
    """Volume band (units in the 52 weeks up to T) and demand type (ADI/CV2 over 52 weeks at T). Data <= T only."""
    band = volume_band_at_origin(weekly, T, ids)
    dt = (load_origins(cache, [T], ["unique_id", "demand_type"]).drop_duplicates("unique_id")
          .set_index("unique_id")["demand_type"].reindex(ids).fillna("no history"))
    return pd.DataFrame({"volume_band": band, "demand_type": dt})


def routed_ids(seg: pd.DataFrame, cfg: dict) -> pd.Index:
    m = pd.Series(False, index=seg.index)
    for col, values in cfg["final"]["routing_segments"].items():
        m |= seg[col].isin(values)
    return seg.index[m]


def forecast_candidates(weekly: pd.DataFrame, T, cfg: dict, work_dir, input_name: str = ALLOWED_INPUT) -> dict:
    T = pd.Timestamp(T)
    h = max(cfg["features"]["horizons"])
    start = cfg["backtest"]["origin_settings"][cfg["lgbm"]["origins_setting"]]["start"]
    origins = training_origins(start, T, cfg)  # every training target <= T
    cache = build_cache(weekly, origins + [T], cfg, work_dir, input_name)
    use_ids = cfg["lgbm"]["use_ids"]
    r = lgbm_forecasts(cache, origins, T, cfg, cfg["features"]["scale_method"], [use_ids], weekly)[use_ids]

    # Every series in the data gets a forecast; series that first sell after T get 0 (complete_with_zeros).
    ids = pd.Index(sorted(weekly["unique_id"].unique()))
    dates = pd.date_range(T + WEEK, T + h * WEEK, freq=WEEK)
    grid = pd.DataFrame({"unique_id": np.repeat(ids.to_numpy(), h), "ds": np.tile(dates, len(ids))})
    lg = complete_with_zeros(r["forecast"], grid)
    lg["forecast"] = lg["forecast"] * cfg["lgbm"]["bias_multiplier"]
    cap_stats = {}
    if cfg["lgbm"]["forecast_cap_k"]:
        lg, cap_stats = cap_forecasts(lg, weekly, T, cfg["lgbm"]["forecast_cap_k"])
    base = complete_with_zeros(baseline_forecasts(weekly, T, cfg)[cfg["final"]["baseline"]], grid)

    seg = segments_at_origin(weekly, T, cache, ids)
    routed = routed_ids(seg, cfg)
    w = cfg["final"]["routing_blend_w"]
    mixed = blend(lg, base, w)
    routing = lg.copy()
    pick = routing["unique_id"].isin(routed).to_numpy()
    routing.loc[pick, "forecast"] = mixed.set_index(["unique_id", "ds"]).loc[
        pd.MultiIndex.from_frame(routing.loc[pick, ["unique_id", "ds"]]), "forecast"].to_numpy()

    out = {"lgbm": lg, "lgbm_w0.5": blend(lg, base, 0.5), "lgbm_w0.7": blend(lg, base, 0.7),
           "routing": routing, "blend_0.25": base}
    for name, f in out.items():
        assert len(f) == len(ids) * h and f["forecast"].notna().all() and (f["forecast"] >= 0).all(), name
    return {"forecasts": out, "segments": seg, "n_routed": len(routed), "cap_stats": cap_stats,
            "train_rows": r["train_rows"], "rows_dropped": r["rows_dropped_for_shift"], "origins": origins,
            "importance": r["importance"], "cache": cache}

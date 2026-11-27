"""Forecasting engine: the same model on any dataset in the standard schema (src/schema.py).

    model_dir = fit(df, cfg)                      # train on all data, save an artifact
    fc = predict(model_dir, df, origin, horizon)  # unique_id, ds, p50, p10, p90
    result = backtest(df, cfg)                    # score folds from config backtest.folds, write a report

The model is the frozen VN1 design: one global LightGBM (tweedie, rows weighted by scale) on
point-in-time features, level-shift restart, cap at k x the series' best recent week.
p50 is that point forecast (VN1-optimised, closer to a mean than a strict median).
p10 / p90 come from either
  * "conformal": residuals of a calibration model, i.e. the same model trained 13 weeks earlier and
    scored on the last 13 weeks, grouped by volume band and horizon, or
  * "quantile_lgbm": two extra LightGBM models with a quantile objective (alpha 0.1 and 0.9).
config intervals.method picks one for predict(); fit() can train both (interval experiment).
"""
import datetime as dt
import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from src.backtest import baseline_forecasts, training_origins
from src.config import PROJECT_ROOT
from src.evaluation import complete_with_zeros, volume_band_at_origin
from src.feature_cache import build_cache, data_fingerprint, load_origins
from src.features import (apply_scaling, base_columns, drop_rows_crossing_shift, group_shift_dates_upto,
                          id_features, make_raw_features, model_features, set_categories)
from src.metrics import vn1_score, vn1_score_by_group
from src.models import lgbm
from src.models.postprocess import cap_forecasts
from src.schema import validate

VERSION = "1.0.0"
WEEK = pd.Timedelta(weeks=1)
INTERVAL_METHODS = ("conformal", "quantile_lgbm")
MIN_GROUP_ROWS = 200  # conformal groups smaller than this fall back to horizon-only, then global


# ----------------------------------------------------------------------------- data
def prepare(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Validate against the schema and return the columns the features use (price -> price_raw)."""
    v = validate(df, cfg)
    limit = cfg.get("max_allowed_date")
    if limit and v["ds"].max() > pd.Timestamp(limit):
        raise ValueError(f"data runs past max_allowed_date={limit}: held-out weeks must not be used")
    v = v.rename(columns={"price": "price_raw"}) if "price" in v.columns else v.assign(price_raw=np.nan)
    return v[base_columns(cfg, v)]


def _training_start(cfg: dict, weekly: pd.DataFrame) -> pd.Timestamp:
    start = cfg["backtest"]["origin_settings"][cfg["lgbm"]["origins_setting"]].get("start")
    if start:
        return pd.Timestamp(start)
    return weekly["ds"].min() + cfg["features"]["season_length"] * WEEK  # one season of history first


def _cache(weekly: pd.DataFrame, origins, cfg: dict, work_dir) -> Path:
    return build_cache(weekly, origins, cfg, work_dir, fingerprint=data_fingerprint(weekly))


def source_hash() -> str:
    h = hashlib.sha256()
    for p in sorted((PROJECT_ROOT / "src").rglob("*.py")):
        h.update(p.relative_to(PROJECT_ROOT).as_posix().encode() + p.read_bytes())
    return h.hexdigest()


# ----------------------------------------------------------------------------- model bundle
@dataclass
class Model:
    """Everything predict() needs. Saved as a directory of LightGBM text files + meta.json (+ conformal.csv)."""
    boosters: dict                       # "p50", optional "q_lo", "q_hi"
    features: list
    categories: dict
    meta: dict
    conformal: pd.DataFrame | None = None
    cfg: dict = field(default_factory=dict)


def _train_boosters(cache, origins, T, cfg, weekly, quantiles: bool) -> tuple[dict, list, dict, int]:
    """Point model (+ optional quantile models) on training origins whose targets are all <= T."""
    train = load_origins(cache, origins)
    if cfg["level_shift"].get("restart_after_shift"):
        train = drop_rows_crossing_shift(train, group_shift_dates_upto(weekly, T, cfg))
    if len(train) and train["target_date"].max() > T:
        raise AssertionError("a training target falls after the training cut-off")
    train = train.drop(columns=[c for c in ["unique_id", "origin", cfg["dataset"].get("item_column")] if c and c in train])
    train = apply_scaling(train, cfg, keep_raw=False)
    cats = set_categories(train, cfg)
    feats = lgbm.select_features(model_features(cfg), cfg["lgbm"]["use_ids"], id_features(cfg))
    boosters = {"p50": lgbm.fit(train, feats, cfg["lgbm"]["params"], weight_by_scale=cfg["lgbm"]["weight_by_scale"])}
    if quantiles:
        iv = cfg["intervals"]
        for key, alpha in [("q_lo", iv["lower"]), ("q_hi", iv["upper"])]:
            params = {**cfg["lgbm"]["params"], "objective": "quantile", "alpha": alpha}
            boosters[key] = lgbm.fit(train, feats, params, weight_by_scale=False)  # every series-week counts once
    return boosters, feats, cats, int(train["target"].notna().sum())


def _frame_at(weekly, T, cfg, cache=None, horizons=None) -> pd.DataFrame:
    """Scaled feature rows at origin T (from the cache if given, else computed now)."""
    hs = horizons or cfg["features"]["horizons"]
    raw = load_origins(cache, [T]) if cache is not None else make_raw_features(weekly[weekly["ds"] <= T], [T], hs, cfg)
    raw = raw[raw["h"].isin(hs)]
    return apply_scaling(raw.reset_index(drop=True), cfg, keep_raw=False)


def _predict_rows(model: Model, frame: pd.DataFrame, weekly: pd.DataFrame, T, cfg, method: str | None) -> pd.DataFrame:
    """unique_id, ds, p50, p10, p90 for the series in `frame` (data <= T only)."""
    set_categories(frame, cfg, model.categories)
    scale = frame["scale"].to_numpy(float)
    out = frame[["unique_id", "target_date", "h"]].rename(columns={"target_date": "ds"}).copy()
    out["p50"] = np.clip(model.boosters["p50"].predict(frame[model.features]), 0, None) * scale * cfg["lgbm"]["bias_multiplier"]
    method = method or cfg["intervals"]["method"]
    if method == "quantile_lgbm":
        if "q_lo" not in model.boosters:
            raise ValueError("model has no quantile boosters: refit with interval method quantile_lgbm")
        out["p10"] = np.clip(model.boosters["q_lo"].predict(frame[model.features]), 0, None) * scale
        out["p90"] = np.clip(model.boosters["q_hi"].predict(frame[model.features]), 0, None) * scale
    elif method == "conformal":
        if model.conformal is None:
            raise ValueError("model has no conformal table: refit with interval method conformal")
        band = volume_band_at_origin(weekly, T, out["unique_id"].unique())
        q = _conformal_lookup(model.conformal, out.assign(band=out["unique_id"].map(band).to_numpy()))
        out["p10"] = out["p50"] + q["q_lo"].to_numpy() * scale
        out["p90"] = out["p50"] + q["q_hi"].to_numpy() * scale
    else:
        raise ValueError(f"unknown interval method {method!r}; use one of {INTERVAL_METHODS}")
    k = cfg["lgbm"].get("forecast_cap_k")
    for col in ("p50", "p10", "p90"):
        if k:
            out[col] = cap_forecasts(out[["unique_id", "ds"]].assign(forecast=out[col]), weekly, T, k)[0]["forecast"].to_numpy()
        out[col] = out[col].clip(lower=0)
    out["p10"] = np.minimum(out["p10"], out["p50"])  # keep p10 <= p50 <= p90
    out["p90"] = np.maximum(out["p90"], out["p50"])
    return out[["unique_id", "ds", "h", "p50", "p10", "p90"]]


# ----------------------------------------------------------------------------- conformal
def _conformal_table(resid: pd.DataFrame, lower: float, upper: float) -> pd.DataFrame:
    """Scaled-residual quantiles by (band, h), with fallbacks by h and overall."""
    rows = []
    for keys, g in [(("band", "h"), resid.groupby(["band", "h"])), (("h",), resid.groupby("h")), ((), [((), resid)])]:
        for k, x in g:
            k = k if isinstance(k, tuple) else (k,)
            rows.append({"level": "+".join(keys) or "all", "band": k[0] if "band" in keys else None,
                         "h": (k[1] if "band" in keys else k[0]) if keys else None,
                         "q_lo": float(x["r"].quantile(lower)), "q_hi": float(x["r"].quantile(upper)), "n": len(x)})
    return pd.DataFrame(rows)


def _conformal_lookup(table: pd.DataFrame, rows: pd.DataFrame) -> pd.DataFrame:
    """Per row: (band, h) quantiles if that group had enough calibration rows, else h-only, else overall."""
    t = table.copy()
    t["h"] = t["h"].astype("Int64")
    full = t[(t["level"] == "band+h") & (t["n"] >= MIN_GROUP_ROWS)].set_index(["band", "h"])
    by_h = t[t["level"] == "h"].set_index("h")
    overall = t[t["level"] == "all"].iloc[0]
    keys = pd.MultiIndex.from_arrays([rows["band"].astype(object), rows["h"].astype("Int64")])
    q = full.reindex(keys)[["q_lo", "q_hi"]].reset_index(drop=True)
    qh = by_h.reindex(rows["h"].astype("Int64"))[["q_lo", "q_hi"]].reset_index(drop=True)
    return q.fillna(qh).fillna({"q_lo": overall["q_lo"], "q_hi": overall["q_hi"]})


def _calibrate(cache, weekly, T, cfg) -> pd.DataFrame:
    """Train the model as it was 13 weeks before T, forecast the 13 weeks up to T, keep scaled residuals."""
    h_max = max(cfg["features"]["horizons"])
    T_cal = T - h_max * WEEK
    origins = training_origins(_training_start(cfg, weekly), T_cal, cfg)
    if not origins:
        start = _training_start(cfg, weekly)
        raise ValueError(
            f"not enough history for conformal intervals at {T.date()}: the calibration model needs a training "
            f"origin between {start.date()} (training start) and {(T_cal - h_max * WEEK).date()}. Use a later "
            "origin / fold, more history, or intervals.method: quantile_lgbm")
    hist = weekly[weekly["ds"] <= T_cal]
    boosters, feats, cats, _ = _train_boosters(cache, origins, T_cal, cfg, hist, quantiles=False)
    m = Model(boosters, feats, cats, {}, None, cfg)
    frame = _frame_at(hist, T_cal, cfg, cache)
    set_categories(frame, cfg, cats)
    p = np.clip(boosters["p50"].predict(frame[feats]), 0, None) * frame["scale"].to_numpy(float) * cfg["lgbm"]["bias_multiplier"]
    pred = frame[["unique_id", "target_date", "h", "scale"]].rename(columns={"target_date": "ds"}).assign(p50=p)
    act = weekly[(weekly["ds"] > T_cal) & (weekly["ds"] <= T)][["unique_id", "ds", "y"]]
    e = pred.merge(act, on=["unique_id", "ds"])
    e["r"] = (e["y"] - e["p50"]) / e["scale"]
    e["band"] = e["unique_id"].map(volume_band_at_origin(weekly, T_cal, e["unique_id"].unique()))
    del m
    return _conformal_table(e, cfg["intervals"]["lower"], cfg["intervals"]["upper"])


# ----------------------------------------------------------------------------- public API
def train_at(weekly: pd.DataFrame, T, cfg: dict, work_dir=None, methods=None) -> tuple[Model, Path]:
    """Train on everything known at T (all targets <= T). Returns (model, feature cache dir).
    `weekly` = prepare()d data; rows after T are never used for training, features or calibration."""
    T = pd.Timestamp(T)
    work_dir = Path(work_dir or cfg["paths"]["processed_dir"])
    methods = list(methods or [cfg["intervals"]["method"]])
    origins = training_origins(_training_start(cfg, weekly), T, cfg)
    if not origins:
        raise ValueError(f"no training origin with {max(cfg['features']['horizons'])} weeks of targets before "
                         f"{T.date()}: training starts {_training_start(cfg, weekly).date()}; need more history")
    h_max = max(cfg["features"]["horizons"])
    cal_origins = training_origins(_training_start(cfg, weekly), T - h_max * WEEK, cfg) if "conformal" in methods else []
    cache = _cache(weekly, set(origins) | set(cal_origins) | {T, T - h_max * WEEK}, cfg, work_dir)
    hist = weekly[weekly["ds"] <= T]
    boosters, feats, cats, n_rows = _train_boosters(cache, origins, T, cfg, hist, quantiles="quantile_lgbm" in methods)
    conformal = _calibrate(cache, weekly, T, cfg) if "conformal" in methods else None
    meta = {"version": VERSION, "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
            "dataset": cfg["dataset"]["name"], "train_end": str(T.date()), "horizons": cfg["features"]["horizons"],
            "n_train_origins": len(origins), "train_rows": n_rows, "interval_methods": methods,
            "default_interval_method": cfg["intervals"]["method"], "source_sha256": source_hash(),
            "config_path": cfg.get("config_path")}
    return Model(boosters, feats, cats, meta, conformal, cfg), cache


def fit(df: pd.DataFrame, cfg: dict, out_dir=None, methods=None) -> Path:
    """Train on all of `df` and save the artifact. Returns the artifact directory."""
    weekly = prepare(df, cfg)
    model, _ = train_at(weekly, weekly["ds"].max(), cfg, methods=methods)
    return save(model, out_dir or Path(cfg["paths"]["artifacts_dir"]) / f"model_{model.meta['train_end']}")


def save(model: Model, out_dir) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for k, b in model.boosters.items():
        b.save_model(str(out_dir / f"model_{k}.txt"))
    if model.conformal is not None:
        model.conformal.to_csv(out_dir / "conformal.csv", index=False)
    meta = {**model.meta, "features": model.features,
            "categories": {k: [str(x) if not isinstance(x, (int, np.integer)) else int(x) for x in v]
                           for k, v in model.categories.items()},
            "config": json.loads(json.dumps(model.cfg, default=str))}
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=1))
    return out_dir


def load(path, cfg: dict | None = None) -> Model:
    """Load an artifact. cfg defaults to the exact config saved inside it at training time."""
    path = Path(path)
    meta = json.loads((path / "meta.json").read_text())
    if cfg is None:
        from src.config import _resolve_paths
        cfg = _resolve_paths(meta["config"])
    boosters = {p.stem.removeprefix("model_"): lgb.Booster(model_file=str(p)) for p in sorted(path.glob("model_*.txt"))}
    conf = pd.read_csv(path / "conformal.csv") if (path / "conformal.csv").exists() else None
    return Model(boosters, meta["features"], meta["categories"], meta, conf, cfg)


def predict(model, df: pd.DataFrame, origin=None, horizon: int | None = None, method: str | None = None,
            cfg: dict | None = None) -> pd.DataFrame:
    """Forecast `horizon` weeks after `origin` (default: last week in df) for every series in df.

    Uses data <= origin only. Series that have not sold by the origin get 0. Returns
    unique_id, ds, p50, p10, p90.
    """
    model = load(model, cfg) if isinstance(model, (str, Path)) else model
    cfg = cfg or model.cfg
    weekly = prepare(df, cfg)
    T = pd.Timestamp(origin) if origin is not None else weekly["ds"].max()
    hs = [h for h in cfg["features"]["horizons"] if h <= (horizon or max(cfg["features"]["horizons"]))]
    hist = weekly[weekly["ds"] <= T]
    frame = _frame_at(hist, T, cfg, horizons=hs)
    rows = _predict_rows(model, frame, hist, T, cfg, method) if len(frame) else pd.DataFrame(
        columns=["unique_id", "ds", "h", "p50", "p10", "p90"])
    ids = pd.Index(sorted(weekly["unique_id"].unique()))
    grid = pd.DataFrame({"unique_id": np.repeat(ids.to_numpy(), len(hs)),
                         "ds": np.tile([T + h * WEEK for h in hs], len(ids))})
    out = grid.merge(rows.drop(columns="h"), on=["unique_id", "ds"], how="left")
    return out.fillna({"p50": 0.0, "p10": 0.0, "p90": 0.0})


# ----------------------------------------------------------------------------- evaluation
def interval_stats(fc: pd.DataFrame, actual: pd.DataFrame) -> dict:
    """Coverage of [p10, p90] over series-weeks, and width relative to actual units."""
    e = fc.merge(actual[["unique_id", "ds", "y"]], on=["unique_id", "ds"])
    inside = (e["y"] >= e["p10"]) & (e["y"] <= e["p90"])
    return {"coverage": float(inside.mean()), "below_p10": float((e["y"] < e["p10"]).mean()),
            "above_p90": float((e["y"] > e["p90"]).mean()),
            "relative_width": float((e["p90"] - e["p10"]).sum() / e["y"].sum()), "rows": len(e)}


def evaluate_fold(weekly, T, cfg, actual=None, work_dir=None, methods=None, baseline: bool = True) -> dict:
    """Train at T, forecast the next 13 weeks, score p50 (+ baseline) and every interval method."""
    T = pd.Timestamp(T)
    h = max(cfg["features"]["horizons"])
    methods = list(methods or [cfg["intervals"]["method"]])
    hist = weekly[weekly["ds"] <= T]
    model, cache = train_at(weekly, T, cfg, work_dir, methods)
    frame = _frame_at(hist, T, cfg, cache)
    if actual is None:
        actual = weekly[(weekly["ds"] > T) & (weekly["ds"] <= T + h * WEEK)][["unique_id", "ds", "y"]]
    keys = actual[["unique_id", "ds"]]
    out = {"origin": T, "methods": {}}
    for m in methods:
        rows = _predict_rows(model, frame.copy(), hist, T, cfg, m)
        fc = keys.merge(rows.drop(columns="h"), on=["unique_id", "ds"], how="left").fillna(0.0)
        out["methods"][m] = {"forecast": fc, **interval_stats(fc, actual)}
    point = out["methods"][methods[0]]["forecast"].rename(columns={"p50": "forecast"})[["unique_id", "ds", "forecast"]]
    out["point"] = point
    out["score"] = vn1_score(point, actual, forecast_col="forecast")
    out["band"] = volume_band_at_origin(weekly, T, actual["unique_id"].unique())
    if baseline:
        b = complete_with_zeros(baseline_forecasts(hist, T, cfg)[cfg["final"]["baseline"]], actual)
        out["baseline_score"] = vn1_score(b, actual, forecast_col="forecast")
    out["model_meta"] = model.meta
    return out


def backtest(df: pd.DataFrame, cfg: dict, folds: dict | None = None, report_path=None, methods=None) -> dict:
    """Score every fold in config backtest.folds (origin -> next 13 weeks must be in df). Writes a report."""
    weekly = prepare(df, cfg)
    folds = {k: pd.Timestamp(v) for k, v in (folds or cfg["backtest"]["folds"]).items()}
    if not folds:
        raise ValueError("no backtest folds: set backtest.folds in the dataset config")
    h = max(cfg["features"]["horizons"])
    last = weekly["ds"].max()
    bad = [k for k, T in folds.items() if T + h * WEEK > last]
    if bad:
        raise ValueError(f"folds {bad} need {h} weeks of actuals after their origin; data ends {last.date()}")
    res = {k: evaluate_fold(weekly, T, cfg, methods=methods) for k, T in folds.items()}
    rows = []
    for k, r in res.items():
        for m, s in r["methods"].items():
            rows.append({"fold": k, "origin": str(r["origin"].date()), "score": r["score"]["score"],
                         "accuracy": r["score"]["accuracy"], "bias": r["score"]["bias"],
                         "baseline_score": r.get("baseline_score", {}).get("score"), "interval_method": m,
                         "coverage_80": s["coverage"], "relative_width": s["relative_width"]})
    table = pd.DataFrame(rows)
    report_path = Path(report_path or Path(cfg["paths"]["reports_dir"]) / "engine_backtest.md")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"# Backtest: {cfg['dataset']['name']}", "",
             f"Engine {VERSION}. Each fold trains on targets <= its origin and forecasts the next {h} weeks. "
             "`score` = VN1 score of p50; `coverage_80` = share of series-weeks inside [p10, p90].", "",
             "| " + " | ".join(table.columns) + " |", "|" + "---|" * table.shape[1]]
    for r in table.itertuples(index=False):
        lines.append("| " + " | ".join(f"{v:.4f}" if isinstance(v, float) else str(v) for v in r) + " |")
    report_path.write_text("\n".join(lines) + "\n")
    return {"table": table, "folds": res, "report": report_path}

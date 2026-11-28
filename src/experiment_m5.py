"""Generalisation test on M5 with the VN1 defaults:   python -m src.experiment_m5 --stage <stage>

prepare   adapter (all 30,490 item x store series) -> seal the last 13 weeks -> cleaning on the training
          weeks -> reports/m5/data_quality.md; write the fixed random sample used for modelling
backtest  automatic rolling folds before the sealed weeks: engine vs moving_avg_13, seasonal_naive,
          blend_0.25 -> reports/m5/backtest.md, outputs/m5/backtest/
freeze    snapshot of config + source, best baseline from the backtest -> reports/m5_config.md
test      ONCE: train on every week before the seal, forecast the 13 sealed weeks, then read them and
          score the engine and the best baseline -> reports/m5/test.md, outputs/m5/test/

The sealed weeks are only read in `test` (src/sealed.py refuses before the freeze). The model and features
are never changed in this experiment.
"""
import argparse
import gc
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from src import engine
from src.backtest import baseline_forecasts
from src.cleaning import run_cleaning
from src.config import PROJECT_ROOT, load_config
from src.evaluation import complete_with_zeros, volume_band_at_origin
from src.folds import auto_folds
from src.metrics import point_metrics
from src.run_phase1 import snapshot
from src.sealed import read_sealed, split_sealed, write_sealed

CONFIG = PROJECT_ROOT / "configs" / "m5.yaml"
BASELINES = ["moving_avg_13", "seasonal_naive", "blend_0.25"]
BANDS = ["top 1%", "top 1-10%", "rest 90%", "first sale after origin"]
WEEK = pd.Timedelta(weeks=1)
MODEL_INPUT = "model_input.parquet"


def paths(cfg: dict) -> dict:
    out = Path(cfg["paths"]["outputs_dir"])
    return {"processed": Path(cfg["paths"]["processed_dir"]), "reports": Path(cfg["paths"]["reports_dir"]),
            "backtest": out / "backtest", "test": out / "test", "freeze": out / "test" / "frozen_config.json"}


def table(df: pd.DataFrame, fmt: dict | None = None) -> str:
    fmt = fmt or {}
    cols = list(df.columns)
    out = ["| " + " | ".join(map(str, cols)) + " |", "|" + "---|" * len(cols)]
    for r in df.itertuples(index=False):
        cells = []
        for c, v in zip(cols, r):
            if isinstance(v, float) and np.isnan(v):
                cells.append("–")
            elif c in fmt:
                cells.append(fmt[c].format(v))
            elif isinstance(v, float):
                cells.append(f"{v:.4f}")
            else:
                cells.append(str(v))
        out.append("| " + " | ".join(cells) + " |")
    return "\n".join(out)


# ----------------------------------------------------------------------------- 1. prepare
def model_columns(cfg: dict) -> list[str]:
    d = cfg["dataset"]
    return ["unique_id", *d["hierarchy"], *([d["item_column"]] if d.get("item_column") else []), "ds", "y", "price"]


def stage_prepare(cfg: dict) -> dict:
    from src.adapters import m5
    from src.schema import validate
    p = paths(cfg)
    p["processed"].mkdir(parents=True, exist_ok=True)
    p["reports"].mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    full = validate(m5.load(cfg), cfg, name="m5 adapter output")
    train, sealed = split_sealed(full, cfg)       # dates only; values of the sealed weeks are not looked at
    write_sealed(sealed, cfg)
    n_ids = full["unique_id"].nunique()
    del full, sealed
    gc.collect()

    weekly, series, diag = run_cleaning(train, None, cfg, return_diagnostics=True)  # refuses > max_allowed_date
    weekly.to_parquet(p["processed"] / "clean.parquet", index=False)
    series.to_parquet(p["processed"] / "series_table.parquet", index=False)

    e = cfg["experiment"]
    ids = np.array(sorted(train["unique_id"].unique()))
    sample = np.sort(np.random.default_rng(e["sample_seed"]).choice(ids, size=e["sample_series"], replace=False))
    cols = [c for c in model_columns(cfg) if c != "price"] + ["price_raw"]
    mi = weekly.loc[weekly["unique_id"].isin(sample), cols].rename(columns={"price_raw": "price"})
    mi.to_parquet(p["processed"] / MODEL_INPUT, index=False)
    pd.Series(sample, name="unique_id").to_frame().to_parquet(p["processed"] / "sample_ids.parquet", index=False)

    info = data_quality_report(cfg, train, weekly, series, diag, sample, n_ids)
    info["seconds"] = round(time.time() - t0)
    print(json.dumps(info, indent=2, default=str))
    return info


def data_quality_report(cfg, train, weekly, series, diag, sample, n_ids) -> dict:
    p = paths(cfg)
    sw = diag["switch"]
    sold = series[~series["never_sold"]]
    demand = (series.assign(units=series["total_units"].fillna(0)).groupby("demand_type")
              .agg(series=("unique_id", "size"), units=("units", "sum")))
    demand["series %"] = demand["series"] / demand["series"].sum()
    demand["units %"] = demand["units"] / demand["units"].sum()
    in_sample = series["unique_id"].isin(sample)
    demand["sample series %"] = series[in_sample]["demand_type"].value_counts(normalize=True).reindex(demand.index)
    order = ["smooth", "erratic", "intermittent", "lumpy", "no_sales"]
    demand = demand.reindex([o for o in order if o in demand.index])

    ls_series = int(series["n_level_shifts"].gt(0).sum())
    ls_events = int(weekly["level_shift"].sum())
    gev = diag["level_shift_events"].copy()
    smoothed_weeks = int(weekly["is_smoothed"].sum())
    stockout = int(weekly["is_suspected_stockout"].sum())
    per_store = weekly.groupby("store_id")["unique_id"].nunique()

    info = {"series": n_ids, "series_sold_in_training": int(len(sold)),
            "training_weeks": f"{train['ds'].min().date()} .. {train['ds'].max().date()} ({train['ds'].nunique()})",
            "sealed_weeks": f"{cfg['seal']['first_week']} .. {cfg['seal']['last_week']} ({cfg['seal']['weeks']})",
            "estimated_switch_date": str(sw.get("estimated_switch_date")),
            "repeat_rate_before": sw.get("rate_before"), "repeat_rate_after": sw.get("rate_after"),
            "smoothed_weeks": smoothed_weeks, "series_level_shift_events": ls_events,
            "series_with_level_shift": ls_series, "group_level_shift_events": len(gev),
            "suspected_stockout_weeks": stockout, "series_weeks": len(weekly),
            "price_shift_series": int(series["has_price_shift"].sum()),
            "demand_types": demand["series"].to_dict(), "sample_series": len(sample)}

    sd = sw.get("estimated_switch_date")
    rate = diag.get("repeat_rate")
    rate_txt = ""
    if rate is not None and len(rate):
        yearly = rate.groupby(rate.index.year).mean()
        rate_txt = ", ".join(f"{y}: {v:.1%}" for y, v in yearly.items())
    lines = [
        "# M5 data quality (training weeks only)",
        "",
        f"`python -m src.experiment_m5 --stage prepare`. Adapter: daily units summed into complete Monday weeks, "
        "series = item × store, price = day-weighted average of the Walmart-week prices (M5 calendar). "
        f"Cleaning: `configs/m5.yaml` = the VN1 thresholds. The last {cfg['seal']['weeks']} weeks "
        f"({info['sealed_weeks']}) are sealed in `{p['processed'].relative_to(PROJECT_ROOT)}/sealed/` and were not "
        "cleaned or looked at.",
        "",
        "## Size",
        "",
        "| item | value |", "|---|---|",
        f"| series (item × store) | {n_ids:,} |",
        f"| sold at least once in training | {len(sold):,} |",
        f"| stores / states / departments / items | {weekly['store_id'].nunique()} / {weekly['state_id'].nunique()} / "
        f"{weekly['dept_id'].nunique()} / {weekly['item_id'].nunique():,} |",
        f"| training weeks | {info['training_weeks']} |",
        f"| series-weeks after first sale | {len(weekly):,} |",
        f"| units in training | {weekly['y'].sum():,.0f} |",
        f"| zero weeks (after first sale) | {weekly['y'].eq(0).mean():.1%} |",
        f"| series per store | {per_store.min():,} – {per_store.max():,} |",
        "",
        "## Demand types (Syntetos–Boylan on the whole trimmed history, VN1 thresholds ADI 1.32 / CV² 0.49)",
        "",
        table(demand.reset_index(), {"series": "{:,}", "units": "{:,.0f}", "series %": "{:.1%}", "units %": "{:.1%}",
                                     "sample series %": "{:.1%}"}),
        "",
        "## Smoothed periods (monthly totals spread over weeks)",
        "",
        f"The VN1 detector (changepoint on the weekly repeat rate: share of positive weeks equal to the previous week) "
        f"puts the switch at **{sd.date() if sd is not None else 'none'}**, repeat rate "
        f"{sw.get('rate_before', float('nan')):.1%} before vs {sw.get('rate_after', float('nan')):.1%} after "
        "(VN1: 20.0% vs 10.1%). By year: " + rate_txt + ".",
        f"Weeks flagged `is_smoothed` (identical run ≥ 3 before that date): {smoothed_weeks:,} "
        f"({smoothed_weeks / len(weekly):.2%} of series-weeks).",
        "",
        "## Level shifts (VN1 rule: 4-week mean > 10× or < 1/10 of the previous 13 weeks, level ≥ 5)",
        "",
        f"- series events: {ls_events:,} in {ls_series:,} series ({ls_series / max(len(sold), 1):.1%} of selling series)",
        f"- group events (> 50% of a group's active series in the same week, groups: "
        f"{', '.join(cfg['level_shift']['group_keys'])}): {len(gev)}",
    ]
    if len(gev):
        g = gev.assign(ds=gev["ds"].dt.date.astype(str))
        lines += ["", table(g.head(30), {"share": "{:.0%}"})]
    lines += [
        "",
        "## Other flags",
        "",
        f"- suspected stockout weeks (zero run 4–13 weeks after ≥ 2 units/week): {stockout:,} "
        f"({stockout / len(weekly):.2%} of series-weeks)",
        f"- series with a price level shift: {info['price_shift_series']:,}",
        "",
        "## Modelling sample",
        "",
        f"Modelling uses a fixed random sample of {len(sample):,} series (seed {cfg['experiment']['sample_seed']}), "
        "because all 30,490 series under the VN1 training-origin rule would need ~33M training rows and do not "
        "fit in 8 GB; the sample gives about VN1's training size. The demand-type mix of the sample is in the "
        "table above.",
        "",
    ]
    (p["reports"] / "data_quality.md").write_text("\n".join(lines))
    return info


def load_model_input(cfg: dict) -> pd.DataFrame:
    return pd.read_parquet(paths(cfg)["processed"] / MODEL_INPUT)


# ----------------------------------------------------------------------------- scoring helpers
def score_rows(fc: pd.DataFrame, actual: pd.DataFrame, band: pd.Series, coverage: pd.DataFrame | None) -> list[dict]:
    """Metrics overall and by volume band. fc: unique_id, ds, forecast (+ p10, p90 if coverage)."""
    rows = []
    for b in ["all"] + BANDS:
        ids = actual["unique_id"].unique() if b == "all" else band.index[band == b]
        a = actual[actual["unique_id"].isin(ids)]
        if not len(a):
            continue
        f = fc[fc["unique_id"].isin(ids)]
        r = {"band": b, "series": a["unique_id"].nunique(), "units": float(a["y"].sum())}
        if r["units"] > 0:
            r.update(point_metrics(f, a))
        else:
            r.update({"score": np.nan, "wape": np.nan, "bias": np.nan})
        if coverage is not None:
            s = engine.interval_stats(coverage[coverage["unique_id"].isin(ids)], a)
            r.update({"coverage_80": s["coverage"], "relative_width": s["relative_width"] if r["units"] > 0 else np.nan})
        rows.append(r)
    return rows


def baseline_set(hist: pd.DataFrame, T, cfg, actual) -> dict:
    b = baseline_forecasts(hist, T, cfg)
    return {m: complete_with_zeros(b[m], actual) for m in BASELINES}


# ----------------------------------------------------------------------------- 3. backtest
def stage_backtest(cfg: dict) -> pd.DataFrame:
    p = paths(cfg)
    p["backtest"].mkdir(parents=True, exist_ok=True)
    p["reports"].mkdir(parents=True, exist_ok=True)
    weekly = engine.prepare(load_model_input(cfg), cfg)   # refuses anything after max_allowed_date
    folds, log = auto_folds(weekly, cfg, cfg["smoothing"].get("switch_date"))
    print("folds:", {k: str(v.date()) for k, v in folds.items()}, flush=True)
    rows, meta = [], {}
    for name, T in folds.items():
        t0 = time.time()
        h = max(cfg["features"]["horizons"])
        r = engine.evaluate_fold(weekly, T, cfg, work_dir=p["processed"], baseline=False)
        hist = weekly[weekly["ds"] <= T]
        actual = weekly[(weekly["ds"] > T) & (weekly["ds"] <= T + h * WEEK)][["unique_id", "ds", "y"]]
        iv = r["methods"][cfg["intervals"]["method"]]["forecast"]
        fcs = {"engine": iv.rename(columns={"p50": "forecast"})[["unique_id", "ds", "forecast"]],
               **baseline_set(hist, T, cfg, actual)}
        for m, fc in fcs.items():
            for row in score_rows(fc, actual, r["band"], iv if m == "engine" else None):
                rows.append({"fold": name, "origin": str(T.date()), "method": m, **row})
            fc.to_parquet(p["backtest"] / f"{m}__{name}.parquet", index=False)
        iv.to_parquet(p["backtest"] / f"engine_intervals__{name}.parquet", index=False)
        meta[name] = {"origin": str(T.date()), "train_rows": r["model_meta"]["train_rows"],
                      "n_train_origins": r["model_meta"]["n_train_origins"], "seconds": round(time.time() - t0)}
        print(f"{name} {T.date()}: " + ", ".join(
            f"{x['method']} {x['score']:.4f}" for x in rows if x["fold"] == name and x["band"] == "all")
              + f" ({meta[name]['seconds']} s, {meta[name]['train_rows']:,} rows)", flush=True)
        del r, fcs, iv
        gc.collect()
    res = pd.DataFrame(rows)
    res.to_csv(p["backtest"] / "scores.csv", index=False)
    (p["backtest"] / "folds.json").write_text(json.dumps({"folds": meta, "log": log}, indent=2, default=str))
    write_backtest_report(cfg, res, meta, log)
    return res


def summary_tables(res: pd.DataFrame) -> dict:
    allb = res[res["band"] == "all"]
    by_fold = allb.pivot_table(index=["fold", "origin"], columns="method", values="score").reset_index()
    mean = (allb.groupby("method")[["score", "wape", "bias", "coverage_80", "relative_width"]].mean()
            .reindex(["engine", *BASELINES]).reset_index())
    wins = (allb.pivot_table(index="fold", columns="method", values="score")
            .pipe(lambda t: pd.Series({m: int((t["engine"] < t[m]).sum()) for m in BASELINES})))
    band = (res.groupby(["band", "method"])[["score", "wape", "bias", "coverage_80", "units"]].mean()
            .reset_index())
    band["band"] = pd.Categorical(band["band"], ["all"] + BANDS, ordered=True)
    band["method"] = pd.Categorical(band["method"], ["engine", *BASELINES], ordered=True)
    return {"by_fold": by_fold, "mean": mean, "wins": wins, "band": band.sort_values(["band", "method"])}


FMT = {"score": "{:.4f}", "wape": "{:.4f}", "bias": "{:+.1%}", "coverage_80": "{:.1%}", "relative_width": "{:.2f}",
       "units": "{:,.0f}", "series": "{:,}"}


def write_backtest_report(cfg, res, meta, log):
    p = paths(cfg)
    s = summary_tables(res)
    detail = res[["fold", "method", "band", "series", "units", "score", "wape", "bias", "coverage_80", "relative_width"]]
    folds = pd.DataFrame([{"fold": k, **v} for k, v in meta.items()])
    lines = [
        "# M5 backtest (VN1 default engine, untouched)",
        "",
        f"`python -m src.experiment_m5 --stage backtest`. {cfg['experiment']['sample_series']:,} sampled item × store "
        f"series, training weeks only (≤ {cfg['max_allowed_date']}). Automatic rolling folds (`src/folds.py`): "
        "origins every 13 weeks back from the last training week; a holiday fold is kept only if the same window a "
        "year earlier is clean weekly data. Each fold trains on targets ≤ its origin and forecasts the next 13 weeks.",
        "",
        "Metrics: `score` = VN1 score (WAPE + |bias|), `wape` = Σ|F−A|/ΣA, `bias` = Σ(F−A)/ΣA (> 0 = over-forecast), "
        "`coverage_80` = share of series-weeks inside the engine's [p10, p90] (conformal; baselines have no interval).",
        "",
        "## Fold selection",
        "",
        table(pd.DataFrame([x for x in log if "origin" in x])),
        "",
        table(folds, {"train_rows": "{:,}"}),
        "",
        "## VN1 score by fold",
        "",
        table(s["by_fold"]),
        "",
        "## Mean over folds",
        "",
        table(s["mean"], FMT),
        "",
        "Folds the engine beats each baseline on: " + ", ".join(f"{m} {v}/{len(meta)}" for m, v in s["wins"].items()) + ".",
        "",
        "## By volume band (mean over folds; band at each origin by units in the previous 52 weeks)",
        "",
        table(s["band"], FMT),
        "",
        "## Every fold × method × band",
        "",
        table(detail, FMT),
        "",
    ]
    (p["reports"] / "backtest.md").write_text("\n".join(lines))


# ----------------------------------------------------------------------------- 4. freeze
def flat(d, prefix=""):
    out = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(flat(v, key + "."))
        else:
            out[key] = v
    return out


def stage_freeze(cfg: dict) -> Path:
    p = paths(cfg)
    if p["freeze"].exists():
        raise SystemExit(f"{p['freeze']} exists: already frozen.")
    res = pd.read_csv(p["backtest"] / "scores.csv")
    mean = res[res["band"] == "all"].groupby("method")["score"].mean()
    best = mean.reindex(BASELINES).idxmin()
    p["test"].mkdir(parents=True, exist_ok=True)
    snap = snapshot(cfg)
    snap["best_baseline"] = best
    snap["backtest_mean_score"] = mean.to_dict()
    p["freeze"].write_text(json.dumps(snap, indent=1))
    write_config_report(cfg, best, mean)
    return p["freeze"]


DATA_KEYS = ("dataset.", "paths.", "max_allowed_date", "seal", "experiment.", "config_path", "splits.", "final.routing",
             "final.selection", "final.min_fold_wins", "lgbm_smoke", "backtest.folds", "backtest.holiday_folds",
             "backtest.lgbm_grid", "backtest.output_dir", "baselines.output_dir", "backtest.origin_settings.weekly_only")


def write_config_report(cfg, best, mean):
    from src.config import get_config
    vn1, m5 = flat(json.loads(json.dumps(get_config(), default=str))), flat(json.loads(json.dumps(cfg, default=str)))
    diff = []
    for k in sorted(set(vn1) | set(m5)):
        if vn1.get(k) != m5.get(k):
            kind = "data / bookkeeping" if k.startswith(DATA_KEYS) else "setting"
            diff.append({"key": k, "VN1": vn1.get(k, "–"), "M5": m5.get(k, "–"), "kind": kind})
    diff = pd.DataFrame(diff)
    settings = diff[diff["kind"] == "setting"]
    folds = json.loads((paths(cfg)["backtest"] / "folds.json").read_text())["folds"]
    model_keys = ["features.scale_method", "features.optional_features", "features.lags", "features.roll_windows",
                  "lgbm.params", "lgbm.weight_by_scale", "lgbm.use_ids", "lgbm.forecast_cap_k", "lgbm.bias_multiplier",
                  "level_shift.restart_after_shift", "intervals.method", "backtest.step_weeks"]
    lines = [
        "# M5 frozen config (no M5-specific tuning)",
        "",
        f"Frozen by `python -m src.experiment_m5 --stage freeze` → `outputs/m5/test/frozen_config.json` (config + "
        "sha256 of every `src/*.py`). The test stage refuses to run if either changes, and runs once.",
        "",
        "## Model (identical to the frozen VN1 model)",
        "",
        "| setting | value |", "|---|---|",
        *[f"| `{k}` | `{m5.get(k)}` |" for k in model_keys],
        "",
        "Global direct multi-horizon LightGBM (tweedie, rows weighted by scale, 100 rounds, default leaves / rate), "
        "point-in-time features recomputed at every origin, training origins every 2 weeks from data start + 52 "
        "weeks, rows crossing a level shift dropped, forecasts capped at 3× the series' best week of the last 52, "
        "conformal 80% intervals by volume band × horizon.",
        "",
        "## Every difference from configs/vn1.yaml",
        "",
        f"Settings (thresholds, features, model) that differ: **{len(settings)}**"
        + (" – listed below." if len(settings) else ". Everything below is data description or bookkeeping."),
        "",
        table(diff.assign(VN1=diff["VN1"].astype(str).str.slice(0, 60), M5=diff["M5"].astype(str).str.slice(0, 60))),
        "",
        "Notes:",
        "- `smoothing.switch_date`: VN1 pinned the date its detector found; M5 follows the same procedure (estimated "
        "in the prepare step, then pinned). It is data, not a tuned value.",
        "- `level_shift.group_keys`: the generic default (every hierarchy level), as VN1 (which lists both of its levels).",
        "- `price_rules.ignore`: the VN1 rule is about one VN1 client; M5 has none.",
        "- `backtest.origin_settings.full.start`: null = data start + 52 weeks, the same rule that gave VN1's date.",
        "",
        "## Test setup",
        "",
        f"- sealed weeks: {cfg['seal']['first_week']} .. {cfg['seal']['last_week']}; train on every week ≤ "
        f"{cfg['max_allowed_date']}, forecast origin {cfg['max_allowed_date']}",
        f"- series: the same {cfg['experiment']['sample_series']:,}-series sample (seed {cfg['experiment']['sample_seed']})",
        f"- backtest folds: " + ", ".join(f"{k} {v['origin']}" for k, v in folds.items()),
        f"- best baseline (lowest mean backtest VN1 score): **{best}** ("
        + ", ".join(f"{m} {mean[m]:.4f}" for m in BASELINES) + f"; engine {mean['engine']:.4f})",
        "",
    ]
    (PROJECT_ROOT / "reports" / "m5_config.md").write_text("\n".join(lines))


def check_frozen(cfg: dict) -> dict:
    p = paths(cfg)
    if not p["freeze"].exists():
        raise SystemExit("Not frozen. Run `python -m src.experiment_m5 --stage freeze` first.")
    frozen, now = json.loads(p["freeze"].read_text()), snapshot(cfg)
    if frozen["config"] != now["config"]:
        raise SystemExit("Config differs from the frozen one.")
    changed = sorted(k for k in set(frozen["source_sha256"]) | set(now["source_sha256"])
                     if frozen["source_sha256"].get(k) != now["source_sha256"].get(k))
    if changed:
        raise SystemExit(f"Source changed since the freeze: {changed}")
    return frozen


# ----------------------------------------------------------------------------- 5. test (once)
def stage_test(cfg: dict) -> dict:
    p = paths(cfg)
    frozen = check_frozen(cfg)
    if (p["test"] / "scores.csv").exists():
        raise SystemExit("The M5 test was already scored. It runs once.")
    t0 = time.time()
    df = load_model_input(cfg)
    weekly = engine.prepare(df, cfg)
    T = pd.Timestamp(cfg["max_allowed_date"])
    assert weekly["ds"].max() == T, weekly["ds"].max()
    h = max(cfg["features"]["horizons"])
    model, cache = engine.train_at(weekly, T, cfg, p["processed"])
    sample = pd.read_parquet(p["processed"] / "sample_ids.parquet")["unique_id"]
    grid = pd.DataFrame({"unique_id": np.repeat(sample.to_numpy(), h),
                         "ds": np.tile([T + k * WEEK for k in range(1, h + 1)], len(sample))})
    frame = engine._frame_at(weekly, T, cfg, cache)
    rows = engine._predict_rows(model, frame, weekly, T, cfg, None)
    iv = grid.merge(rows.drop(columns="h"), on=["unique_id", "ds"], how="left").fillna(0.0)
    best = frozen["best_baseline"]
    fcs = {"engine": iv.rename(columns={"p50": "forecast"})[["unique_id", "ds", "forecast"]],
           best: baseline_set(weekly, T, cfg, grid.assign(y=np.nan))[best]}
    for m, fc in fcs.items():  # on disk before any sealed value is read
        fc.to_parquet(p["test"] / f"{m}.parquet", index=False)
    iv.to_parquet(p["test"] / "engine_intervals.parquet", index=False)

    actual = read_sealed(cfg, p["freeze"])
    actual = actual[actual["unique_id"].isin(sample)][["unique_id", "ds", "y"]]
    assert set(actual["ds"]) == set(grid["ds"]) and len(actual) == len(grid), "sealed weeks != forecast weeks"
    band = volume_band_at_origin(weekly, T, sample)
    res = []
    for m, fc in fcs.items():
        for row in score_rows(fc, actual, band, iv if m == "engine" else None):
            res.append({"method": m, **row})
    res = pd.DataFrame(res)
    res.to_csv(p["test"] / "scores.csv", index=False)
    weekly_tot = actual.groupby("ds")["y"].sum().rename("actual").to_frame()
    for m, fc in fcs.items():
        weekly_tot[m] = fc.groupby("ds")["forecast"].sum()
    meta = {"train_rows": model.meta["train_rows"], "n_train_origins": model.meta["n_train_origins"],
            "seconds": round(time.time() - t0), "best_baseline": best}
    (p["test"] / "meta.json").write_text(json.dumps(meta, indent=2))
    write_test_report(cfg, res, weekly_tot, meta)
    print(res[res["band"] == "all"].to_string(index=False))
    return meta


def write_test_report(cfg, res, wk, meta):
    p = paths(cfg)
    w = wk.reset_index(names="week")
    w["week"] = w["week"].dt.date.astype(str)
    lines = [
        "# M5 sealed test (run once)",
        "",
        f"Frozen config (`reports/m5_config.md`), trained on every week ≤ {cfg['max_allowed_date']} "
        f"({meta['n_train_origins']} origins, {meta['train_rows']:,} rows), forecast the sealed weeks "
        f"{cfg['seal']['first_week']} .. {cfg['seal']['last_week']}. Best baseline from the backtest: "
        f"`{meta['best_baseline']}`. {cfg['experiment']['sample_series']:,} sampled series.",
        "",
        "## Total and by volume band (at the forecast origin)",
        "",
        table(res[["method", "band", "series", "units", "score", "wape", "bias", "coverage_80", "relative_width"]], FMT),
        "",
        "## Weekly totals",
        "",
        table(w, {c: "{:,.0f}" for c in w.columns if c != "week"}),
        "",
    ]
    (p["reports"] / "test.md").write_text("\n".join(lines))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", required=True, choices=["prepare", "backtest", "freeze", "test"])
    ap.add_argument("--config", default=str(CONFIG))
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    {"prepare": stage_prepare, "backtest": stage_backtest, "freeze": stage_freeze, "test": stage_test}[args.stage](cfg)


if __name__ == "__main__":
    main()

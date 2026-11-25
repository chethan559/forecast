"""Phase 1: run ONCE with the frozen config.   python -m src.freeze   then   python -m src.run_phase1

Guards:
  * refuses to run unless outputs/phase1/frozen_config.json exists and matches the current config and
    the current source code (sha256 of every src/*.py), so nothing can change after the freeze;
  * refuses to run twice (outputs/phase1/scores.json) unless --force;
  * trains and forecasts from phase0_clean.parquet only, writes the forecasts to disk, and only then
    reads phase1_actuals.parquet to score them. Phase 2 is never read.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.backtest import _table, baseline_forecasts, lgbm_forecasts, training_origins
from src.config import PROJECT_ROOT, get_config
from src.evaluation import complete_with_zeros, load_phase0, volume_band_at_origin
from src.feature_cache import build_cache, load_origins
from src.features import us_holiday_weeks
from src.metrics import vn1_score, vn1_score_by_group
from src.models.postprocess import cap_forecasts

WEEK = pd.Timedelta(weeks=1)
OUT = PROJECT_ROOT / "outputs" / "phase1"
FREEZE_FILE = OUT / "frozen_config.json"
BASELINE = "blend_0.25"


# ----------------------------------------------------------------------------- freeze
def snapshot(cfg: dict) -> dict:
    src = sorted((PROJECT_ROOT / "src").rglob("*.py"))
    return {"config": json.loads(json.dumps(cfg, default=str)),
            "source_sha256": {str(p.relative_to(PROJECT_ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in src}}


def freeze(cfg: dict) -> Path:
    OUT.mkdir(parents=True, exist_ok=True)
    if FREEZE_FILE.exists():
        raise SystemExit(f"{FREEZE_FILE} already exists: the config is frozen. Delete it only if Phase 1 has not run.")
    FREEZE_FILE.write_text(json.dumps(snapshot(cfg), indent=1))
    return FREEZE_FILE


def check_frozen(cfg: dict) -> None:
    if not FREEZE_FILE.exists():
        raise SystemExit("No frozen config. Run `python -m src.freeze` first.")
    frozen, now = json.loads(FREEZE_FILE.read_text()), snapshot(cfg)
    if frozen["config"] != now["config"]:
        raise SystemExit("Config differs from the frozen one. Phase 1 must run with the frozen config.")
    changed = [k for k in set(frozen["source_sha256"]) | set(now["source_sha256"])
               if frozen["source_sha256"].get(k) != now["source_sha256"].get(k)]
    if changed:
        raise SystemExit(f"Source changed since the freeze: {sorted(changed)}")


# ----------------------------------------------------------------------------- forecast (Phase 0 data only)
def forecast_phase1(cfg: dict) -> dict:
    processed = cfg["paths"]["processed_dir"]
    weekly = load_phase0(processed)
    T = pd.Timestamp(cfg["splits"]["phase1_forecast_origin"])
    assert weekly["ds"].max() == T, "forecast origin must be the last Phase 0 week"
    h = max(cfg["features"]["horizons"])
    start = cfg["backtest"]["origin_settings"][cfg["lgbm"]["origins_setting"]]["start"]
    train_origins = training_origins(start, T, cfg)  # every training target <= T (Phase 0)
    cache = build_cache(weekly, train_origins + [T], cfg, processed)
    use_ids = cfg["lgbm"]["use_ids"]
    r = lgbm_forecasts(cache, train_origins, T, cfg, cfg["features"]["scale_method"], [use_ids], weekly)[use_ids]

    ids = pd.Index(sorted(weekly["unique_id"].unique()))
    dates = pd.date_range(T + WEEK, T + h * WEEK, freq=WEEK)
    grid = pd.DataFrame({"unique_id": np.repeat(ids, h), "ds": np.tile(dates, len(ids))})
    lg = complete_with_zeros(r["forecast"], grid)
    lg["forecast"] = lg["forecast"] * cfg["lgbm"]["bias_multiplier"]
    if cfg["lgbm"]["forecast_cap_k"]:
        lg, cap_stats = cap_forecasts(lg, weekly, T, cfg["lgbm"]["forecast_cap_k"])
    else:
        cap_stats = {}
    base = complete_with_zeros(baseline_forecasts(weekly, T, cfg)[BASELINE], grid)
    for name, f in [("lgbm_final", lg), (BASELINE, base)]:
        assert len(f) == len(ids) * h and f["forecast"].notna().all() and (f["forecast"] >= 0).all(), name
        assert set(f["ds"]) == set(dates), name
    return {"weekly": weekly, "T": T, "lgbm": lg, "baseline": base, "cap_stats": cap_stats,
            "train_origins": train_origins, "train_rows": r["train_rows"], "rows_dropped": r["rows_dropped_for_shift"],
            "cache": cache, "importance": r["importance"]}


# ----------------------------------------------------------------------------- score (reads Phase 1 actuals)
def score_and_report(res: dict, cfg: dict) -> dict:
    actual = pd.read_parquet(cfg["paths"]["processed_dir"] / "phase1_actuals.parquet")[["unique_id", "ds", "y"]]
    weekly, T = res["weekly"], res["T"]
    models = {"lgbm_final": res["lgbm"], BASELINE: res["baseline"]}
    total = {}
    for name, f in models.items():
        s = vn1_score(f, actual, forecast_col="forecast")
        e = f.merge(actual, on=["unique_id", "ds"])
        total[name] = {**s, "signed_bias": float((e["forecast"] - e["y"]).sum() / e["y"].sum())}

    ids = actual["unique_id"].unique()
    band = volume_band_at_origin(weekly, T, ids)
    dtype = (load_origins(res["cache"], [T], ["unique_id", "demand_type"]).drop_duplicates("unique_id")
             .set_index("unique_id")["demand_type"].reindex(ids).fillna("no history"))
    groups = {dim: {n: vn1_score_by_group(f, actual, g, forecast_col="forecast") for n, f in models.items()}
              for dim, g in [("volume_band", band), ("demand_type", dtype)]}

    wk = actual.groupby("ds")["y"].sum().rename("actual").to_frame()
    for n, f in models.items():
        wk[n] = f.groupby("ds")["forecast"].sum()
        wk[f"{n} bias"] = (wk[n] - wk["actual"]) / wk["actual"]
    hol = us_holiday_weeks([T.year, T.year + 1])
    wk["holiday"] = [("Thanksgiving" if d in hol["thanksgiving"] else "Cyber week" if d in hol["cyber"]
                      else "Christmas" if d in hol["christmas"] else "") for d in wk.index]

    e = res["lgbm"].merge(actual, on=["unique_id", "ds"])
    e["ae"] = (e["forecast"] - e["y"]).abs()
    per = e.groupby("unique_id").agg(forecast_per_week=("forecast", "mean"), actual_per_week=("y", "mean"),
                                     actual_max_week=("y", "max"), abs_error=("ae", "sum"))
    per["error_share"] = per["abs_error"] / per["abs_error"].sum()
    top = per.sort_values("abs_error", ascending=False).head(10)
    hist = weekly[weekly["ds"] <= T]
    g = hist[hist["unique_id"].isin(top.index)].groupby("unique_id")
    ly = hist[hist["unique_id"].isin(top.index) & hist["ds"].between(T - 51 * WEEK, T - 39 * WEEK)]  # same 13 weeks last year
    top = top.join(pd.DataFrame({
        "mean_13wk_before": g["y"].apply(lambda s: s.tail(13).mean()),
        "max_52wk_before": g["y"].apply(lambda s: s.tail(52).max()),
        "same_weeks_last_year_mean": ly.groupby("unique_id")["y"].mean(),
    })).join(res["baseline"].groupby("unique_id")["forecast"].mean().rename("baseline_per_week"))
    top["demand_type"] = dtype.reindex(top.index)
    top["band"] = band.reindex(top.index)
    st = pd.read_parquet(cfg["paths"]["processed_dir"] / "series_table.parquet").set_index("unique_id")
    top = top.join(st[["level_shift_date", "group_level_shift_date", "has_suspected_stockout"]])

    for n, f in models.items():
        f.to_parquet(OUT / f"{n}.parquet", index=False)
    out = {"total": total, "train_rows": res["train_rows"], "n_train_origins": len(res["train_origins"]),
           "rows_dropped": res["rows_dropped"], "cap_stats": res["cap_stats"]}
    (OUT / "scores.json").write_text(json.dumps(out, indent=2, default=float))
    write_report(out, groups, wk, top, res, cfg)
    return out


def write_report(out, groups, wk, top, res, cfg):
    pct = "{:+.1%}"
    d = json.loads((PROJECT_ROOT / "outputs" / "round3" / "decisions.json").read_text())
    r2 = pd.read_csv(PROJECT_ROOT / "outputs" / "round2" / "scores.csv")
    base_bt = r2[r2["config"] == BASELINE].set_index("fold")["score"]
    tot = pd.DataFrame([{"model": n, "score": v["score"], "accuracy": v["accuracy"], "bias": v["bias"],
                         "signed bias": v["signed_bias"]} for n, v in out["total"].items()])
    cmp = pd.DataFrame([
        {"model": "lgbm_final", "backtest mean (A–D)": d["final_backtest_mean"],
         "backtest fold A (holiday)": d["final_backtest_scores"]["A"], "Phase 1": out["total"]["lgbm_final"]["score"]},
        {"model": BASELINE, "backtest mean (A–D)": base_bt.mean(), "backtest fold A (holiday)": base_bt["A"],
         "Phase 1": out["total"][BASELINE]["score"]},
    ])

    def grp(dim, order):
        a, b = groups[dim]["lgbm_final"], groups[dim][BASELINE]
        t = pd.DataFrame({"units": a["unit_share"], "lgbm score": a["score"], "baseline score": b["score"],
                          "lgbm signed bias": a["signed_bias"], "baseline signed bias": b["signed_bias"],
                          "lgbm error share": a["abs_error_share"]})
        t = t.reindex([o for o in order if o in t.index] + [o for o in t.index if o not in order])
        return _table(t.reset_index(names=dim), {"units": "{:.1%}", "lgbm score": "{:.3f}", "baseline score": "{:.3f}",
                                                 "lgbm signed bias": pct, "baseline signed bias": pct,
                                                 "lgbm error share": "{:.1%}"})

    w = wk.reset_index(names="week")
    w["week"] = w["week"].dt.date.astype(str)
    tp = top.reset_index()
    for c in ["level_shift_date", "group_level_shift_date"]:
        tp[c] = pd.to_datetime(tp[c]).dt.date.astype(str).replace("NaT", "")
    lines = [
        "# Phase 1 result (run once with the frozen config)",
        "",
        f"Trained on Phase 0 only: {out['n_train_origins']} origins ({res['train_origins'][0].date()} → "
        f"{res['train_origins'][-1].date()}), {out['train_rows']:,} rows"
        + (f", {out['rows_dropped']:,} dropped for crossing a level shift" if out["rows_dropped"] else "")
        + f". Forecast origin {res['T'].date()}, forecasts {(res['T'] + WEEK).date()} → {(res['T'] + 13 * WEEK).date()}. "
        "Config: `reports/final_config.md`. Forecasts: `outputs/phase1/`. Phase 2 not read.",
        "",
        "## Total",
        "",
        _table(tot, {"score": "{:.4f}", "accuracy": "{:.4f}", "bias": "{:.4f}", "signed bias": pct}),
        "",
        "## Phase 1 vs the backtest",
        "",
        _table(cmp, {"backtest mean (A–D)": "{:.4f}", "backtest fold A (holiday)": "{:.4f}", "Phase 1": "{:.4f}"}),
        "",
        "## By volume band (at the forecast origin)",
        "",
        grp("volume_band", ["top 1%", "top 1-10%", "rest 90%", "first sale after origin"]),
        "",
        "## By demand type (at the forecast origin)",
        "",
        grp("demand_type", ["smooth", "erratic", "intermittent", "lumpy", "no_sales"]),
        "",
        "## Week by week",
        "",
        _table(w, {"actual": "{:,.0f}", "lgbm_final": "{:,.0f}", BASELINE: "{:,.0f}",
                   "lgbm_final bias": pct, f"{BASELINE} bias": pct}),
        "",
        "## 10 series with the largest absolute error (lgbm_final)",
        "",
        _table(tp[["unique_id", "band", "demand_type", "mean_13wk_before", "max_52wk_before", "same_weeks_last_year_mean",
                   "forecast_per_week", "baseline_per_week", "actual_per_week", "actual_max_week", "error_share",
                   "level_shift_date", "group_level_shift_date", "has_suspected_stockout"]],
               {"mean_13wk_before": "{:,.0f}", "max_52wk_before": "{:,.0f}", "same_weeks_last_year_mean": "{:,.0f}",
                "forecast_per_week": "{:,.0f}", "baseline_per_week": "{:,.0f}", "actual_per_week": "{:,.0f}",
                "actual_max_week": "{:,.0f}", "error_share": "{:.1%}"}),
        "",
        "`same_weeks_last_year_mean` = mean of 2022-10-10 → 2023-01-02 (the Phase 1 weeks one year earlier).",
        "",
    ]
    (PROJECT_ROOT / "reports" / "phase1.md").write_text("\n".join(lines))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true", help="run again even though Phase 1 was already scored")
    args = ap.parse_args()
    cfg = get_config()
    check_frozen(cfg)
    if (OUT / "scores.json").exists() and not args.force:
        raise SystemExit("Phase 1 was already scored (outputs/phase1/scores.json). It runs once.")
    res = forecast_phase1(cfg)
    for n, f in [("lgbm_final", res["lgbm"]), (BASELINE, res["baseline"])]:  # forecasts on disk before scoring
        f.to_parquet(OUT / f"{n}.parquet", index=False)
    out = score_and_report(res, cfg)
    print(json.dumps(out["total"], indent=2, default=float))


if __name__ == "__main__":
    main()

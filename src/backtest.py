"""Backtest harness: several Phase 0 validation origins, LightGBM grid + baselines.   python -m src.backtest

Reads ONE data file: data/processed/phase0_clean.parquet (via src.evaluation.load_phase0).
Every choice (blend weight, best LightGBM config) is made from these backtest scores only.

Folds (config backtest.folds): each fold has a validation origin T and forecasts T+1 .. T+13.
Training origins for a fold run every `step_weeks` from the setting's start date, and only those
whose 13th target is <= T are kept, so no training target falls inside the validation window.
"""
import gc
import json
import resource
import time
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import get_config
from src.evaluation import actuals_after, complete_with_zeros, load_phase0, volume_band_at_origin
from src.feature_cache import build_cache, load_origins
from src.features import (apply_scaling, drop_rows_crossing_shift, group_shift_dates_upto, model_features,
                          set_categories)
from src.metrics import vn1_score, vn1_score_by_group
from src.models import lgbm
from src.models.baselines import blend, moving_avg, seasonal_index, seasonal_naive

WEEK = pd.Timedelta(weeks=1)
GROUP_ORDER = {"volume_band": ["top 1%", "top 1-10%", "rest 90%", "first sale after origin"],
               "demand_type": ["smooth", "erratic", "intermittent", "lumpy", "no_sales", "first sale after origin"]}


# ----------------------------------------------------------------------------- plan
def training_origins(start, valid_origin, cfg: dict) -> list[pd.Timestamp]:
    """Origins every step_weeks from start whose last target (T + 13) is <= valid_origin.

    Before the weekly switch the step can be widened (step_weeks_before_switch) to save memory.
    """
    b = cfg["backtest"]
    h_max = max(cfg["features"]["horizons"])
    valid_origin = pd.Timestamp(valid_origin)
    cands = pd.date_range(pd.Timestamp(start), valid_origin, freq=b["step_weeks"] * WEEK)
    keep = [t for t in cands if t + h_max * WEEK <= valid_origin]
    ratio = max(1, b["step_weeks_before_switch"] // b["step_weeks"])
    switch = pd.Timestamp(cfg["features"]["switch_date"])
    return [t for i, t in enumerate(keep) if t >= switch or i % ratio == 0]


def fold_plan(cfg: dict) -> list[dict]:
    plan = []
    for fold, origin in cfg["backtest"]["folds"].items():
        for setting, s in cfg["backtest"]["origin_settings"].items():
            plan.append({"fold": fold, "origin": pd.Timestamp(origin), "setting": setting,
                         "train_origins": training_origins(s["start"], origin, cfg)})
    return plan


def lgbm_name(setting: str, scale: str, use_ids: bool) -> str:
    return f"lgbm_{setting}_{scale}_{'ids' if use_ids else 'noids'}"


# ----------------------------------------------------------------------------- one fold
def baseline_forecasts(weekly: pd.DataFrame, T: pd.Timestamp, cfg: dict) -> dict[str, pd.DataFrame]:
    b, h = cfg["baselines"], max(cfg["features"]["horizons"])
    train = weekly[weekly["ds"] <= T]
    ma = moving_avg(train, h, 13)
    sn = seasonal_naive(train, h, b["season_length"], b["fallback_window"])
    out = {"moving_avg_13": ma, "seasonal_naive": sn,
           "seasonal_index": seasonal_index(train, h, 13, b["season_length"], tuple(cfg["backtest"]["seasonal_index_clip"]))}
    for w in cfg["backtest"]["blend_weights"]:
        out[f"blend_{w:g}"] = blend(sn, ma, w)  # w x seasonal naive + (1 - w) x MA13
    return out


def lgbm_forecasts(cache, train_origins, T, cfg, scale: str, ids_options, weekly=None) -> dict:
    """Fit one LightGBM per ids option on the fold's training origins; predict at T.

    With level_shift.restart_after_shift, training rows crossing a group level-shift event
    (detected from data <= T) are dropped; `weekly` is then required.
    """
    feats_all = model_features(cfg)
    train = load_origins(cache, train_origins)
    n_dropped = 0
    if cfg["level_shift"].get("restart_after_shift"):
        if weekly is None:
            raise ValueError("restart_after_shift needs the weekly data to find level-shift events")
        before = len(train)
        train = drop_rows_crossing_shift(train, group_shift_dates_upto(weekly, T, cfg))
        n_dropped = before - len(train)
    train = train.drop(columns=["unique_id", "product", "origin"])  # strings / unused: save memory
    assert train["target_date"].max() <= T, "a training target falls inside the validation window"
    train = apply_scaling(train, cfg, scale, keep_raw=False)
    cats = set_categories(train)
    valid = apply_scaling(load_origins(cache, [T]), cfg, scale, keep_raw=False)
    set_categories(valid, cats)
    out = {}
    for use_ids in ids_options:
        feats = lgbm.select_features(feats_all, use_ids)
        t0 = time.time()
        model = lgbm.fit(train, feats, cfg["lgbm"]["params"], weight_by_scale=cfg["lgbm"]["weight_by_scale"])
        out[use_ids] = {"forecast": lgbm.predict(model, valid, feats), "importance": lgbm.importance(model, feats),
                        "fit_seconds": time.time() - t0, "train_rows": int(train["target"].notna().sum()),
                        "rows_dropped_for_shift": n_dropped}
        del model
    del train, valid
    gc.collect()
    return out


def _score_row(fc, actual, **meta) -> dict:
    s = vn1_score(fc, actual, forecast_col="forecast")
    e = fc.merge(actual, on=["unique_id", "ds"])
    return {**meta, "score": s["score"], "accuracy": s["accuracy"], "bias": s["bias"],
            "signed_bias": (e["forecast"] - e["y"]).sum() / e["y"].sum()}


# ----------------------------------------------------------------------------- run
def run_backtest(cfg: dict, processed_dir=None, out_dir=None, reports_dir=None, write_report: bool = True) -> dict:
    processed_dir = Path(processed_dir or cfg["paths"]["processed_dir"])
    out_dir = Path(out_dir or cfg["backtest"]["output_dir"])
    reports_dir = Path(reports_dir or cfg["paths"]["reports_dir"])
    (out_dir / "forecasts").mkdir(parents=True, exist_ok=True)
    h = max(cfg["features"]["horizons"])
    grid = cfg["backtest"]["lgbm_grid"]
    weekly = load_phase0(processed_dir)
    plan = fold_plan(cfg)

    t0 = time.time()
    needed = {o for p in plan if p["setting"] in grid["origins"] for o in p["train_origins"] + [p["origin"]]}
    cache = build_cache(weekly, needed, cfg, processed_dir)
    cache_seconds = time.time() - t0

    rows, forecasts, importances, groups, skipped, fit_log = [], {}, [], {}, [], []
    for fold, origin in cfg["backtest"]["folds"].items():
        T = pd.Timestamp(origin)
        actual = actuals_after(weekly, T, h)
        ids = actual["unique_id"].unique()
        valid_meta = load_origins(cache, [T], ["unique_id", "demand_type"]).drop_duplicates("unique_id")
        dtype = valid_meta.set_index("unique_id")["demand_type"].reindex(ids).fillna("first sale after origin")
        groups[fold] = {"volume_band": volume_band_at_origin(weekly, T, ids), "demand_type": dtype}

        for name, fc in baseline_forecasts(weekly, T, cfg).items():
            fc = complete_with_zeros(fc, actual)
            forecasts[(name, fold)] = fc
            rows.append(_score_row(fc, actual, fold=fold, method=name, family="baseline"))

        for p in [p for p in plan if p["fold"] == fold and p["setting"] in grid["origins"]]:
            if not p["train_origins"]:
                skipped.append(f"fold {fold}, setting `{p['setting']}`: no training origin has all 13 targets "
                               f"<= {T.date()} (setting starts {cfg['backtest']['origin_settings'][p['setting']]['start']})")
                continue
            for scale in grid["scale"]:
                res = lgbm_forecasts(cache, p["train_origins"], T, cfg, scale, grid["ids"], weekly)
                for use_ids, r in res.items():
                    name = lgbm_name(p["setting"], scale, use_ids)
                    fc = complete_with_zeros(r["forecast"], actual)
                    forecasts[(name, fold)] = fc
                    rows.append(_score_row(fc, actual, fold=fold, method=name, family="lgbm", setting=p["setting"],
                                           scale=scale, ids=use_ids))
                    importances.append(r["importance"].assign(method=name, fold=fold))
                    fit_log.append({"fold": fold, "method": name, "train_rows": r["train_rows"],
                                    "n_train_origins": len(p["train_origins"]), "fit_seconds": round(r["fit_seconds"], 1)})
                    print(f"fold {fold} {name}: {rows[-1]['score']:.4f}  ({r['train_rows']:,} rows, {r['fit_seconds']:.0f}s)",
                          flush=True)

    scores = pd.DataFrame(rows)
    scores.to_csv(out_dir / "scores_by_fold.csv", index=False)
    pd.DataFrame(fit_log).to_csv(out_dir / "fit_log.csv", index=False)
    for (name, fold), fc in forecasts.items():
        fc.assign(fold=fold).to_parquet(out_dir / "forecasts" / f"{name}__{fold}.parquet", index=False)

    summary = summarise(scores, cfg)
    peak_gb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9  # bytes on macOS
    run_info = {"cache_seconds": round(cache_seconds), "total_seconds": round(time.time() - t0),
                "peak_memory_gb": round(peak_gb, 2), "n_cached_origins": len(needed), "skipped": skipped}
    result = {"scores": scores, "summary": summary, "forecasts": forecasts, "groups": groups,
              "importances": pd.concat(importances, ignore_index=True) if importances else pd.DataFrame(),
              "fit_log": pd.DataFrame(fit_log), "run_info": run_info, "weekly": weekly}
    if write_report:
        write_backtest_report(result, cfg, reports_dir)
    print(json.dumps({"best_lgbm": summary["best_lgbm"], "best_baseline": summary["best_baseline"],
                      "blend_w": summary["blend_w"], **{k: v for k, v in run_info.items() if k != "skipped"}},
                     indent=2, default=str))
    return result


# ----------------------------------------------------------------------------- choices (backtest only)
def summarise(scores: pd.DataFrame, cfg: dict) -> dict:
    """All model choices, made from backtest scores only (no Phase 1 involved)."""
    folds = list(cfg["backtest"]["folds"])
    non_holiday = [f for f in folds if f not in cfg["backtest"]["holiday_folds"]]
    wide = scores.pivot(index="method", columns="fold", values="score").reindex(columns=folds)
    wide["mean_all"] = wide[folds].mean(axis=1, skipna=False)          # NaN if a fold is missing
    wide["mean_non_holiday"] = wide[non_holiday].mean(axis=1, skipna=False)
    fam = scores.drop_duplicates("method").set_index("method")["family"]
    wide["family"] = fam

    blends = [m for m in wide.index if m.startswith("blend_")]
    blend_best = wide.loc[blends, "mean_all"].idxmin() if blends else None
    base_candidates = [m for m in ["moving_avg_13", "seasonal_naive", "seasonal_index", blend_best] if m in wide.index]
    best_baseline = wide.loc[base_candidates, "mean_all"].idxmin()
    lg = wide[wide["family"].eq("lgbm")]
    complete = lg[lg["mean_all"].notna()]
    best_lgbm = (complete["mean_all"].idxmin() if len(complete) else
                 lg["mean_non_holiday"].idxmin() if len(lg) else None)
    return {"wide": wide.sort_values(["mean_all", "mean_non_holiday"]), "blend_w": blend_best,
            "best_baseline": best_baseline, "best_lgbm": best_lgbm, "folds": folds, "non_holiday": non_holiday}


def paired_effects(scores: pd.DataFrame) -> pd.DataFrame:
    """Average score change from flipping one LightGBM factor, others held equal, on folds both sides have."""
    lg = scores[scores["family"].eq("lgbm")].copy()
    out = []
    for factor, a, b in [("setting", "full", "weekly_only"), ("scale", "mean13", "max13_52"), ("ids", True, False)]:
        if lg[factor].nunique() < 2:
            continue
        others = [c for c in ["setting", "scale", "ids", "fold"] if c != factor]
        p = lg.pivot_table(index=others, columns=factor, values="score").dropna()
        if p.empty:
            continue
        d = p[b] - p[a]
        out.append({"change": f"{factor}: {a} → {b}", "mean score change": d.mean(), "pairs": len(d),
                    "better in": f"{int((d < 0).sum())}/{len(d)}",
                    "folds compared": ", ".join(sorted(p.index.get_level_values("fold").unique()))})
    return pd.DataFrame(out)


# ----------------------------------------------------------------------------- report
def _table(df: pd.DataFrame, fmt: dict | None = None) -> str:
    fmt = fmt or {}
    cols = list(df.columns)
    lines = ["| " + " | ".join(map(str, cols)) + " |", "|" + "---|" * len(cols)]
    for r in df.itertuples(index=False):
        cells = []
        for c, v in zip(cols, r):
            if c in fmt and isinstance(v, (int, float, np.floating)) and not pd.isna(v):
                cells.append(fmt[c].format(v))
            elif isinstance(v, (float, np.floating)):
                cells.append("–" if pd.isna(v) else f"{v:.4f}")
            else:
                cells.append(str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _pooled(result, method, dim):
    """Score by group, pooling every fold (keys prefixed with the fold so overlapping weeks don't collide)."""
    fcs, acts, maps = [], [], []
    for fold in result["summary"]["folds"]:
        if (method, fold) not in result["forecasts"]:
            continue
        fc = result["forecasts"][(method, fold)]
        fcs.append(fc.assign(unique_id=fold + "|" + fc["unique_id"]))
        acts.append(fc[["unique_id", "ds"]].assign(unique_id=fold + "|" + fc["unique_id"]))
        g = result["groups"][fold][dim]
        maps.append(pd.Series(g.to_numpy(), index=fold + "|" + g.index))
    fc, grp = pd.concat(fcs), pd.concat(maps)
    actual = _pooled_actuals(result, method)
    return vn1_score_by_group(fc, actual, grp, forecast_col="forecast")


def _pooled_actuals(result, method):
    w = result["weekly"]
    out = []
    h = 13
    for fold, origin in result["cfg_folds"].items():
        if (method, fold) in result["forecasts"]:
            a = actuals_after(w, pd.Timestamp(origin), h)
            out.append(a.assign(unique_id=fold + "|" + a["unique_id"]))
    return pd.concat(out)


def _by_horizon(result, method):
    parts = []
    for fold, origin in result["cfg_folds"].items():
        if (method, fold) in result["forecasts"]:
            a = actuals_after(result["weekly"], pd.Timestamp(origin), 13)
            e = result["forecasts"][(method, fold)].merge(a, on=["unique_id", "ds"])
            parts.append(e.assign(h=(e["ds"] - pd.Timestamp(origin)).dt.days // 7, err=e["forecast"] - e["y"]))
    e = pd.concat(parts)
    g = e.groupby("h")
    return (g["err"].apply(lambda x: x.abs().sum()) + g["err"].sum().abs()) / g["y"].sum()


def write_backtest_report(result, cfg, reports_dir):
    result["cfg_folds"] = cfg["backtest"]["folds"]
    s, scores, info = result["summary"], result["scores"], result["run_info"]
    folds, nh = s["folds"], s["non_holiday"]
    wide = s["wide"].reset_index()
    best_l, best_b = s["best_lgbm"], s["best_baseline"]
    pct = "{:+.1%}"

    lines = [
        "# Backtest: LightGBM vs baselines on Phase 0 folds",
        "",
        "`python -m src.backtest`. Reads only `phase0_clean.parquet`; Phase 1 and 2 are not used. "
        "Every choice below (blend weight, best configuration) comes from these scores.",
        "",
        "## Setup",
        "",
        "| fold | origin | targets | training origins: full | training origins: weekly_only |",
        "|---|---|---|---|---|",
    ]
    plan = fold_plan(cfg)
    for fold, origin in cfg["backtest"]["folds"].items():
        T = pd.Timestamp(origin)
        n = {p["setting"]: p["train_origins"] for p in plan if p["fold"] == fold}
        desc = lambda o: f"{len(o)} ({o[0].date()} → {o[-1].date()})" if o else "none"  # noqa: E731
        lines.append(f"| {fold}{' (holiday)' if fold in cfg['backtest']['holiday_folds'] else ''} | {T.date()} | "
                     f"{(T + WEEK).date()} → {(T + 13 * WEEK).date()} | {desc(n.get('full', []))} | "
                     f"{desc(n.get('weekly_only', []))} |")
    lines += [
        "",
        f"- LightGBM: tweedie, default params (100 trees), seed 0, rows weighted by scale. "
        f"Grid: scale {cfg['backtest']['lgbm_grid']['scale']} × client/warehouse {{with, without}} × "
        f"training origins {cfg['backtest']['lgbm_grid']['origins']}.",
        "- Series that first sell after a fold's origin get 0 from every method.",
        f"- Run: {info['n_cached_origins']} origins cached ({info['cache_seconds']} s), total {info['total_seconds']} s, "
        f"peak memory {info['peak_memory_gb']} GB.",
    ]
    lines += [f"- Skipped: {x}." for x in info["skipped"]]
    lines += [
        "",
        "## Scores (VN1, lower is better)",
        "",
        f"`mean all` = mean over folds {', '.join(folds)} (– if a fold is missing); "
        f"`mean non-holiday` = mean over {', '.join(nh)}.",
        "",
        _table(wide[["method"] + folds + ["mean_all", "mean_non_holiday"]].rename(
            columns={"mean_all": "mean all", "mean_non_holiday": "mean non-holiday"})),
        "",
        f"- Blend weight chosen by mean score over all folds: **{s['blend_w']}** "
        f"(w × seasonal naive + (1 − w) × MA13).",
        f"- Best baseline: **{best_b}**. Best LightGBM (all folds): **{best_l}**.",
        "",
        "## Does each LightGBM choice matter?",
        "",
        "Score change when flipping one factor with the others held fixed, averaged over matching pairs "
        "(negative = the change helps).",
        "",
        _table(paired_effects(scores), {"mean score change": "{:+.4f}"}),
        "",
        "## Accuracy, bias and signed bias per fold",
        "",
    ]
    keep = [m for m in wide["method"] if m in ("moving_avg_13", "seasonal_naive", "seasonal_index", s["blend_w"])
            or m.startswith("lgbm_")]
    det = scores[scores["method"].isin(keep)].sort_values(["fold", "score"])
    lines += [_table(det[["fold", "method", "score", "accuracy", "bias", "signed_bias"]], {"signed_bias": pct}), ""]

    for dim, title in [("volume_band", "volume band at origin"), ("demand_type", "demand type at origin")]:
        tl, tb = _pooled(result, best_l, dim), _pooled(result, best_b, dim)
        t = pd.DataFrame({"units": tl["unit_share"], f"{best_l}": tl["score"], f"{best_b}": tb["score"],
                          f"signed bias {best_l}": tl["signed_bias"], f"signed bias {best_b}": tb["signed_bias"],
                          f"error share {best_l}": tl["abs_error_share"]})
        order = GROUP_ORDER[dim]
        t = t.reindex([g for g in order if g in t.index] + [g for g in t.index if g not in order]).reset_index(names=dim)
        lines += [f"## Best LightGBM vs best baseline by {title} (all folds pooled)", "",
                  _table(t, {"units": "{:.1%}", f"signed bias {best_l}": pct, f"signed bias {best_b}": pct,
                             f"error share {best_l}": "{:.1%}"}), ""]
    hz = pd.DataFrame({"h": range(1, 14), best_l: _by_horizon(result, best_l).to_numpy(),
                       best_b: _by_horizon(result, best_b).to_numpy()})
    lines += ["## By horizon (all folds pooled)", "", _table(hz), ""]

    hol = cfg["backtest"]["holiday_folds"]
    for fold in hol:
        T = pd.Timestamp(cfg["backtest"]["folds"][fold])
        a = actuals_after(result["weekly"], T, 13).groupby("ds")["y"].sum().rename("actual")
        cols = {}
        for m in dict.fromkeys([best_l, best_b, "moving_avg_13", "seasonal_naive"]):
            if (m, fold) in result["forecasts"]:
                cols[m] = result["forecasts"][(m, fold)].groupby("ds")["forecast"].sum()
        t = pd.concat([a, pd.DataFrame(cols)], axis=1)
        for m in cols:
            t[f"{m} vs actual"] = (t[m] - t["actual"]) / t["actual"]
        t = t.reset_index(names="week")
        t["week"] = t["week"].dt.date.astype(str)
        fmt = {c: "{:,.0f}" for c in ["actual", *cols]} | {f"{m} vs actual": pct for m in cols}
        lines += [f"## Fold {fold} (holiday): total units per week", "", _table(t, fmt), ""]

    imp = result["importances"]
    if len(imp) and best_l:
        top = (imp[imp["method"].eq(best_l)].groupby("feature")["gain_share"].mean()
               .sort_values(ascending=False).head(20).reset_index())
        lines += [f"## Top 20 features of {best_l} (gain share, mean over folds)", "",
                  _table(top, {"gain_share": "{:.1%}"}), ""]
    fl = result["fit_log"]
    if len(fl):
        lines += ["## Training set sizes", "",
                  _table(fl.groupby(["fold"]).agg(train_rows=("train_rows", "max"), origins=("n_train_origins", "max"),
                                                  fit_seconds=("fit_seconds", "mean")).reset_index(),
                         {"train_rows": "{:,.0f}", "origins": "{:.0f}", "fit_seconds": "{:.0f}"}), "",
                  "(largest setting per fold)", ""]
    (Path(reports_dir) / "backtest.md").write_text("\n".join(lines))


if __name__ == "__main__":
    run_backtest(get_config())

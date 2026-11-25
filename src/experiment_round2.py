"""Round 2 on the backtest folds (Phase 0 only):   python -m src.experiment_round2

1. Locked default: full-history origins, max13_52 scale, no client/warehouse (config lgbm.*).
2. Explain the fold B blow-ups of the weekly_only + client/warehouse configs.
3. Forecast cap (k x max weekly sales of the last 52 weeks), k = 1.5 / 2 / 3.
4. Fold A (holiday) diagnosis: LightGBM vs best baseline by band / demand type, peak weeks, top-10 plot.
5. Two seasonality improvements (optional feature groups "uplift" and "holiday_distance").
6. 50/50 blend of the best LightGBM with the best baseline.
Writes reports/experiments_round2.md, reports/figures/foldA_top10.png, outputs/round2/.
"""
import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.backtest import _table, baseline_forecasts, lgbm_forecasts, training_origins  # noqa: E402
from src.config import get_config  # noqa: E402
from src.evaluation import actuals_after, complete_with_zeros, load_phase0, volume_band_at_origin  # noqa: E402
from src.feature_cache import build_cache, load_origins  # noqa: E402
from src.features import us_holiday_weeks  # noqa: E402
from src.metrics import vn1_score, vn1_score_by_group  # noqa: E402
from src.models.baselines import blend  # noqa: E402
from src.models.postprocess import cap_forecasts  # noqa: E402
from src.report import AQUA, BLUE, GRID, INK2, ORANGE  # noqa: E402

WEEK = pd.Timedelta(weeks=1)
BEST_BASELINE = "blend_0.25"  # chosen in the backtest by mean score over all folds
BLOWUPS = ["lgbm_weekly_only_mean13_ids", "lgbm_weekly_only_max13_52_ids"]


def _score(fc, actual) -> dict:
    s = vn1_score(fc, actual, forecast_col="forecast")
    e = fc.merge(actual, on=["unique_id", "ds"])
    return {"score": s["score"], "accuracy": s["accuracy"], "bias": s["bias"],
            "signed_bias": (e["forecast"] - e["y"]).sum() / e["y"].sum()}


def run(cfg: dict, processed_dir=None, out_dir=None, reports_dir=None) -> dict:
    processed_dir = Path(processed_dir or cfg["paths"]["processed_dir"])
    out_dir = Path(out_dir or cfg["paths"]["processed_dir"].parents[1] / "outputs" / "round2")
    reports_dir = Path(reports_dir or cfg["paths"]["reports_dir"])
    (reports_dir / "figures").mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    h = max(cfg["features"]["horizons"])
    folds = {f: pd.Timestamp(o) for f, o in cfg["backtest"]["folds"].items()}
    weekly = load_phase0(processed_dir)
    start = cfg["backtest"]["origin_settings"][cfg["lgbm"]["origins_setting"]]["start"]
    plan = {f: training_origins(start, T, cfg) for f, T in folds.items()}
    cache = build_cache(weekly, {o for v in plan.values() for o in v} | set(folds.values()), cfg, processed_dir)
    actual = {f: actuals_after(weekly, T, h) for f, T in folds.items()}

    fc: dict[str, dict[str, pd.DataFrame]] = {}   # config name -> fold -> forecast
    notes: dict[str, dict] = {}
    imps = {}

    # ---- baselines
    for f, T in folds.items():
        for name, b in baseline_forecasts(weekly, T, cfg).items():
            if name in ("moving_avg_13", "seasonal_naive", BEST_BASELINE):
                fc.setdefault(name, {})[f] = complete_with_zeros(b, actual[f])

    # ---- LightGBM variants (locked defaults; optional feature groups switched on per variant)
    def lgbm_variant(name, optional):
        c = get_config({"features": {"optional_features": optional}})
        for f, T in folds.items():
            r = lgbm_forecasts(cache, plan[f], T, c, c["features"]["scale_method"], [c["lgbm"]["use_ids"]])
            r = r[c["lgbm"]["use_ids"]]
            fc.setdefault(name, {})[f] = complete_with_zeros(r["forecast"], actual[f])
            imps.setdefault(name, []).append(r["importance"].assign(fold=f))
            print(f"{name} fold {f}: {_score(fc[name][f], actual[f])['score']:.4f}", flush=True)

    lgbm_variant("lgbm_default", [])

    # ---- cap
    cap_stats = []
    for k in (1.5, 2.0, 3.0):
        name = f"lgbm_default + cap k={k:g}"
        for f, T in folds.items():
            fc.setdefault(name, {})[f], st = cap_forecasts(fc["lgbm_default"][f], weekly, T, k)
            cap_stats.append({"k": k, "fold": f, **st})
    cap_stats = pd.DataFrame(cap_stats)

    def mean_score(name):
        return np.mean([_score(fc[name][f], actual[f])["score"] for f in folds])

    k_best = 2.0
    use_cap = bool(mean_score("lgbm_default + cap k=2") < mean_score("lgbm_default"))
    post = (lambda x, f: cap_forecasts(x, weekly, folds[f], k_best)[0]) if use_cap else (lambda x, f: x)
    suffix = " + cap k=2" if use_cap else ""

    # ---- fold B blow-ups (forecasts saved by src.backtest), with and without the cap
    bt_dir = Path(cfg["backtest"]["output_dir"]) / "forecasts"
    for m in BLOWUPS:
        p = bt_dir / f"{m}__B.parquet"
        if p.exists():
            raw = pd.read_parquet(p)[["unique_id", "ds", "forecast"]]
            fc[f"{m} (fold B only)"] = {"B": raw}
            fc[f"{m} (fold B only) + cap k=2"] = {"B": cap_forecasts(raw, weekly, folds["B"], 2.0)[0]}
    blow = blowup_table(weekly, folds["B"], actual["B"], bt_dir)

    # ---- seasonality improvements (same post-processing as the default)
    lgbm_variant("lgbm + uplift", ["uplift"])
    lgbm_variant("lgbm + holiday_distance", ["holiday_distance"])
    for name in ["lgbm_default", "lgbm + uplift", "lgbm + holiday_distance"]:
        fc[name + suffix] = {f: post(fc[name][f], f) for f in folds}
    candidates = ["lgbm_default" + suffix, "lgbm + uplift" + suffix, "lgbm + holiday_distance" + suffix]
    best_lgbm = min(candidates, key=mean_score)

    # ---- blend LightGBM 50/50 with the best baseline
    blend_name = f"0.5 × {best_lgbm} + 0.5 × {BEST_BASELINE}"
    fc[blend_name] = {f: blend(fc[best_lgbm][f], fc[BEST_BASELINE][f], 0.5) for f in folds}

    # ---- scores
    rows = []
    for name, per_fold in fc.items():
        for f, x in per_fold.items():
            rows.append({"config": name, "fold": f, **_score(x, actual[f])})
    scores = pd.DataFrame(rows)
    scores.to_csv(out_dir / "scores.csv", index=False)
    cap_stats.to_csv(out_dir / "cap_stats.csv", index=False)

    diag = fold_a_diagnosis(weekly, folds["A"], actual["A"], fc["lgbm_default" + suffix]["A"], fc[BEST_BASELINE]["A"],
                            cache, reports_dir / "figures" / "foldA_top10.png", "lgbm_default" + suffix)
    result = {"scores": scores, "cap_stats": cap_stats, "use_cap": use_cap, "best_lgbm": best_lgbm,
              "blend_name": blend_name, "blowups": blow, "diag": diag, "imps": imps, "folds": folds,
              "plan": plan, "seconds": round(time.time() - t0)}
    write_report(result, cfg, reports_dir)
    print(json.dumps({"use_cap": use_cap, "best_lgbm": best_lgbm, "seconds": result["seconds"]}, indent=2))
    return result


# ----------------------------------------------------------------------------- diagnostics
def blowup_table(weekly, T, actual, bt_dir) -> dict:
    """Top-10 error series of the worst blow-up config on fold B, with their history at the origin."""
    m = BLOWUPS[-1]
    f_ids = pd.read_parquet(bt_dir / f"{m}__B.parquet")
    f_no = pd.read_parquet(bt_dir / f"{m.replace('_ids', '_noids')}__B.parquet")
    e = f_ids.merge(actual, on=["unique_id", "ds"])
    e["ae"] = (e["forecast"] - e["y"]).abs()
    per = e.groupby("unique_id").agg(forecast_per_week=("forecast", "mean"), actual_per_week=("y", "mean"), ae=("ae", "sum"))
    per["error_share"] = per["ae"] / per["ae"].sum()
    top = per.sort_values("ae", ascending=False).head(10)
    hist = weekly[weekly["ds"] <= T]
    g = hist[hist["unique_id"].isin(top.index)].groupby("unique_id")
    info = pd.DataFrame({
        "warehouse": g["warehouse"].first(), "first_sale": g["ds"].min().dt.date,
        "mean_13wk_before_origin": g["y"].apply(lambda s: s.tail(13).mean()),
        "max_52wk_before_origin": g["y"].apply(lambda s: s.tail(52).max()),
        "mean_before_2022_11_07": hist[hist["unique_id"].isin(top.index) & (hist["ds"] < "2022-11-07")]
        .groupby("unique_id")["y"].apply(lambda s: s.tail(13).mean()),
    })
    no = f_no.groupby("unique_id")["forecast"].mean().rename("forecast_per_week_without_ids")
    t = top.join(info).join(no)

    # Warehouse 251 jump evidence (data <= fold B origin).
    w = weekly[(weekly["warehouse"] == 251) & (weekly["ds"] <= T)]
    pre = w[(w["ds"] >= "2022-08-01") & (w["ds"] < "2022-11-07")].groupby("unique_id")["y"].mean()
    post = w[(w["ds"] >= "2022-11-07")].groupby("unique_id")["y"].mean()
    ratio = (post / pre.replace(0, np.nan)).dropna()
    tot_fc, tot_y = e["forecast"].sum(), e["y"].sum()
    wh = e.merge(weekly[["unique_id", "warehouse"]].drop_duplicates(), on="unique_id")
    return {"top": t, "model": m, "total_forecast": tot_fc, "total_actual": tot_y,
            "top10_share": top["error_share"].sum(),
            "wh251_error_share": wh.loc[wh["warehouse"] == 251, "ae"].sum() / e["ae"].sum(),
            "wh251_jump_median": ratio.median(), "wh251_n_compared": len(ratio),
            "wh251_share_over_20x": (ratio > 20).mean(),
            "wh251_new_series": int((w.groupby("unique_id")["ds"].min() >= "2022-11-07").sum())}


def fold_a_diagnosis(weekly, T, actual, f_lgbm, f_base, cache, fig_path, lgbm_label) -> dict:
    ids = actual["unique_id"].unique()
    band = volume_band_at_origin(weekly, T, ids)
    dt = load_origins(cache, [T], ["unique_id", "demand_type"]).drop_duplicates("unique_id").set_index("unique_id")["demand_type"]
    dt = dt.reindex(ids).fillna("first sale after origin")
    tables = {}
    for dim, grp in [("volume_band", band), ("demand_type", dt)]:
        a = vn1_score_by_group(f_lgbm, actual, grp, forecast_col="forecast")
        b = vn1_score_by_group(f_base, actual, grp, forecast_col="forecast")
        tables[dim] = pd.DataFrame({"units": a["unit_share"], "lgbm score": a["score"], "baseline score": b["score"],
                                    "lgbm signed bias": a["signed_bias"], "baseline signed bias": b["signed_bias"],
                                    "lgbm error share": a["abs_error_share"]})

    # Peak = Thanksgiving week .. the week before Christmas week; holiday tail = Christmas week .. 2 weeks after.
    hol = us_holiday_weeks([T.year])
    tg, xm = min(hol["thanksgiving"]), min(hol["christmas"])
    periods = {"before peak": (T + WEEK, tg - WEEK), "peak (Thanksgiving → pre-Christmas)": (tg, xm - WEEK),
               "Christmas → New Year": (xm, xm + 2 * WEEK), "January": (xm + 3 * WEEK, T + 13 * WEEK)}
    e = actual.merge(f_lgbm, on=["unique_id", "ds"]).merge(f_base, on=["unique_id", "ds"], suffixes=("_l", "_b"))
    per = []
    for name, (a, b) in periods.items():
        x = e[e["ds"].between(a, b)]
        per.append({"period": f"{name} ({a.date()} → {b.date()})", "units": x["y"].sum(),
                    "lgbm signed bias": (x["forecast_l"] - x["y"]).sum() / x["y"].sum(),
                    "baseline signed bias": (x["forecast_b"] - x["y"]).sum() / x["y"].sum()})
    per = pd.DataFrame(per)

    # Top-10 series by units in the 52 weeks up to the origin.
    last52 = weekly[(weekly["ds"] <= T) & (weekly["ds"] > T - 52 * WEEK)]
    top = last52.groupby("unique_id")["y"].sum().sort_values(ascending=False).head(10).index
    top_e = e[e["unique_id"].isin(top)]
    top_peak = top_e[top_e["ds"].between(tg, xm - WEEK)]
    top_stats = {"units_share": top_e["y"].sum() / e["y"].sum(),
                 "lgbm_peak_bias": (top_peak["forecast_l"] - top_peak["y"]).sum() / top_peak["y"].sum(),
                 "base_peak_bias": (top_peak["forecast_b"] - top_peak["y"]).sum() / top_peak["y"].sum()}
    plot_top10(weekly, T, top, f_lgbm, f_base, fig_path, lgbm_label)
    return {"tables": tables, "periods": per, "top_stats": top_stats, "top_ids": list(top)}


def plot_top10(weekly, T, top, f_lgbm, f_base, path, lgbm_label):
    start, end = T - 60 * WEEK, T + 13 * WEEK
    hol = us_holiday_weeks([T.year - 1, T.year])
    fig, axes = plt.subplots(5, 2, figsize=(13, 13), sharex=True)
    for ax, uid in zip(axes.flat, top):
        d = weekly[(weekly["unique_id"] == uid) & weekly["ds"].between(start, end)]
        for wk in hol["thanksgiving"] | hol["cyber"]:
            ax.axvspan(wk, wk + WEEK, color=GRID, lw=0)
        for wk in hol["christmas"]:
            ax.axvspan(wk, wk + WEEK, color=GRID, lw=0, alpha=0.7)
        ax.plot(d["ds"], d["y"], color=BLUE, lw=1.3)
        for f, colr in [(f_lgbm, ORANGE), (f_base, AQUA)]:
            x = f[f["unique_id"] == uid]
            ax.plot(x["ds"], x["forecast"], color=colr, lw=1.6)
        ax.axvline(T, color=INK2, ls="--", lw=0.8)
        ax.set_title(uid, loc="left", fontsize=9)
        ax.set_ylim(bottom=0)
        ax.tick_params(axis="x", labelrotation=30)
    handles = [plt.Line2D([], [], color=BLUE, label="actual"),
               plt.Line2D([], [], color=ORANGE, label=lgbm_label),
               plt.Line2D([], [], color=AQUA, label=BEST_BASELINE),
               plt.Rectangle((0, 0), 1, 1, color=GRID, label="Thanksgiving, Cyber & Christmas weeks"),
               plt.Line2D([], [], color=INK2, ls="--", label="fold A origin")]
    fig.legend(handles=handles, loc="lower center", ncol=5, fontsize=8, frameon=False)
    fig.suptitle("Fold A: 10 highest-volume series (last 52 weeks before origin), 2021 and 2022 holidays",
                 x=0.01, ha="left", fontweight="bold")
    fig.tight_layout(rect=(0, 0.03, 1, 0.98))
    fig.savefig(path, dpi=120)
    plt.close(fig)


# ----------------------------------------------------------------------------- report
def write_report(r, cfg, reports_dir):
    sc, folds = r["scores"], list(r["folds"])
    pct = "{:+.1%}"

    def cell(name, f):
        x = sc[(sc["config"] == name) & (sc["fold"] == f)]
        return "–" if x.empty else f"{x['score'].iloc[0]:.3f} ({x['signed_bias'].iloc[0]:+.1%})"

    order = (["moving_avg_13", "seasonal_naive", BEST_BASELINE, "lgbm_default",
              "lgbm_default + cap k=1.5", "lgbm_default + cap k=2", "lgbm_default + cap k=3"]
             + [n for n in sc["config"].unique() if n.startswith("lgbm +") and "cap" not in n]
             + [n for n in sc["config"].unique() if n.startswith("lgbm +") and "cap" in n]
             + [r["blend_name"]] + [n for n in sc["config"].unique() if "fold B only" in n])
    order = list(dict.fromkeys(n for n in order if n in set(sc["config"])))
    main = []
    for name in order:
        x = sc[sc["config"] == name]
        full = x["fold"].nunique() == len(folds)
        main.append({"config": name, **{f: cell(name, f) for f in folds},
                     "average": f"{x['score'].mean():.3f}" if full else "–",
                     "avg signed bias": f"{x['signed_bias'].mean():+.1%}" if full else "–"})
    main = pd.DataFrame(main)

    b = r["blowups"]
    top = b["top"].reset_index()[["unique_id", "warehouse", "first_sale", "mean_before_2022_11_07",
                                  "mean_13wk_before_origin", "max_52wk_before_origin", "forecast_per_week",
                                  "forecast_per_week_without_ids", "actual_per_week", "error_share"]]
    cs = r["cap_stats"].groupby("k").agg(forecasts_capped=("share_capped", "mean"), series_capped=("series_capped", "mean"),
                                         units_removed=("units_removed_share", "mean")).reset_index()
    d = r["diag"]

    def diag_table(t, dim):
        return _table(t.reset_index(names=dim), {"units": "{:.1%}", "lgbm score": "{:.3f}", "baseline score": "{:.3f}",
                                                 "lgbm signed bias": pct, "baseline signed bias": pct,
                                                 "lgbm error share": "{:.1%}"})

    imp = pd.concat(r["imps"][r["best_lgbm"].replace(" + cap k=2", "")]).groupby("feature")["gain_share"].mean()
    new_feats = imp[imp.index.str.startswith(("ly_", "tw_weeks"))]
    lines = [
        "# Round 2: locked default, cap, fold A diagnosis, seasonality, blend",
        "",
        "`python -m src.experiment_round2`. Phase 0 only (backtest folds A–D); Phase 1 and 2 untouched. "
        f"Runtime {r['seconds']} s.",
        "",
        "**Locked default** (config `lgbm` / `features`): training origins `full` (from 2021-07-05), scale `max13_52`, "
        "no client/warehouse features, tweedie, default params, rows weighted by scale.",
        "",
        "## Results: every configuration × fold",
        "",
        "Cell = VN1 score (signed bias: + over-forecast, − under-forecast). Lower score is better. "
        f"Fold A = holiday (origin {r['folds']['A'].date()}).",
        "",
        _table(main),
        "",
        f"- Cap adopted for the variants below the default: **{'yes' if r['use_cap'] else 'no'}** "
        "(decided by the 4-fold average of the default with vs without cap k=2).",
        f"- Best LightGBM: **{r['best_lgbm']}**. Best baseline: **{BEST_BASELINE}** (from the backtest).",
        "",
        "## 2. Fold B blow-ups (2.34 / 3.22)",
        "",
        f"Config `{b['model']}` on fold B: total forecast {b['total_forecast']:,.0f} units vs actual "
        f"{b['total_actual']:,.0f} ({b['total_forecast'] / b['total_actual']:.1f}×). The top 10 series make "
        f"{b['top10_share']:.0%} of the absolute error; **warehouse 251 makes {b['wh251_error_share']:.0%}**.",
        "",
        "Cause: on **2022-11-07 warehouse 251 (client 2) jumped ~"
        f"{b['wh251_jump_median']:.0f}× overnight** (median after/before ratio over {b['wh251_n_compared']} series that sold "
        f"on both sides; {b['wh251_share_over_20x']:.0%} jumped > 20×; {b['wh251_new_series']} new series started there "
        "that week). The `weekly_only` training origins for fold B (2022-09-05 → 2022-10-31) have pre-jump scales "
        "(~1 unit/week) and post-jump targets (hundreds/week), so scaled targets of several hundred, all in warehouse 251. "
        "With `warehouse` as a feature the model learns \"warehouse 251 → ×hundreds\" and applies it at the fold B origin, "
        "where the scale is already post-jump: the multiplier is applied twice. Without client/warehouse it can't isolate "
        "the warehouse; with full history the 5 contaminated origins are diluted by 30 clean ones.",
        "",
        _table(top, {"mean_before_2022_11_07": "{:,.1f}", "mean_13wk_before_origin": "{:,.0f}",
                     "max_52wk_before_origin": "{:,.0f}", "forecast_per_week": "{:,.0f}",
                     "forecast_per_week_without_ids": "{:,.0f}", "actual_per_week": "{:,.0f}", "error_share": "{:.1%}"}),
        "",
        "Not caught by cleaning: the price-shift detector looks at price, and nothing looks for a level shift in "
        "*sales* across a whole warehouse. Worth adding (flag + per-series regime, like `price_regime`).",
        "",
        "## 3. Forecast cap",
        "",
        "forecast ≤ k × the series' max weekly sales in the 52 weeks up to the origin (0 if it sold nothing then). "
        "Averages over the four folds for the default model:",
        "",
        _table(cs, {"forecasts_capped": "{:.2%}", "series_capped": "{:,.0f}", "units_removed": "{:.2%}"}),
        "",
        "Scores are in the main table. The cap also fixes the fold B blow-ups (rows marked *fold B only*).",
        "",
        "## 4. Fold A (holiday) diagnosis",
        "",
        f"LightGBM = `{'lgbm_default' + (' + cap k=2' if r['use_cap'] else '')}`; baseline = `{BEST_BASELINE}`.",
        "",
        "### Signed bias by period",
        "",
        _table(d["periods"], {"units": "{:,.0f}", "lgbm signed bias": pct, "baseline signed bias": pct}),
        "",
        "### By volume band",
        "",
        diag_table(d["tables"]["volume_band"].reindex(
            [g for g in ["top 1%", "top 1-10%", "rest 90%", "first sale after origin"] if g in d["tables"]["volume_band"].index]),
            "volume band"),
        "",
        "### By demand type at origin",
        "",
        diag_table(d["tables"]["demand_type"], "demand type"),
        "",
        f"### Top 10 series ({d['top_stats']['units_share']:.0%} of fold A units)",
        "",
        f"Peak-period signed bias for these 10: LightGBM {d['top_stats']['lgbm_peak_bias']:+.1%}, "
        f"baseline {d['top_stats']['base_peak_bias']:+.1%}.",
        "",
        "![fold A top 10](figures/foldA_top10.png)",
        "",
        "## 5. Seasonality improvements",
        "",
        "- `uplift`: `ly_uplift_series`, `ly_uplift_client` = last year's target weeks ÷ last year's level at the same "
        "point (series and client totals). Gives the model last year's *shape* without last year's *level*.",
        "- `holiday_distance`: `tw_weeks_to_thanksgiving`, `tw_weeks_to_christmas` (signed weeks, ±13).",
        "",
        "Gain share of the new features in the best LightGBM (mean over folds): "
        + (", ".join(f"`{k}` {v:.1%}" for k, v in new_feats.sort_values(ascending=False).items()) or "n/a (not in best model)") + ".",
        "",
        "## 6. Blend",
        "",
        f"`{r['blend_name']}`: see the main table.",
        "",
    ]
    (Path(reports_dir) / "experiments_round2.md").write_text("\n".join(lines))


if __name__ == "__main__":
    run(get_config())

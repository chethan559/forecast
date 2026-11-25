"""Feature matrix for inner validation + LightGBM smoke test.   python -m src.run_features

Reads ONE file: data/processed/phase0_clean.parquet (src.evaluation.load_phase0). No Phase 1 / Phase 2 file is opened
(tests/test_features.py makes those files unreadable and runs this end to end).
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import get_config
from src.evaluation import complete_with_zeros, load_phase0, volume_band_at_origin
from src.features import CATEGORICAL, feature_specs, make_features, model_features
from src.metrics import vn1_score, vn1_score_by_group
from src.models.baselines import moving_avg
from src.splits import inner_validation_split

def build(cfg: dict, processed_dir=None, reports_dir=None, run_smoke: bool = True) -> dict:
    processed_dir = Path(processed_dir or cfg["paths"]["processed_dir"])
    reports_dir = Path(reports_dir or cfg["paths"]["reports_dir"])
    reports_dir.mkdir(parents=True, exist_ok=True)
    weekly = load_phase0(processed_dir)
    split = inner_validation_split(cfg)
    origins = split["train_origins"] + [split["valid_origin"]]

    X = make_features(weekly, origins, cfg["features"]["horizons"], cfg)
    X["split"] = np.where(X["origin"].eq(split["valid_origin"]), "valid", "train")
    # Guard: no training target may fall inside the validation window.
    assert X.loc[X["split"].eq("train"), "target_date"].max() <= split["valid_origin"]
    X.to_parquet(processed_dir / "features_inner.parquet", index=False)

    feats = model_features(cfg)
    info = {"rows": len(X), "rows_train": int(X["split"].eq("train").sum()),
            "rows_valid": int(X["split"].eq("valid").sum()), "columns": X.shape[1], "n_features": len(feats),
            "n_train_origins": len(split["train_origins"]),
            "memory_mb": round(X.memory_usage(deep=True).sum() / 1e6),
            "file_mb": round((processed_dir / "features_inner.parquet").stat().st_size / 1e6, 1)}

    smoke = run_smoke_test(X, weekly, split, cfg) if run_smoke else None
    write_features_report(X, info, smoke, split, cfg, reports_dir)
    print(json.dumps({**info, **({"lgbm": smoke["lgbm"]["score"], "ma13": smoke["ma13"]["score"]} if smoke else {})},
                     indent=2, default=float))
    return {"info": info, "smoke": smoke, "X": X}


def run_smoke_test(X: pd.DataFrame, weekly: pd.DataFrame, split: dict, cfg: dict) -> dict:
    from src.models.lgbm import fit_predict

    T = split["valid_origin"]
    feats = model_features(cfg)
    train, valid = X[X["split"].eq("train")], X[X["split"].eq("valid")]
    # Actuals = last 13 weeks of Phase 0, every series that has rows there.
    actual = weekly[(weekly["ds"] > T) & (weekly["ds"] <= split["valid_target_end"])][["unique_id", "ds", "y"]]

    model, fc, imp = fit_predict(train, valid, feats, cfg["lgbm_smoke"])
    fc = complete_with_zeros(fc, actual)
    base = complete_with_zeros(moving_avg(weekly[weekly["ds"] <= T], len(cfg["features"]["horizons"]), 13), actual)

    ids = actual["unique_id"].unique()
    band = volume_band_at_origin(weekly, T, ids)
    dtype = valid.drop_duplicates("unique_id").set_index("unique_id")["demand_type"].astype(str)
    dtype = dtype.reindex(ids).fillna("first sale after origin")
    by = {}
    for name, f in [("lgbm", fc), ("ma13", base)]:
        by[name] = {"band": vn1_score_by_group(f, actual, band, forecast_col="forecast"),
                    "demand_type": vn1_score_by_group(f, actual, dtype, forecast_col="forecast")}
        e = f.merge(actual, on=["unique_id", "ds"])
        by[name]["by_h"] = (e.assign(h=((e["ds"] - T).dt.days // 7), err=e["forecast"] - e["y"])
                            .groupby("h").apply(lambda d: (d["err"].abs().sum() + abs(d["err"].sum())) / d["y"].sum(),
                                                include_groups=False))
    # Diagnostic variants (same default params): why is `warehouse` so dominant?
    t = train["target"].dropna()
    big = train.loc[t.index[t > 50]]
    variants = []
    no_ids = [c for c in feats if c not in ("client", "warehouse")]
    for label, kw, fs in [("default (above)", {}, feats), ("without client/warehouse", {}, no_ids),
                          ("weight = scale", {"weight_by_scale": True}, feats),
                          ("weight = scale, without client/warehouse", {"weight_by_scale": True}, no_ids),
                          ("scaled target clipped at 20", {"clip_target": 20.0}, feats)]:
        if label.startswith("default"):
            s, top = vn1_score(fc, actual, forecast_col="forecast"), imp
        else:
            _, f_v, top = fit_predict(train, valid, fs, cfg["lgbm_smoke"], **kw)
            s = vn1_score(complete_with_zeros(f_v, actual), actual, forecast_col="forecast")
        variants.append({"variant": label, "score": round(s["score"], 4), "bias": round(s["bias"], 4),
                         "top feature": f"{top['feature'].iloc[0]} ({top['gain_share'].iloc[0]:.0%})"})
    target_stats = {"q50": t.quantile(.5), "q99": t.quantile(.99), "q999": t.quantile(.999), "max": t.max(),
                    "big_rows_share": len(big) / len(t), "big_target_share": big["target"].sum() / t.sum(),
                    "big_top_warehouse": big["warehouse"].astype(int).value_counts(normalize=True).head(1)}

    # Leak screen: rank correlation of each numeric feature with the scaled target (sample of train).
    sample = train[train["target"].notna()].sample(min(300_000, len(train)), random_state=0)
    num = [c for c in feats if c not in CATEGORICAL]
    corr = sample[num].corrwith(sample["target"], method="spearman").abs().sort_values(ascending=False)
    return {"lgbm": vn1_score(fc, actual, forecast_col="forecast"),
            "ma13": vn1_score(base, actual, forecast_col="forecast"),
            "importance": imp, "by": by, "corr": corr, "variants": pd.DataFrame(variants),
            "target_stats": target_stats, "n_actual_series": len(ids),
            "n_cold_series": int(band.eq("first sale after origin").sum()),
            "cold_units_share": float(actual[actual["unique_id"].isin(band.index[band.eq("first sale after origin")])]["y"].sum()
                                      / actual["y"].sum())}


# ----------------------------------------------------------------------------- report
def _table(df: pd.DataFrame) -> str:
    lines = ["| " + " | ".join(map(str, df.columns)) + " |", "|" + "---|" * df.shape[1]]
    lines += ["| " + " | ".join(str(v) for v in r) + " |" for r in df.itertuples(index=False)]
    return "\n".join(lines)


def write_features_report(X, info, smoke, split, cfg, reports_dir):
    specs = pd.DataFrame(feature_specs(cfg))
    specs["% missing"] = [f"{X[n].isna().mean():.1%}" for n in specs["name"]]
    specs["scaled"] = specs["scaled"].map({True: "yes", False: ""})
    o = split["train_origins"]
    lines = [
        "# Features: global direct multi-horizon model",
        "",
        "Built by `python -m src.run_features` from `data/processed/phase0_clean.parquet` only. "
        "No Phase 1 or Phase 2 file is read. One row per (series, origin T, horizon h); target = y at T + h.",
        "",
        "## Rules",
        "",
        "- Every feature uses weeks ≤ T only. Flags from the cleaned file (smoothed, stockout, price regimes) "
        "and series_table stats are **not** used: they were computed on all of Phase 0. They are recomputed at "
        "each origin from data ≤ T.",
        "- A series is included at origin T only if its first sale is ≤ T. Weeks before the first sale are "
        "missing (NaN), not zero.",
        f"- Scale = mean of the last {cfg['features']['scale_window']} weeks at T (fallback: mean since first "
        f"sale; floor {cfg['features']['scale_floor']}). Target and features marked *scaled* are divided by it; "
        "unscaled copies are kept as `<name>_raw` (not model inputs).",
        f"- Client 0 prices before {cfg['price_shift']['ignore_prices_before'].get(0)} are treated as missing.",
        "- The weekly-data switch date (2022-08-29) is used as a fixed calendar constant for `smoothed_share_*`.",
        "",
        "## Dataset",
        "",
        f"- Training origins: {len(o)}, every {cfg['splits']['origin_step_weeks']} weeks from {o[0].date()} to "
        f"{o[-1].date()} (the last origin whose 13th target is ≤ the validation origin).",
        f"- Validation origin: {split['valid_origin'].date()}, targets {split['valid_target_start'].date()} to "
        f"{split['valid_target_end'].date()} (the last 13 weeks of Phase 0).",
        f"- Rows: **{info['rows']:,}** ({info['rows_train']:,} train, {info['rows_valid']:,} valid), "
        f"{info['columns']} columns, {info['n_features']} model features. "
        f"{info['memory_mb']:,} MB in memory (float32), {info['file_mb']} MB on disk.",
        "",
        "## Feature list",
        "",
        "`% missing` is over all rows (train + valid).",
        "",
        _table(specs[["name", "group", "definition", "scaled", "% missing"]]),
        "",
    ]
    if smoke:
        lg, ma = smoke["lgbm"], smoke["ma13"]
        imp = smoke["importance"].head(20).assign(gain_share=lambda d: d["gain_share"].map(lambda v: f"{v:.1%}"))
        bands = ["top 1%", "top 1-10%", "rest 90%", "first sale after origin"]
        bt = pd.DataFrame({
            "units": smoke["by"]["lgbm"]["band"]["unit_share"],
            "lgbm score": smoke["by"]["lgbm"]["band"]["score"],
            "ma13 score": smoke["by"]["ma13"]["band"]["score"],
            "lgbm signed bias": smoke["by"]["lgbm"]["band"]["signed_bias"],
            "ma13 signed bias": smoke["by"]["ma13"]["band"]["signed_bias"],
            "lgbm error share": smoke["by"]["lgbm"]["band"]["abs_error_share"],
        }).reindex([b for b in bands if b in smoke["by"]["lgbm"]["band"].index])
        fmt = {"units": "{:.1%}", "lgbm score": "{:.3f}", "ma13 score": "{:.3f}", "lgbm signed bias": "{:+.1%}",
               "ma13 signed bias": "{:+.1%}", "lgbm error share": "{:.1%}"}
        bt = bt.apply(lambda c: c.map(lambda v, f=fmt[c.name]: f.format(v))).reset_index(names="volume band")
        dt = pd.DataFrame({"lgbm": smoke["by"]["lgbm"]["demand_type"]["score"],
                           "ma13": smoke["by"]["ma13"]["demand_type"]["score"],
                           "units": smoke["by"]["lgbm"]["demand_type"]["unit_share"]}).round(3)
        dt["units"] = dt["units"].map(lambda v: f"{v:.1%}")
        hh = pd.DataFrame({"h": smoke["by"]["lgbm"]["by_h"].index,
                           "lgbm": smoke["by"]["lgbm"]["by_h"].round(3).to_numpy(),
                           "ma13": smoke["by"]["ma13"]["by_h"].round(3).to_numpy()})
        corr = smoke["corr"].head(10).round(3).reset_index()
        corr.columns = ["feature", "|spearman| with scaled target"]
        lines += [
            "## Smoke test: LightGBM (tweedie, default params) vs moving_avg_13",
            "",
            "Trained on the training origins, predicted at the validation origin, unscaled, scored with "
            "`vn1_score` on the last 13 weeks of Phase 0. Default parameters, 100 trees: a pipeline check, not a tuned model.",
            "",
            "| model | score | accuracy | bias |",
            "|---|---|---|---|",
            f"| LightGBM | **{lg['score']:.4f}** | {lg['accuracy']:.4f} | {lg['bias']:.4f} |",
            f"| moving_avg_13 | {ma['score']:.4f} | {ma['accuracy']:.4f} | {ma['bias']:.4f} |",
            "",
            f"{smoke['n_cold_series']} series first sold after the validation origin; both models forecast 0 for "
            f"them ({smoke['cold_units_share']:.1%} of validation units).",
            "",
            "### Score by volume band (band from units in the 52 weeks up to the origin)",
            "",
            _table(bt),
            "",
            "### Score by demand type at origin",
            "",
            _table(dt.reset_index(names="demand_type")),
            "",
            "### Score by horizon",
            "",
            _table(hh),
            "",
            "### Top 20 features (gain)",
            "",
            _table(imp[["feature", "gain_share", "splits"]]),
            "",
            "### Why `warehouse` dominates: diagnostic variants (default params each)",
            "",
            f"Scaled-target distribution (train): median {smoke['target_stats']['q50']:.2f}, 99th pct "
            f"{smoke['target_stats']['q99']:.1f}, 99.9th pct {smoke['target_stats']['q999']:.0f}, max "
            f"{smoke['target_stats']['max']:.0f}. Rows with scaled target > 50 are "
            f"{smoke['target_stats']['big_rows_share']:.2%} of training rows but hold "
            f"{smoke['target_stats']['big_target_share']:.0%} of the summed scaled target"
            + (f"; {smoke['target_stats']['big_top_warehouse'].iloc[0]:.0%} of them are warehouse "
               f"{smoke['target_stats']['big_top_warehouse'].index[0]}" if len(smoke['target_stats']['big_top_warehouse']) else "")
            + f" (scale method: `{cfg['features']['scale_method']}`). They come from series with a tiny scale "
            "(quiet for 13 weeks) that then sell again. Unweighted tweedie on the scaled target chases those rows, "
            "and a high-cardinality categorical is the easiest way to isolate them. This is a loss/scale "
            "mismatch, not a time leak (warehouse is constant per series).",
            "",
            _table(smoke["variants"]),
            "",
            "### Leak screen: features most correlated with the scaled target (training rows)",
            "",
            _table(corr),
            "",
        ]
    (Path(reports_dir) / "features.md").write_text("\n".join(lines))


if __name__ == "__main__":
    build(get_config())

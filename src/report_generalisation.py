"""Tables for reports/generalisation.md: VN1 and M5 side by side, same metrics.   python -m src.report_generalisation

VN1: folds B, C, D (Phase 0 backtest), E = Phase 1 and P2 = Phase 2, all already scored and frozen; the
engine forecasts are the saved ones (outputs/intervals, outputs/phase2), baselines are recomputed from
the training data at each origin. Nothing here feeds a decision.
M5: outputs/m5/backtest and outputs/m5/test.
Writes outputs/m5/generalisation_tables.md and outputs/m5/generalisation.csv.
"""
import pandas as pd

from src.config import PROJECT_ROOT, get_config, load_config
from src.evaluation import complete_with_zeros, load_phase0, volume_band_at_origin
from src.experiment_m5 import BASELINES, CONFIG, FMT, paths, score_rows, table, baseline_set

WEEK = pd.Timedelta(weeks=1)


def vn1_rows() -> pd.DataFrame:
    cfg = get_config()
    proc = cfg["paths"]["processed_dir"]
    p0 = load_phase0(proc)
    folds = {k: cfg["backtest"]["folds"][k] for k in ("B", "C", "D")} | {"E": cfg["splits"]["phase1_forecast_origin"]}
    rows = []
    for f, T in folds.items():
        T = pd.Timestamp(T)
        iv = pd.read_parquet(PROJECT_ROOT / "outputs" / "intervals" / f"conformal__{f}.parquet")
        if f == "E":
            actual = pd.read_parquet(proc / "phase1_actuals.parquet")[["unique_id", "ds", "y"]]
        else:
            actual = p0[(p0["ds"] > T) & (p0["ds"] <= T + 13 * WEEK)][["unique_id", "ds", "y"]]
        hist = p0[p0["ds"] <= T]
        band = volume_band_at_origin(p0, T, actual["unique_id"].unique())
        fcs = {"engine": iv.rename(columns={"p50": "forecast"})[["unique_id", "ds", "forecast"]],
               **baseline_set(hist, T, cfg, actual)}
        for m, fc in fcs.items():
            for r in score_rows(fc, actual, band, iv if m == "engine" else None):
                rows.append({"dataset": "VN1", "fold": f, "origin": str(T.date()), "method": m, **r})
    # Phase 2: trained on Phase 0 + 1 re-cleaned; no intervals were produced for it
    T = pd.Timestamp("2024-01-01")
    hist = pd.read_parquet(proc / "phase2_run" / "train_clean.parquet",
                           columns=["unique_id", "client", "warehouse", "product", "ds", "y"])
    actual = pd.read_parquet(proc / "phase2_actuals.parquet")[["unique_id", "ds", "y"]]
    band = volume_band_at_origin(hist, T, actual["unique_id"].unique())
    fcs = {"engine": complete_with_zeros(pd.read_parquet(PROJECT_ROOT / "outputs" / "phase2" / "lgbm.parquet"), actual),
           **baseline_set(hist, T, cfg, actual)}
    for m, fc in fcs.items():
        for r in score_rows(fc, actual, band, None):
            rows.append({"dataset": "VN1", "fold": "P2", "origin": str(T.date()), "method": m, **r})
    return pd.DataFrame(rows)


def m5_rows() -> pd.DataFrame:
    p = paths(load_config(CONFIG))
    bt = pd.read_csv(p["backtest"] / "scores.csv").assign(dataset="M5")
    out = [bt]
    if (p["test"] / "scores.csv").exists():
        te = pd.read_csv(p["test"] / "scores.csv").assign(dataset="M5", fold="test",
                                                           origin=load_config(CONFIG)["max_allowed_date"])
        out.append(te)
    return pd.concat(out, ignore_index=True)


def main():
    res = pd.concat([vn1_rows(), m5_rows()], ignore_index=True)
    res.to_csv(PROJECT_ROOT / "outputs" / "m5" / "generalisation.csv", index=False)
    allb = res[res["band"] == "all"]
    by_fold = allb.pivot_table(index=["dataset", "fold", "origin"], columns="method", values="score",
                               sort=False).reset_index()
    by_fold = by_fold[["dataset", "fold", "origin", "engine", *[b for b in BASELINES if b in by_fold]]]
    backtest_folds = {"VN1": ["B", "C", "D"], "M5": ["F1", "F2", "F3", "F4"]}
    summ = []
    for ds, folds in backtest_folds.items():
        for m in ["engine", *BASELINES]:
            x = allb[(allb["dataset"] == ds) & allb["fold"].isin(folds) & (allb["method"] == m)]
            summ.append({"dataset": ds, "set": "backtest mean", "method": m, **x[["score", "wape", "bias", "coverage_80"]].mean()})
        for f in ({"VN1": ["E", "P2"], "M5": ["test"]}[ds]):
            for m in ["engine", *BASELINES]:
                x = allb[(allb["dataset"] == ds) & (allb["fold"] == f) & (allb["method"] == m)]
                if len(x):
                    summ.append({"dataset": ds, "set": f, "method": m, **x[["score", "wape", "bias", "coverage_80"]].iloc[0]})
    summ = pd.DataFrame(summ)
    band = (res[res["band"] != "all"].groupby(["dataset", "band", "method"], sort=False)
            [["score", "bias", "coverage_80"]].mean().reset_index())
    unit = (res[(res["band"] != "all") & (res["method"] == "engine")].groupby(["dataset", "band"], sort=False)["units"].sum()
            / res[(res["band"] == "all") & (res["method"] == "engine")].groupby("dataset")["units"].sum())
    band = band.merge(unit.rename("unit_share").reset_index(), on=["dataset", "band"])
    lines = ["## VN1 score by fold", "", table(by_fold), "", "## Summary", "", table(summ, FMT), "",
             "## By volume band (mean over every fold and test, incl. held-out)", "",
             table(band, FMT | {"unit_share": "{:.1%}"}), ""]
    (PROJECT_ROOT / "outputs" / "m5" / "generalisation_tables.md").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()

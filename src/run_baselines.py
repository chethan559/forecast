"""Baselines: train on Phase 0, score on Phase 1.   python -m src.run_baselines

Reads the cleaned files from data/processed/ (run `python -m src.run_cleaning` first).
Phase 2 is never read here.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import get_config
from src.metrics import vn1_score, vn1_score_by_group
from src.models.baselines import run_all

GROUP_ORDER = {
    "demand_type": ["smooth", "erratic", "intermittent", "lumpy"],
    "is_new": [False, True],
    "is_inactive": [False, True],
    "volume_band": ["top 1%", "top 1-10%", "rest 90%"],
}


def volume_bands(series: pd.DataFrame, cuts=(0.01, 0.10)) -> pd.Series:
    """Label series by training-volume rank: top 1%, next 9%, rest. Uses Phase 0 volume only."""
    n = len(series)
    r = series.set_index("unique_id")["volume_rank"]
    a, b = int(np.ceil(cuts[0] * n)), int(np.ceil(cuts[1] * n))
    return pd.Series(np.where(r <= a, "top 1%", np.where(r <= b, "top 1-10%", "rest 90%")), index=r.index)


def check_forecast(fc: pd.DataFrame, ids: pd.Index, dates: pd.DatetimeIndex, h: int) -> None:
    """Every series gets exactly h forecasts on the Phase 1 dates; no NaN, no negatives."""
    assert set(fc["unique_id"]) == set(ids), "forecast series != training series"
    assert fc.groupby("unique_id").size().eq(h).all(), f"not exactly {h} rows per series"
    assert set(fc["ds"]) == set(dates), "forecast dates != Phase 1 dates"
    assert not fc.duplicated(["unique_id", "ds"]).any()
    assert fc["forecast"].notna().all() and (fc["forecast"] >= 0).all()


def evaluate(cfg: dict, processed_dir=None, out_dir=None, reports_dir=None, write_report: bool = True):
    p = cfg["paths"]
    processed_dir = Path(processed_dir or p["processed_dir"])
    out_dir = Path(out_dir or cfg["baselines"]["output_dir"])
    reports_dir = Path(reports_dir or p["reports_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    h = cfg["baselines"]["h"]

    train = pd.read_parquet(processed_dir / "phase0_clean.parquet", columns=["unique_id", "ds", "y"])
    actual = pd.read_parquet(processed_dir / "phase1_actuals.parquet")
    series = pd.read_parquet(processed_dir / "series_table.parquet")
    p1_dates = pd.DatetimeIndex(sorted(actual["ds"].unique()))
    assert len(p1_dates) == h

    groups = {
        "demand_type": series.set_index("unique_id")["demand_type"],
        "is_new": series.set_index("unique_id")["is_new"],
        "is_inactive": series.set_index("unique_id")["is_inactive"],
        "volume_band": volume_bands(series, cfg["baselines"]["volume_bands"]),
    }

    forecasts = run_all(train, cfg)
    totals, by_group = [], []
    for name, fc in forecasts.items():
        check_forecast(fc, pd.Index(series["unique_id"]), p1_dates, h)
        fc.to_parquet(out_dir / f"{name}.parquet", index=False)
        s = vn1_score(fc, actual, forecast_col="forecast")
        err = fc.merge(actual, on=["unique_id", "ds"])
        totals.append({"method": name, **s,
                       "signed_bias": (err["forecast"] - err["y"]).sum() / err["y"].sum()})
        for dim, grp in groups.items():
            t = vn1_score_by_group(fc, actual, grp, forecast_col="forecast")
            by_group.append(t.reset_index().rename(columns={"group": "value"}).assign(method=name, dimension=dim))

    totals = pd.DataFrame(totals).sort_values("score").reset_index(drop=True)
    by_group = pd.concat(by_group, ignore_index=True)
    by_group["value"] = by_group["value"].astype(str)
    totals.to_csv(out_dir / "scores_total.csv", index=False)
    by_group.to_csv(out_dir / "scores_by_group.csv", index=False)

    if write_report:
        write_baseline_report(totals, by_group, forecasts, actual, series, groups, cfg, reports_dir)
    print(totals[["method", "score", "accuracy", "bias", "signed_bias"]].round(4).to_string(index=False))
    return totals, by_group


# ----------------------------------------------------------------------------- report
def _fmt_table(df: pd.DataFrame, floatfmt=".3f") -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(map(str, cols)) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        cells = [format(v, floatfmt) if isinstance(v, (float, np.floating)) else
                 (f"{v:,}" if isinstance(v, (int, np.integer)) else str(v)) for v in r]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def write_baseline_report(totals, by_group, forecasts, actual, series, groups, cfg, reports_dir):
    best = totals["method"].iloc[0]
    methods = list(totals["method"])
    pct = lambda v: f"{v:+.1%}"  # noqa: E731

    lines = [
        "# Baselines: Phase 0 → Phase 1",
        "",
        f"Train on Phase 0 (cleaned, {len(series):,} series), forecast the 13 Phase 1 weeks "
        f"({actual['ds'].min().date()} to {actual['ds'].max().date()}), score with the VN1 metric. "
        "Lower is better; all-zero = 2.0. Phase 2 is not used. Command: `python -m src.run_baselines`.",
        "",
        "`signed_bias` = sum(F − A) / sum(A): negative = under-forecast.",
        "",
        "## Overall",
        "",
        _fmt_table(totals[["method", "score", "accuracy", "bias", "signed_bias"]].assign(
            signed_bias=totals["signed_bias"].map(pct)), ".4f"),
        "",
        f"Winner: **{best}**.",
        "",
    ]
    if np.isclose(totals.set_index("method").loc["moving_avg_13", "score"],
                  totals.set_index("method").loc["moving_avg_13_inactive_zero", "score"]):
        lines += ["`moving_avg_13_inactive_zero` scores exactly the same as `moving_avg_13`: a series with no "
                  "sales in the last 13 weeks already has a 13-week mean of 0, so setting it to 0 changes nothing.", ""]

    labels = {"demand_type": "demand type", "is_new": "is_new (< 52 weeks of history)",
              "is_inactive": "is_inactive (no sales in last 13 training weeks)",
              "volume_band": "volume band (by Phase 0 units)"}
    lines += ["## Score by group", "",
              "VN1 score computed inside each group. The `units` and `error share` columns are for the winner "
              f"(`{best}`): share of Phase 1 units, and share of the total absolute error |F − A|.", ""]
    for dim, order in GROUP_ORDER.items():
        sub = by_group[by_group["dimension"].eq(dim)]
        wide = sub.pivot(index="value", columns="method", values="score")[methods]
        w = sub[sub["method"].eq(best)].set_index("value")
        wide.insert(0, "series", w["n_series"].astype(int))
        wide.insert(1, "units", w["unit_share"].map(lambda v: f"{v:.1%}"))
        wide.insert(2, "error share", w["abs_error_share"].map(lambda v: f"{v:.1%}"))
        wide.insert(3, f"signed bias ({best})", w["signed_bias"].map(pct))
        wide = wide.reindex([str(v) for v in order if str(v) in wide.index])
        lines += [f"### {labels[dim]}", "", _fmt_table(wide.reset_index().rename(columns={"value": dim})), ""]

    # Where the error comes from: cross of volume band x demand type, and concentration across series.
    fc = forecasts[best].merge(actual, on=["unique_id", "ds"])
    fc["abs_err"] = (fc["forecast"] - fc["y"]).abs()
    fc["err"] = fc["forecast"] - fc["y"]
    fc["band"] = fc["unique_id"].map(groups["volume_band"])
    fc["dtype"] = fc["unique_id"].map(groups["demand_type"])
    cross = (fc.pivot_table(index="band", columns="dtype", values="abs_err", aggfunc="sum")
             / fc["abs_err"].sum()).reindex(GROUP_ORDER["volume_band"])[GROUP_ORDER["demand_type"]]
    per_series = fc.groupby("unique_id")["abs_err"].sum().sort_values(ascending=False)
    cum = per_series.cumsum() / per_series.sum()
    top_n = lambda q: int(np.ceil(q * len(per_series)))  # noqa: E731
    by_week = fc.groupby("ds").agg(actual=("y", "sum"), forecast=("forecast", "sum"))
    by_week["signed_bias"] = (by_week["forecast"] - by_week["actual"]) / by_week["actual"]

    lines += [
        f"## Where the error comes from (`{best}`)",
        "",
        "Share of total absolute error by volume band × demand type:",
        "",
        _fmt_table(cross.map(lambda v: f"{v:.1%}").reset_index().rename(columns={"band": "volume band"})),
        "",
        f"Concentration across series: the worst 1% of series ({top_n(0.01)}) make "
        f"{cum.iloc[top_n(0.01) - 1]:.1%} of absolute error; the worst 10% make {cum.iloc[top_n(0.10) - 1]:.1%}.",
        "",
        "Total units per week, actual vs forecast:",
        "",
        _fmt_table(by_week.reset_index().assign(ds=lambda d: d["ds"].dt.date.astype(str),
                                                actual=lambda d: d["actual"].round().astype(int),
                                                forecast=lambda d: d["forecast"].round().astype(int),
                                                signed_bias=lambda d: d["signed_bias"].map(pct))),
        "",
    ]
    (Path(reports_dir) / "baselines.md").write_text("\n".join(lines))


if __name__ == "__main__":
    evaluate(get_config())

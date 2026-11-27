"""Prediction-interval experiment on VN1:   python -m src.experiment_intervals

Folds B, C, D (Phase 0 backtest) and E (= Phase 1: train on Phase 0, scored on Phase 1 actuals).
Two ways to get p10 / p90 around the frozen point forecast (p50):
  quantile_lgbm : LightGBM with a quantile objective (alpha 0.1 / 0.9), trained on the same rows
  conformal     : the model as it was 13 weeks before the origin, scored on those 13 weeks; quantiles
                  of its scaled residuals by volume band and horizon (no data after the origin)
Target: 80% of series-weeks inside [p10, p90]. Rule: the method whose mean coverage over B-E is
closest to 80%; if both are within 3 points of 80%, the one with the narrower intervals.
Writes reports/intervals.md and outputs/intervals/.
"""
import json
import time

import pandas as pd

from src.config import PROJECT_ROOT, get_config
from src.engine import INTERVAL_METHODS, evaluate_fold, interval_stats, prepare
from src.evaluation import load_phase0

OUT = PROJECT_ROOT / "outputs" / "intervals"
KNOWN_P50 = {"B": 0.4904291261066848, "C": 0.4478167419702745, "D": 0.47390878846894835, "E": 0.5711712612959301}
TOLERANCE = 0.03


def choose(summary: pd.DataFrame) -> tuple[str, str]:
    s = summary.set_index("method")
    gap = (s["coverage"] - 0.80).abs()
    close = gap[gap <= TOLERANCE].index
    if len(close) == 2:
        m = s.loc[close, "relative_width"].idxmin()
        return m, f"both within {TOLERANCE:.0%} of 80% coverage; narrower intervals"
    m = gap.idxmin()
    return m, "closest to 80% coverage"


def run(cfg: dict) -> dict:
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    processed = cfg["paths"]["processed_dir"]
    raw = load_phase0(processed).rename(columns={"price_raw": "price"})
    weekly = prepare(raw, cfg)
    folds = {k: v for k, v in cfg["backtest"]["folds"].items() if k in ("B", "C", "D")}
    folds["E"] = cfg["splits"]["phase1_forecast_origin"]
    p1 = pd.read_parquet(processed / "phase1_actuals.parquet")[["unique_id", "ds", "y"]]

    rows, bands = [], []
    for f, T in folds.items():
        r = evaluate_fold(weekly, T, cfg, actual=p1 if f == "E" else None, work_dir=processed,
                          methods=list(INTERVAL_METHODS), baseline=False)
        for m, s in r["methods"].items():
            rows.append({"fold": f, "method": m, "p50_score": r["score"]["score"],
                         **{k: s[k] for k in ("coverage", "below_p10", "above_p90", "relative_width")}})
            fc = s["forecast"]
            for b, ids in r["band"].groupby(r["band"]).groups.items():
                act = (p1 if f == "E" else weekly[weekly["ds"] > pd.Timestamp(T)])[["unique_id", "ds", "y"]]
                st = interval_stats(fc[fc["unique_id"].isin(ids)], act[act["unique_id"].isin(ids)])
                bands.append({"fold": f, "method": m, "band": b, **st})
            s["forecast"].to_parquet(OUT / f"{m}__{f}.parquet", index=False)
        print(f"fold {f}: p50 {r['score']['score']:.4f}; " + ", ".join(
            f"{m} cov {s['coverage']:.3f} width {s['relative_width']:.2f}" for m, s in r["methods"].items()), flush=True)

    res = pd.DataFrame(rows)
    by_band = pd.DataFrame(bands)
    summary = res.groupby("method")[["coverage", "below_p10", "above_p90", "relative_width"]].mean().reset_index()
    chosen, why = choose(summary)
    p50_check = {f: {"engine": float(res.loc[res["fold"] == f, "p50_score"].iloc[0]), "reported": KNOWN_P50[f]}
                 for f in folds}
    decision = {"chosen": chosen, "reason": why, "summary": summary.to_dict("records"), "p50_check": p50_check,
                "seconds": round(time.time() - t0)}
    res.to_csv(OUT / "by_fold.csv", index=False)
    by_band.to_csv(OUT / "by_band.csv", index=False)
    (OUT / "decision.json").write_text(json.dumps(decision, indent=2, default=float))
    write_report(res, by_band, summary, decision)
    print(json.dumps(decision, indent=2, default=float))
    return decision


def write_report(res, by_band, summary, d):
    def tbl(df, fmt):
        cols = list(df.columns)
        out = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
        for r in df.itertuples(index=False):
            out.append("| " + " | ".join(fmt.get(c, "{}").format(v) for c, v in zip(cols, r)) + " |")
        return "\n".join(out)

    pct = {"coverage": "{:.1%}", "below_p10": "{:.1%}", "above_p90": "{:.1%}", "relative_width": "{:.2f}",
           "p50_score": "{:.4f}"}
    bb = (by_band.groupby(["method", "band"])[["coverage", "relative_width"]].mean().reset_index()
          .sort_values(["band", "method"]))
    chk = pd.DataFrame([{"fold": f, "engine p50 score": v["engine"], "reported score": v["reported"],
                         "match": abs(v["engine"] - v["reported"]) < 1e-9} for f, v in d["p50_check"].items()])
    lines = [
        "# Prediction intervals (p10 / p90) on VN1",
        "",
        "`python -m src.experiment_intervals`. Folds B, C, D (Phase 0 backtest) and E = Phase 1. Target: 80% of "
        "series-weeks inside [p10, p90]. `relative_width` = Σ(p90 − p10) / Σ actual units (lower = narrower).",
        "",
        "- **quantile_lgbm**: two extra LightGBM models, quantile objective (alpha 0.1 / 0.9), same training rows.",
        "- **conformal**: the model trained 13 weeks before the origin, scored on those 13 weeks; 10th/90th percentiles "
        "of its scaled residuals by volume band × horizon (falls back to horizon, then overall, below 200 rows).",
        "",
        "## Coverage by fold",
        "",
        tbl(res[["fold", "method", "p50_score", "coverage", "below_p10", "above_p90", "relative_width"]], pct),
        "",
        "## Mean over folds B–E",
        "",
        tbl(summary, pct),
        "",
        f"**Chosen: `{d['chosen']}`** ({d['reason']}). Rule: closest to 80% coverage; if both are within "
        f"{TOLERANCE:.0%} of it, the narrower one.",
        "",
        "## Coverage by volume band (mean over folds)",
        "",
        tbl(bb, pct),
        "",
        "## Point forecast check",
        "",
        "The engine's p50 must equal the frozen model's reported scores:",
        "",
        tbl(chk, {"engine p50 score": "{:.10f}", "reported score": "{:.10f}"}),
        "",
    ]
    (PROJECT_ROOT / "reports" / "intervals.md").write_text("\n".join(lines))


if __name__ == "__main__":
    run(get_config())

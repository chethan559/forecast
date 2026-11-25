"""Round 3 (last before Phase 1), Phase 0 folds only:   python -m src.experiment_round3

1. Default config (uplift features, cap k=3) with level_shift.restart_after_shift off vs on.
   Keep it on only if the 4-fold average is not worse.
2. Bias multiplier, leave-one-fold-out: fit one factor on 3 folds (minimise their mean VN1 score),
   apply it to the 4th. Adopt only if the mean of the 4 held-out scores improves; then refit on all 4.
   Order of post-processing: model -> multiplier -> cap.
Writes reports/experiments_round3.md and outputs/round3/decisions.json. Does not edit the config.
"""
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from src.backtest import _table, lgbm_forecasts, training_origins
from src.config import get_config
from src.evaluation import actuals_after, complete_with_zeros, load_phase0
from src.feature_cache import build_cache

WEEK = pd.Timedelta(weeks=1)
GRID = np.round(np.arange(0.80, 1.2001, 0.01), 2)


def _cap_values(weekly, T, keys: pd.DataFrame, k: float, window: int = 52) -> np.ndarray:
    """k x max weekly sales in the last `window` weeks up to T, aligned to `keys` rows (same as postprocess.cap_forecasts)."""
    hist = weekly[(weekly["ds"] <= T) & (weekly["ds"] > T - window * WEEK)]
    cap = k * hist.groupby("unique_id")["y"].max()
    return keys["unique_id"].map(cap).fillna(0.0).to_numpy(float)


def _vn1(f: np.ndarray, a: np.ndarray) -> tuple[float, float]:
    e = f - a
    return (np.abs(e).sum() + abs(e.sum())) / a.sum(), e.sum() / a.sum()


def run(cfg: dict, processed_dir=None, out_dir=None, reports_dir=None) -> dict:
    processed_dir = Path(processed_dir or cfg["paths"]["processed_dir"])
    out_dir = Path(out_dir or cfg["paths"]["processed_dir"].parents[1] / "outputs" / "round3")
    reports_dir = Path(reports_dir or cfg["paths"]["reports_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    h = max(cfg["features"]["horizons"])
    k = cfg["lgbm"]["forecast_cap_k"]
    folds = {f: pd.Timestamp(o) for f, o in cfg["backtest"]["folds"].items()}
    weekly = load_phase0(processed_dir)
    start = cfg["backtest"]["origin_settings"][cfg["lgbm"]["origins_setting"]]["start"]
    plan = {f: training_origins(start, T, cfg) for f, T in folds.items()}
    needed = {o for v in plan.values() for o in v} | set(folds.values())

    # Aligned arrays per fold: actual y, cap value.
    data = {}
    for f, T in folds.items():
        a = actuals_after(weekly, T, h)
        data[f] = {"keys": a[["unique_id", "ds"]], "y": a["y"].to_numpy(float), "cap": _cap_values(weekly, T, a, k)}

    raw, dropped, scores = {}, {}, []
    for restart in (False, True):
        c = get_config({"level_shift": {"restart_after_shift": restart}})
        cache = build_cache(weekly, needed, c, processed_dir)
        for f, T in folds.items():
            r = lgbm_forecasts(cache, plan[f], T, c, c["features"]["scale_method"], [c["lgbm"]["use_ids"]], weekly)
            r = r[c["lgbm"]["use_ids"]]
            fc = complete_with_zeros(r["forecast"], data[f]["keys"].assign(y=0))
            raw[(restart, f)] = fc["forecast"].to_numpy(float)
            dropped[(restart, f)] = r["rows_dropped_for_shift"]
            s, b = _vn1(np.minimum(raw[(restart, f)], data[f]["cap"]), data[f]["y"])
            scores.append({"restart": restart, "fold": f, "score": s, "signed_bias": b,
                           "train_rows": r["train_rows"], "rows_dropped": r["rows_dropped_for_shift"]})
            print(f"restart={restart} fold {f}: {s:.4f} (bias {b:+.1%}, dropped {r['rows_dropped_for_shift']:,})", flush=True)
    scores = pd.DataFrame(scores)
    mean = scores.groupby("restart")["score"].mean()
    restart_on = bool(mean[True] <= mean[False])

    # Leave-one-fold-out bias multiplier on the chosen setting.
    S = pd.DataFrame({f: [_vn1(np.minimum(raw[(restart_on, f)] * m, data[f]["cap"]), data[f]["y"])[0] for m in GRID]
                      for f in folds}, index=GRID)
    lofo = []
    for f in folds:
        others = [g for g in folds if g != f]
        m_f = float(S[others].mean(axis=1).idxmin())
        lofo.append({"held_out_fold": f, "multiplier_fit_on_others": m_f,
                     "held_out_score_with": S.loc[m_f, f], "held_out_score_without": S.loc[1.0, f]})
    lofo = pd.DataFrame(lofo)
    adopt = bool(lofo["held_out_score_with"].mean() < lofo["held_out_score_without"].mean())
    m_all = float(S.mean(axis=1).idxmin())
    m_final = m_all if adopt else 1.0

    decisions = {"restart_after_shift": restart_on, "restart_mean_off": float(mean[False]),
                 "restart_mean_on": float(mean[True]), "bias_multiplier_adopted": adopt,
                 "bias_multiplier": m_final, "multiplier_fit_on_all_folds": m_all,
                 "lofo_mean_with": float(lofo["held_out_score_with"].mean()),
                 "lofo_mean_without": float(lofo["held_out_score_without"].mean()),
                 "final_backtest_scores": {f: float(S.loc[m_final, f]) for f in folds},
                 "final_backtest_mean": float(S.loc[m_final].mean()), "seconds": round(time.time() - t0)}
    (out_dir / "decisions.json").write_text(json.dumps(decisions, indent=2))
    scores.to_csv(out_dir / "restart_scores.csv", index=False)
    S.to_csv(out_dir / "multiplier_curve.csv")
    write_report(scores, lofo, S, decisions, cfg, reports_dir)
    print(json.dumps(decisions, indent=2))
    return decisions


def write_report(scores, lofo, S, d, cfg, reports_dir):
    folds = list(cfg["backtest"]["folds"])
    t = scores.pivot(index="restart", columns="fold", values="score")[folds]
    b = scores.pivot(index="restart", columns="fold", values="signed_bias")[folds]
    dr = scores.pivot(index="restart", columns="fold", values="rows_dropped")[folds]
    rows = []
    for r in (False, True):
        rows.append({"restart_after_shift": "on" if r else "off",
                     **{f: f"{t.loc[r, f]:.4f} ({b.loc[r, f]:+.1%})" for f in folds},
                     "average": f"{t.loc[r].mean():.4f}", "training rows dropped": " / ".join(f"{int(dr.loc[r, f]):,}" for f in folds)})
    curve = S.loc[[0.9, 0.95, 0.97, 0.98, 0.99, 1.0, 1.01, 1.02, 1.03, 1.05, 1.1]].copy()
    curve["mean"] = curve.mean(axis=1)
    curve = curve.reset_index(names="multiplier")
    lines = [
        "# Round 3: level-shift restart and bias multiplier",
        "",
        "`python -m src.experiment_round3`. Phase 0 folds A–D only. Config: default (full origins, max13_52, no ids, "
        f"uplift features), cap k={cfg['lgbm']['forecast_cap_k']:g}. Runtime {d['seconds']} s.",
        "",
        "## 1. restart_after_shift",
        "",
        "On: features use only weeks from the latest warehouse/client level-shift event; training rows whose "
        "(origin, target] window contains an event are dropped. Cell = score (signed bias).",
        "",
        _table(pd.DataFrame(rows)),
        "",
        f"Decision: **{'on' if d['restart_after_shift'] else 'off'}** (rule: on only if the average is not worse; "
        f"off {d['restart_mean_off']:.4f}, on {d['restart_mean_on']:.4f}).",
        "",
        "## 2. Bias multiplier (leave-one-fold-out)",
        "",
        "For each fold: fit the multiplier on the other three (grid 0.80–1.20, step 0.01, minimising their mean "
        "score), apply it to the held-out fold. Forecast × multiplier, then cap.",
        "",
        _table(lofo, {"multiplier_fit_on_others": "{:.2f}", "held_out_score_with": "{:.4f}",
                      "held_out_score_without": "{:.4f}"}),
        "",
        f"Mean held-out score: with {d['lofo_mean_with']:.4f}, without {d['lofo_mean_without']:.4f}. "
        f"Decision: **{'adopt' if d['bias_multiplier_adopted'] else 'do not adopt'}**"
        + (f", final multiplier fit on all four folds = **{d['bias_multiplier']:.2f}**." if d["bias_multiplier_adopted"]
           else f" (multiplier stays 1.0; the all-fold optimum would be {d['multiplier_fit_on_all_folds']:.2f})."),
        "",
        "Score by multiplier (all folds):",
        "",
        _table(curve, {"multiplier": "{:.2f}"}),
        "",
        "## Final backtest score of the frozen setup",
        "",
        " | ".join(f"{f}: {v:.4f}" for f, v in d["final_backtest_scores"].items()) + f" | **mean {d['final_backtest_mean']:.4f}**",
        "",
    ]
    (Path(reports_dir) / "experiments_round3.md").write_text("\n".join(lines))


if __name__ == "__main__":
    run(get_config())

"""Final selection round:   python -m src.experiment_final

Folds B, C, D (Phase 0 backtest) and E (= Phase 1: train on Phase 0, forecast 2023-10-09 .. 2024-01-01).
Fold A is reported for reference only (its prior-year holiday is the smoothed 2021 data).
Candidates vs frozen LightGBM: lgbm_w0.5, lgbm_w0.7 (blends with blend_0.25) and segment routing.
Rule: best mean over B-E, and it must beat frozen LightGBM on >= 3 of those 4 folds; else keep LightGBM.
Phase 2 is never read here. Writes reports/final_round.md and outputs/final_round/decision.json.
"""
import json
import time
from pathlib import Path

import pandas as pd

from src.backtest import _table
from src.config import PROJECT_ROOT, get_config
from src.evaluation import actuals_after, load_phase0
from src.final_model import CANDIDATES, forecast_candidates
from src.metrics import vn1_score

OUT = PROJECT_ROOT / "outputs" / "final_round"


def _score(fc, actual):
    s = vn1_score(fc, actual, forecast_col="forecast")
    e = fc.merge(actual, on=["unique_id", "ds"])
    return s["score"], (e["forecast"] - e["y"]).sum() / e["y"].sum()


def run(cfg: dict) -> dict:
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    processed = cfg["paths"]["processed_dir"]
    weekly = load_phase0(processed)
    h = max(cfg["features"]["horizons"])
    folds = {f: pd.Timestamp(o) for f, o in cfg["backtest"]["folds"].items()}
    folds["E"] = pd.Timestamp(cfg["splits"]["phase1_forecast_origin"])
    p1 = pd.read_parquet(processed / "phase1_actuals.parquet")[["unique_id", "ds", "y"]]

    rows, extra = [], {}
    for f, T in folds.items():
        c = forecast_candidates(weekly, T, cfg, processed)
        actual = p1 if f == "E" else actuals_after(weekly, T, h)
        for name, fc in c["forecasts"].items():
            s, b = _score(fc, actual)
            rows.append({"fold": f, "candidate": name, "score": s, "signed_bias": b})
            fc.to_parquet(OUT / f"{name}__{f}.parquet", index=False)
        extra[f] = {"n_routed": c["n_routed"], "train_rows": c["train_rows"]}
        print(f"fold {f}: " + ", ".join(f"{r['candidate']} {r['score']:.4f}" for r in rows if r["fold"] == f), flush=True)
        if f == "E":  # determinism check against the forecast scored once in the Phase 1 run
            saved = pd.read_parquet(PROJECT_ROOT / "outputs" / "phase1" / "lgbm_final.parquet")
            m = saved.merge(c["forecasts"]["lgbm"], on=["unique_id", "ds"], suffixes=("_saved", "_now"))
            extra[f]["max_abs_diff_vs_phase1_run"] = float((m["forecast_saved"] - m["forecast_now"]).abs().max())
    scores = pd.DataFrame(rows)
    scores.to_csv(OUT / "scores.csv", index=False)

    sel = cfg["final"]["selection_folds"]
    wide = scores.pivot(index="candidate", columns="fold", values="score")
    mean_sel = wide[sel].mean(axis=1)
    wins = {c: int((wide.loc[c, sel] < wide.loc["lgbm", sel]).sum()) for c in CANDIDATES if c != "lgbm"}
    eligible = [c for c in wins if wins[c] >= cfg["final"]["min_fold_wins"] and mean_sel[c] < mean_sel["lgbm"]]
    chosen = min(eligible, key=lambda c: mean_sel[c]) if eligible else "lgbm"
    best_by_mean = min([c for c in CANDIDATES if c != "lgbm"], key=lambda c: mean_sel[c])
    decision = {"chosen": chosen, "mean_B_E": {c: float(mean_sel[c]) for c in wide.index}, "wins_vs_lgbm": wins,
                "best_candidate_by_mean": best_by_mean, "eligible": eligible, "fold_info": extra,
                "seconds": round(time.time() - t0)}
    (OUT / "decision.json").write_text(json.dumps(decision, indent=2))
    write_report(scores, decision, cfg)
    print(json.dumps(decision, indent=2))
    return decision


def write_report(scores, d, cfg):
    sel = cfg["final"]["selection_folds"]
    order = ["lgbm", "lgbm_w0.5", "lgbm_w0.7", "routing", "blend_0.25"]
    folds = ["A"] + sel
    rows = []
    for c in order:
        x = scores[scores["candidate"] == c].set_index("fold")
        rows.append({"candidate": c + (" (frozen)" if c == "lgbm" else " (baseline)" if c == "blend_0.25" else ""),
                     **{f: f"{x.loc[f, 'score']:.4f} ({x.loc[f, 'signed_bias']:+.1%})" for f in folds},
                     "mean B–E": f"{d['mean_B_E'][c]:.4f}",
                     "beats frozen on": f"{d['wins_vs_lgbm'][c]}/4" if c in d["wins_vs_lgbm"] else "–"})
    routed = ", ".join(f"{f}: {v['n_routed']:,}" for f, v in d["fold_info"].items())
    lines = [
        "# Final selection round",
        "",
        "`python -m src.experiment_final`. Folds B, C, D (Phase 0 backtest) and E = Phase 1 (train on Phase 0, "
        "forecast 2023-10-09 → 2024-01-01). Fold A shown for reference only (prior-year holiday is smoothed 2021 data). "
        f"Phase 2 not read. Runtime {d['seconds']} s.",
        "",
        "Cell = VN1 score (signed bias). Candidates:",
        "",
        "- `lgbm_w0.5`, `lgbm_w0.7`: w × frozen LightGBM + (1 − w) × blend_0.25",
        "- `routing`: frozen LightGBM, except top-1% volume and lumpy series (segments from data ≤ origin) which get "
        f"0.5 × LightGBM + 0.5 × blend_0.25. Series routed per fold: {routed}.",
        "",
        _table(pd.DataFrame(rows)),
        "",
        f"Rule: lowest mean over {', '.join(sel)}, and beats frozen LightGBM on ≥ {cfg['final']['min_fold_wins']} of 4 folds.",
        f"Best candidate by mean: `{d['best_candidate_by_mean']}`. Eligible: {', '.join(f'`{e}`' for e in d['eligible']) or 'none'}. "
        f"**Chosen: `{d['chosen']}`.**",
        "",
        f"Determinism check: fold E LightGBM vs the forecast from the one-off Phase 1 run, max abs difference "
        f"{d['fold_info']['E'].get('max_abs_diff_vs_phase1_run', float('nan')):.2e}.",
        "",
    ]
    (Path(cfg["paths"]["reports_dir"]) / "final_round.md").write_text("\n".join(lines))


if __name__ == "__main__":
    run(get_config())

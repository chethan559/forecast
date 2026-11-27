"""Phase 2 (sealed test): run ONCE.   python -m src.run_phase2 --freeze   then   python -m src.run_phase2

1. Re-run cleaning on Phase 0 + Phase 1 combined (raw CSVs through the adapter; Phase 2 is not loaded).
2. Train the frozen config on it, forecast origin 2024-01-01, forecast 2024-01-08 .. 2024-04-01.
3. Write the forecasts to disk, and only then read Phase 2 actuals to score:
   the chosen method (config final.method), the frozen LightGBM, and blend_0.25.
Guards: outputs/phase2/frozen_config.json must match the current config + source; runs once.
"""
import argparse
import json

import pandas as pd

from src.adapters.vn1 import concat_phases, load_phase
from src.backtest import _table
from src.cleaning import run_cleaning
from src.config import PROJECT_ROOT, get_config
from src.final_model import forecast_candidates
from src.metrics import vn1_score, vn1_score_by_group
from src.run_phase1 import snapshot

OUT = PROJECT_ROOT / "outputs" / "phase2"
WORK = PROJECT_ROOT / "data" / "processed" / "phase2_run"
FREEZE_FILE = OUT / "frozen_config.json"
TRAIN_FILE = "train_clean.parquet"
LEADERBOARD_WINNER = 0.4637
WEEK = pd.Timedelta(weeks=1)


def freeze(cfg):
    OUT.mkdir(parents=True, exist_ok=True)
    if FREEZE_FILE.exists():
        raise SystemExit(f"{FREEZE_FILE} exists: already frozen.")
    FREEZE_FILE.write_text(json.dumps(snapshot(cfg), indent=1))
    print(f"frozen: {FREEZE_FILE}")


def check_frozen(cfg):
    if not FREEZE_FILE.exists():
        raise SystemExit("Not frozen. Run `python -m src.run_phase2 --freeze` first.")
    frozen, now = json.loads(FREEZE_FILE.read_text()), snapshot(cfg)
    if frozen["config"] != now["config"]:
        raise SystemExit("Config differs from the frozen one.")
    changed = sorted(k for k in set(frozen["source_sha256"]) | set(now["source_sha256"])
                     if frozen["source_sha256"].get(k) != now["source_sha256"].get(k))
    if changed:
        raise SystemExit(f"Source changed since the freeze: {changed}")


def build_training_data(cfg, work_dir=WORK) -> pd.DataFrame:
    """Phase 0 + Phase 1 from the raw CSVs, cleaned together with the frozen cleaning config."""
    f, raw = cfg["paths"]["files"], cfg["paths"]["raw_dir"]
    both = concat_phases(load_phase(raw, f["phase0_sales"], f["phase0_price"]),
                         load_phase(raw, f["phase1_sales"], f["phase1_price"]))
    clean, _ = run_cleaning(both, None, cfg)  # refuses anything after max_allowed_date (2024-01-01)
    work_dir.mkdir(parents=True, exist_ok=True)
    clean.to_parquet(work_dir / TRAIN_FILE, index=False)
    return pd.read_parquet(work_dir / TRAIN_FILE, columns=["unique_id", "client", "warehouse", "product", "ds", "y", "price_raw"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--freeze", action="store_true")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    cfg = get_config()
    if args.freeze:
        return freeze(cfg)
    check_frozen(cfg)
    if (OUT / "scores.json").exists() and not args.force:
        raise SystemExit("Phase 2 was already scored. It runs once.")

    weekly = build_training_data(cfg)
    T = weekly["ds"].max()
    assert T == pd.Timestamp("2024-01-01"), T
    c = forecast_candidates(weekly, T, cfg, WORK, TRAIN_FILE)
    methods = list(dict.fromkeys([cfg["final"]["method"], "lgbm", cfg["final"]["baseline"]]))
    for m in methods:  # on disk before any Phase 2 value is read
        c["forecasts"][m].to_parquet(OUT / f"{m}.parquet", index=False)

    actual = pd.read_parquet(cfg["paths"]["processed_dir"] / "phase2_actuals.parquet")[["unique_id", "ds", "y"]]
    assert set(actual["ds"]) == set(c["forecasts"]["lgbm"]["ds"]), "forecast weeks != Phase 2 weeks"
    total, groups, weekly_tot = {}, {}, actual.groupby("ds")["y"].sum().rename("actual").to_frame()
    for m in methods:
        f = c["forecasts"][m]
        s = vn1_score(f, actual, forecast_col="forecast")
        e = f.merge(actual, on=["unique_id", "ds"])
        total[m] = {**s, "signed_bias": float((e["forecast"] - e["y"]).sum() / e["y"].sum())}
        groups[m] = {dim: vn1_score_by_group(f, actual, c["segments"][dim], forecast_col="forecast")
                     for dim in ["volume_band", "demand_type"]}
        weekly_tot[m] = f.groupby("ds")["forecast"].sum()
    out = {"total": total, "methods": methods, "chosen": cfg["final"]["method"], "train_rows": c["train_rows"],
           "n_origins": len(c["origins"]), "rows_dropped": c["rows_dropped"], "n_routed": c["n_routed"]}
    (OUT / "scores.json").write_text(json.dumps(out, indent=2, default=float))
    write_report(out, groups, weekly_tot, c, cfg)
    print(json.dumps(total, indent=2, default=float))


def write_report(out, groups, wk, c, cfg):
    pct = "{:+.1%}"
    tot = pd.DataFrame([{"method": m + (" (chosen)" if m == out["chosen"] else ""), "score": v["score"],
                         "accuracy": v["accuracy"], "bias": v["bias"], "signed bias": v["signed_bias"],
                         "vs leaderboard winner": v["score"] - LEADERBOARD_WINNER}
                        for m, v in out["total"].items()])

    def grp(dim, order):
        cols = {}
        for m in out["methods"]:
            g = groups[m][dim]
            cols[f"{m} score"] = g["score"]
            cols[f"{m} bias"] = g["signed_bias"]
        t = pd.DataFrame({"units": groups[out["methods"][0]][dim]["unit_share"], **cols})
        t = t.reindex([o for o in order if o in t.index] + [o for o in t.index if o not in order])
        fmt = {"units": "{:.1%}"} | {k: ("{:.3f}" if k.endswith("score") else pct) for k in cols}
        return _table(t.reset_index(names=dim), fmt)

    w = wk.copy()
    for m in out["methods"]:
        w[f"{m} bias"] = (w[m] - w["actual"]) / w["actual"]
    w = w.reset_index(names="week")
    w["week"] = w["week"].dt.date.astype(str)
    lines = [
        "# Phase 2 result (sealed test, run once)",
        "",
        f"Cleaning re-run on Phase 0 + Phase 1 combined; trained on {out['n_origins']} origins "
        f"({c['origins'][0].date()} → {c['origins'][-1].date()}), {out['train_rows']:,} rows"
        + (f" ({out['rows_dropped']:,} dropped for crossing a level shift)" if out["rows_dropped"] else "")
        + ". Forecast origin 2024-01-01, forecasts 2024-01-08 → 2024-04-01. Config: `reports/final_config.md`. "
        "Forecasts: `outputs/phase2/`.",
        "",
        "## Total",
        "",
        _table(tot, {"score": "{:.4f}", "accuracy": "{:.4f}", "bias": "{:.4f}", "signed bias": pct,
                     "vs leaderboard winner": "{:+.4f}"}),
        "",
        f"VN1 Phase 2 leaderboard winner: **{LEADERBOARD_WINNER}**. Not a like-for-like comparison: the official "
        "scoring removed stockout weeks from the actuals, and we have no stockout data, so every zero week counts "
        "here. Our scores are therefore harsher than the leaderboard's would be for the same forecasts.",
        "",
        "## By volume band (at the forecast origin)",
        "",
        grp("volume_band", ["top 1%", "top 1-10%", "rest 90%", "first sale after origin"]),
        "",
        "## By demand type (at the forecast origin)",
        "",
        grp("demand_type", ["smooth", "erratic", "intermittent", "lumpy", "no_sales"]),
        "",
        "## Week by week (total units)",
        "",
        _table(w, {"actual": "{:,.0f}"} | {m: "{:,.0f}" for m in out["methods"]}
               | {f"{m} bias": pct for m in out["methods"]}),
        "",
    ]
    (PROJECT_ROOT / "reports" / "phase2.md").write_text("\n".join(lines))


if __name__ == "__main__":
    main()

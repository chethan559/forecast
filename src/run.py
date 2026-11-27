"""One command for any dataset:

    python -m src.run --config configs/<name>.yaml --stage clean|backtest|fit|predict

clean     adapter -> schema check -> cleaning flags; writes <processed_dir>/standard.parquet and
          clean.parquet, and <reports_dir>/data_summary.md (VN1: the original full cleaning outputs)
backtest  scores the folds in backtest.folds -> <reports_dir>/engine_backtest.md
fit       trains on all data -> model artifact in <artifacts_dir>/model_<last week>/
predict   loads the newest (or --model) artifact -> <outputs_dir>/forecast_<origin>.csv
"""
import argparse
import importlib
import json
from pathlib import Path

import pandas as pd

from src import engine
from src.config import load_config
from src.schema import summary, validate

ADAPTERS = {"vn1": "src.adapters.vn1", "m5": "src.adapters.m5", "shopify_orders": "src.adapters.shopify_orders",
            "long_csv": "src.adapters.long_csv"}


def load_standard(cfg: dict) -> pd.DataFrame:
    """Adapter output, validated against the schema."""
    name = cfg["dataset"]["adapter"]
    if name not in ADAPTERS:
        raise SystemExit(f"unknown adapter {name!r}; choose one of {sorted(ADAPTERS)}")
    return validate(importlib.import_module(ADAPTERS[name]).load(cfg), cfg, name=f"{name} adapter output")


def standard_data(cfg: dict) -> pd.DataFrame:
    """Read the cleaned-stage standard file if present, else run the adapter."""
    p = Path(cfg["paths"]["processed_dir"]) / "standard.parquet"
    return pd.read_parquet(p) if p.exists() else load_standard(cfg)


def stage_clean(cfg: dict) -> dict:
    from src.cleaning import run_cleaning
    out = Path(cfg["paths"]["processed_dir"])
    out.mkdir(parents=True, exist_ok=True)
    df = load_standard(cfg)
    df.to_parquet(out / "standard.parquet", index=False)
    if cfg["dataset"]["adapter"] == "vn1":  # keep the full VN1 cleaning outputs and report
        from src.run_cleaning import build_outputs
        build_outputs(cfg)
    weekly, series, diag = run_cleaning(df, None, cfg, return_diagnostics=True)
    weekly.to_parquet(out / "clean.parquet", index=False)
    series.to_parquet(out / "series_table.parquet" if cfg["dataset"]["adapter"] != "vn1" else out / "series_table_engine.parquet",
                      index=False)
    info = {**summary(df), "smoothing": {k: str(v) for k, v in diag["switch"].items()},
            "level_shift_group_events": len(diag["level_shift_events"]),
            "suspected_stockout_weeks": int(weekly["is_suspected_stockout"].sum()),
            "demand_types": series["demand_type"].value_counts().to_dict()}
    rep = Path(cfg["paths"]["reports_dir"])
    rep.mkdir(parents=True, exist_ok=True)
    lines = [f"# Data summary: {cfg['dataset']['name']}", "", "| item | value |", "|---|---|"]
    lines += [f"| {k} | {v} |" for k, v in info.items()]
    (rep / "data_summary.md").write_text("\n".join(lines) + "\n")
    return info


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--stage", required=True, choices=["clean", "backtest", "fit", "predict"])
    ap.add_argument("--origin", help="predict: forecast origin (default: last week in the data)")
    ap.add_argument("--horizon", type=int, default=None, help="predict: weeks ahead (default: all configured)")
    ap.add_argument("--model", help="predict: artifact directory (default: newest in artifacts_dir)")
    ap.add_argument("--interval-method", choices=list(engine.INTERVAL_METHODS))
    args = ap.parse_args(argv)
    cfg = load_config(args.config)

    if args.stage == "clean":
        print(json.dumps(stage_clean(cfg), indent=2, default=str))
    elif args.stage == "backtest":
        res = engine.backtest(standard_data(cfg), cfg)
        print(res["table"].to_string(index=False))
        print(f"report: {res['report']}")
    elif args.stage == "fit":
        print(f"model saved: {engine.fit(standard_data(cfg), cfg)}")
    elif args.stage == "predict":
        model = args.model
        if model is None:
            arts = sorted(Path(cfg["paths"]["artifacts_dir"]).glob("model_*"))
            if not arts:
                raise SystemExit("no model artifact found: run --stage fit first (or pass --model)")
            model = arts[-1]
        fc = engine.predict(model, standard_data(cfg), args.origin, args.horizon, args.interval_method)
        out = Path(cfg["paths"]["outputs_dir"])
        out.mkdir(parents=True, exist_ok=True)
        origin = (pd.Timestamp(args.origin) if args.origin else fc["ds"].min() - pd.Timedelta(weeks=1)).date()
        path = out / f"forecast_{origin}.csv"
        fc.to_csv(path, index=False)
        print(f"{len(fc):,} rows -> {path}")


if __name__ == "__main__":
    main()

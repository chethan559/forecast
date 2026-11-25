"""One command for the whole cleaning stage:  python -m src.run_cleaning

1. Load all phases through the VN1 adapter (Phase 2 only for the shape check + actuals export).
2. Clean Phase 0 (training) and Phase 0+1 (for the final model) with the same config.
3. Write parquet outputs and the data quality report.
"""
import argparse
import json
from pathlib import Path

import pandas as pd

from src.adapters.vn1 import concat_phases, load_phase
from src.cleaning import run_cleaning
from src.config import get_config


def check_same_series(*frames: pd.DataFrame) -> None:
    """All phases must contain exactly the same series (shape check; reads no values)."""
    ids = [set(f["unique_id"].unique()) for f in frames]
    assert all(i == ids[0] for i in ids), "phases do not share the same series"


def build_outputs(cfg: dict, raw_dir=None, out_dir=None, reports_dir=None, make_report: bool = True) -> dict:
    p = cfg["paths"]
    raw_dir = Path(raw_dir or p["raw_dir"])
    out_dir = Path(out_dir or p["processed_dir"])
    reports_dir = Path(reports_dir or p["reports_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    f = p["files"]

    phase0 = load_phase(raw_dir, f["phase0_sales"], f["phase0_price"])
    phase1 = load_phase(raw_dir, f["phase1_sales"], f["phase1_price"])
    phase2 = load_phase(raw_dir, f["phase2_sales"])  # held-out: never passed to cleaning
    check_same_series(phase0, phase1, phase2)
    for a, b in [(phase0, phase1), (phase1, phase2)]:
        assert b["ds"].min() - a["ds"].max() == pd.Timedelta(weeks=1), "phases are not back-to-back"

    clean0, series0, diag0 = run_cleaning(phase0, None, cfg, return_diagnostics=True)
    clean01, series01, diag01 = run_cleaning(concat_phases(phase0, phase1), None, cfg, return_diagnostics=True)

    actual_cols = ["unique_id", "client", "warehouse", "product", "ds", "y"]
    clean0.to_parquet(out_dir / "phase0_clean.parquet", index=False)
    clean01.to_parquet(out_dir / "phase01_clean.parquet", index=False)
    phase1[actual_cols].to_parquet(out_dir / "phase1_actuals.parquet", index=False)
    phase2[actual_cols].to_parquet(out_dir / "phase2_actuals.parquet", index=False)
    series0.to_parquet(out_dir / "series_table.parquet", index=False)
    series01.to_parquet(out_dir / "series_table_phase01.parquet", index=False)
    diag0["price_shift_events"].to_parquet(out_dir / "price_shift_events.parquet", index=False)
    diag0["level_shift_events"].to_parquet(out_dir / "level_shift_events.parquet", index=False)

    summary = {
        "n_series": int(series0["unique_id"].nunique()),
        "phase0_rows_after_trim": len(clean0),
        "phase01_rows_after_trim": len(clean01),
        "switch_date_phase0": str(diag0["switch"]["switch_date"].date()),
        "switch_date_phase01": str(diag01["switch"]["switch_date"].date()),
    }
    if make_report:
        from src.report import write_report
        write_report(clean0, series0, diag0, phase0, cfg, reports_dir)
    print(json.dumps(summary, indent=2))
    return summary


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--no-report", action="store_true", help="skip charts + markdown report")
    args = ap.parse_args()
    build_outputs(get_config(), make_report=not args.no_report)


if __name__ == "__main__":
    main()

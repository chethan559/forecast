"""Sealed test weeks: the last `seal.weeks` weeks of a dataset, split off right after the adapter.

Only the adapter step sees them (to split them off and check their shape). Cleaning, features,
backtest and every decision use the training weeks only (<= max_allowed_date, which engine.prepare
and run_cleaning both enforce). read_sealed() opens them only when the config has been frozen,
for the one-off test run.
"""
from pathlib import Path

import pandas as pd

WEEK = pd.Timedelta(weeks=1)
SEALED_FILE = "test_actuals.parquet"


class SealedError(RuntimeError):
    pass


def sealed_path(cfg: dict) -> Path:
    return Path(cfg["paths"]["processed_dir"]) / "sealed" / SEALED_FILE


def split_sealed(df: pd.DataFrame, cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(training weeks, sealed weeks). Uses dates only; checks them against the pinned config values."""
    s = cfg["seal"]
    weeks = sorted(df["ds"].unique())
    if len(weeks) <= s["weeks"]:
        raise SealedError(f"only {len(weeks)} weeks of data; cannot seal {s['weeks']}")
    cutoff = pd.Timestamp(weeks[-s["weeks"] - 1])
    expected = {"max_allowed_date": cutoff, "first_week": pd.Timestamp(weeks[-s["weeks"]]),
                "last_week": pd.Timestamp(weeks[-1])}
    pinned = {"max_allowed_date": cfg.get("max_allowed_date"), "first_week": s.get("first_week"),
              "last_week": s.get("last_week")}
    for k, v in pinned.items():
        if v is not None and pd.Timestamp(v) != expected[k]:
            raise SealedError(f"config {k}={v} but the data give {expected[k].date()}")
    return df[df["ds"] <= cutoff].reset_index(drop=True), df[df["ds"] > cutoff].reset_index(drop=True)


def write_sealed(sealed: pd.DataFrame, cfg: dict) -> Path:
    p = sealed_path(cfg)
    p.parent.mkdir(parents=True, exist_ok=True)
    sealed[["unique_id", "ds", "y"]].to_parquet(p, index=False)
    return p


def read_sealed(cfg: dict, freeze_file: Path) -> pd.DataFrame:
    """The sealed actuals. Refuses unless the frozen config snapshot exists."""
    if not Path(freeze_file).exists():
        raise SealedError(f"the sealed test weeks can only be read after the config is frozen ({freeze_file})")
    return pd.read_parquet(sealed_path(cfg))

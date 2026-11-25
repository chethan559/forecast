"""Shared helpers for scoring forecasts on Phase 0 windows (feature smoke test and backtest)."""
from pathlib import Path

import numpy as np
import pandas as pd

ALLOWED_INPUT = "phase0_clean.parquet"  # the only data file the modelling code may read
WEEK = pd.Timedelta(weeks=1)


def load_phase0(processed_dir) -> pd.DataFrame:
    path = Path(processed_dir) / ALLOWED_INPUT
    return pd.read_parquet(path, columns=["unique_id", "client", "warehouse", "product", "ds", "y", "price_raw"])


def actuals_after(weekly: pd.DataFrame, origin: pd.Timestamp, h: int) -> pd.DataFrame:
    """y for the h weeks after the origin (all series that have rows there)."""
    m = (weekly["ds"] > origin) & (weekly["ds"] <= origin + h * WEEK)
    return weekly.loc[m, ["unique_id", "ds", "y"]].reset_index(drop=True)


def complete_with_zeros(fc: pd.DataFrame, actual: pd.DataFrame) -> pd.DataFrame:
    """Series that had not sold by the origin can't be forecast; give them 0 so they still count."""
    out = actual[["unique_id", "ds"]].merge(fc, on=["unique_id", "ds"], how="left")
    out["forecast"] = out["forecast"].fillna(0.0)
    return out


def volume_band_at_origin(weekly: pd.DataFrame, origin: pd.Timestamp, ids) -> pd.Series:
    """top 1% / top 1-10% / rest, by units in the 52 weeks up to the origin (data <= origin only)."""
    w = weekly[(weekly["ds"] <= origin) & (weekly["ds"] > origin - 52 * WEEK)]
    units = w.groupby("unique_id")["y"].sum().reindex(ids).fillna(0)
    rank = units.rank(ascending=False, method="first")
    n = (units > 0).sum()
    band = np.where(rank <= np.ceil(0.01 * n), "top 1%", np.where(rank <= np.ceil(0.10 * n), "top 1-10%", "rest 90%"))
    band = pd.Series(band, index=units.index)
    started = weekly[weekly["ds"] <= origin]["unique_id"].unique()
    band[~band.index.isin(started)] = "first sale after origin"
    return band

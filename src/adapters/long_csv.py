"""Generic adapter: a CSV that is already one row per series and week (any column names).

adapter_args:
  file: path (relative to paths.raw_dir)
  columns: {unique_id: <col>, ds: <col>, y: <col>, price: <col>, in_stock: <col>, <hierarchy>: <col>}
  aggregate_to_monday: true   # optional: sum daily/any-date rows into Monday weeks and fill gaps with 0
"""
from pathlib import Path

import pandas as pd


def load(cfg: dict) -> pd.DataFrame:
    a = cfg["dataset"].get("adapter_args", {})
    path = Path(a["file"])
    path = path if path.is_absolute() else Path(cfg["paths"]["raw_dir"]) / path
    df = pd.read_csv(path)
    mapping = a.get("columns", {})
    df = df.rename(columns={v: k for k, v in mapping.items()})
    if a.get("aggregate_to_monday"):
        df["ds"] = pd.to_datetime(df["ds"])
        df["ds"] = df["ds"] - pd.to_timedelta(df["ds"].dt.day_of_week, unit="D")
        attrs = [c for c in cfg["dataset"]["hierarchy"] if c in df.columns]
        agg = {"y": ("y", "sum")} | ({"price": ("price", "mean")} if "price" in df else {})
        w = df.groupby(["unique_id", "ds"]).agg(**agg).reset_index()
        last = w["ds"].max()
        grid = pd.concat([pd.DataFrame({"unique_id": u, "ds": pd.date_range(s, last, freq="W-MON")})
                          for u, s in w.groupby("unique_id")["ds"].min().items()], ignore_index=True)
        df = grid.merge(w, on=["unique_id", "ds"], how="left").fillna({"y": 0.0})
        if attrs:
            df = df.merge(pd.read_csv(path).rename(columns={v: k for k, v in mapping.items()})
                          .groupby("unique_id")[attrs].first(), left_on="unique_id", right_index=True, how="left")
    return df

"""M5 (Walmart) adapter: daily wide CSVs -> standard weekly long frame (Monday weeks).

Files (Kaggle "M5 Forecasting - Accuracy"):
  sales_train_evaluation.csv  id, item_id, dept_id, cat_id, store_id, state_id, d_1 .. d_1941 (daily units)
  calendar.csv                date, wm_yr_wk, d, ... (maps d_N to a date and a Walmart week)
  sell_prices.csv             store_id, item_id, wm_yr_wk, sell_price (price per Walmart week, Sat..Fri)

Output (src/schema.py): unique_id = item_id + "_" + store_id, ds (Monday), y, price,
plus hierarchy columns state_id, store_id, cat_id, dept_id and item column item_id.
Only complete Monday-to-Sunday weeks are kept (M5 starts on a Saturday, so the first and last
partial weeks are dropped). Price = average daily price over the week's days that had a price
(each Walmart week contributes the number of its days that fall in the Monday week).
"""
from pathlib import Path

import numpy as np
import pandas as pd

ID_COLS = ["item_id", "dept_id", "cat_id", "store_id", "state_id"]


def _week_map(calendar: pd.DataFrame) -> pd.DataFrame:
    cal = calendar[["d", "date", "wm_yr_wk"]].copy()
    cal["date"] = pd.to_datetime(cal["date"])
    cal["monday"] = cal["date"] - pd.to_timedelta(cal["date"].dt.day_of_week, unit="D")
    return cal


def load_m5(raw_dir, sales_file="sales_train_evaluation.csv", calendar_file="calendar.csv",
            prices_file="sell_prices.csv", max_series: int | None = None) -> pd.DataFrame:
    raw_dir = Path(raw_dir)
    sales = pd.read_csv(raw_dir / sales_file, nrows=max_series)
    cal = _week_map(pd.read_csv(raw_dir / calendar_file))
    day_cols = [c for c in sales.columns if c.startswith("d_")]
    missing = [c for c in ID_COLS if c not in sales.columns]
    if missing:
        raise ValueError(f"{sales_file}: missing columns {missing}")
    if not day_cols:
        raise ValueError(f"{sales_file}: no daily columns d_1 .. d_N")

    # Daily -> weekly by summing column blocks (fast on the wide matrix; no 60M-row melt).
    day_week = cal.set_index("d")["monday"].reindex(day_cols)
    if day_week.isna().any():
        raise ValueError(f"{calendar_file} has no date for {int(day_week.isna().sum())} sales columns")
    n_days = day_week.value_counts()
    full_weeks = sorted(n_days.index[n_days == 7])  # drop partial first / last weeks
    X = sales[day_cols].to_numpy(dtype=float)
    weekly = np.stack([X[:, (day_week == w).to_numpy()].sum(axis=1) for w in full_weeks], axis=1)

    uid = sales["item_id"].astype(str) + "_" + sales["store_id"].astype(str)
    out = pd.DataFrame({
        "unique_id": np.repeat(uid.to_numpy(), len(full_weeks)),
        "ds": np.tile(np.array(full_weeks, dtype="datetime64[ns]"), len(sales)),
        "y": weekly.reshape(-1),
    })
    for c in ID_COLS:
        out[c] = np.repeat(sales[c].astype(str).to_numpy(), len(full_weeks))

    prices_path = raw_dir / prices_file
    if prices_path.exists():
        prices = pd.read_csv(prices_path)
        # Days of each Walmart week that fall into each Monday week (a Walmart week spans two).
        overlap = (cal[cal["monday"].isin(full_weeks)].groupby(["wm_yr_wk", "monday"]).size()
                   .rename("days").reset_index())
        p = prices.merge(overlap, on="wm_yr_wk")
        p["unique_id"] = p["item_id"].astype(str) + "_" + p["store_id"].astype(str)
        p["pd"] = p["sell_price"] * p["days"]
        wp = p.groupby(["unique_id", "monday"]).agg(pd_sum=("pd", "sum"), days=("days", "sum"))
        wp["price"] = wp["pd_sum"] / wp["days"]
        out = out.merge(wp["price"].reset_index().rename(columns={"monday": "ds"}), on=["unique_id", "ds"], how="left")
    else:
        out["price"] = np.nan
    return out.sort_values(["unique_id", "ds"], kind="stable").reset_index(drop=True)


def load(cfg: dict) -> pd.DataFrame:
    """Adapter entry point used by src.run (config dataset.adapter_args)."""
    a = cfg["dataset"].get("adapter_args", {})
    return load_m5(cfg["paths"]["raw_dir"], a.get("sales_file", "sales_train_evaluation.csv"),
                   a.get("calendar_file", "calendar.csv"), a.get("prices_file", "sell_prices.csv"),
                   a.get("max_series"))

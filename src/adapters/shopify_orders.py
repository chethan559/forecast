"""Shopify orders export (Admin > Orders > Export, CSV) -> standard weekly units per SKU.

The export has one row per line item. Order-level fields (Financial Status, Created at, Cancelled at,
...) are often filled only on an order's first row, so they are forward-filled within each order.

Rules (configurable through dataset.adapter_args):
  * Refunds excluded: orders whose Financial Status is in `exclude_financial_status`
    (default refunded, partially_refunded, voided) are dropped. The standard export does not say
    which line items of a partially refunded order were returned, so the whole order is dropped.
  * Cancelled orders (Cancelled at filled) are dropped.
  * Week = Monday of the order's local date (the date part of "Created at", already in shop time).
  * SKU = "Lineitem sku"; rows without a SKU fall back to "Lineitem name" (prefixed "name:").
  * Units = "Lineitem quantity"; price = quantity-weighted mean of "Lineitem price".
  * Every SKU gets one row per week from its first sale to the last complete week of the export
    (missing weeks = 0 units, the schema needs consecutive weeks). A partial final week is dropped.
  * Optional `by_location: true` makes one series per SKU x "Location" (hierarchy column `location`).
  * `vendor` (from "Vendor") is kept as a hierarchy column (constant per SKU: first seen value).

Output (src/schema.py): unique_id, ds, y, price, vendor [, location].
"""
from pathlib import Path

import numpy as np
import pandas as pd

ORDER_FIELDS = ["Financial Status", "Created at", "Cancelled at"]
DEFAULT_EXCLUDE = ["refunded", "partially_refunded", "voided"]
REQUIRED = ["Name", "Created at", "Lineitem quantity", "Lineitem name"]


def load_orders(path, exclude_financial_status=None, by_location: bool = False,
                drop_partial_last_week: bool = True) -> pd.DataFrame:
    df = pd.read_csv(path, dtype=str, keep_default_na=False, na_values=[""])
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"{path}: not a Shopify orders export, missing columns {missing}")
    for c in ORDER_FIELDS + ["Lineitem sku", "Lineitem price", "Vendor", "Location"]:
        if c not in df.columns:
            df[c] = np.nan
    df[ORDER_FIELDS] = df.groupby("Name", sort=False)[ORDER_FIELDS].ffill()

    excl = {s.lower() for s in (DEFAULT_EXCLUDE if exclude_financial_status is None else exclude_financial_status)}
    status = df["Financial Status"].fillna("").str.strip().str.lower()
    keep = ~status.isin(excl) & df["Cancelled at"].isna()
    stats = {"rows": len(df), "rows_refunded_or_voided": int(status.isin(excl).sum()),
             "rows_cancelled": int(df["Cancelled at"].notna().sum())}
    df = df[keep].copy()

    # Local order date = date part of "2024-01-05 10:22:11 +1100" (already in the shop's time zone).
    df["date"] = pd.to_datetime(df["Created at"].str.slice(0, 10), errors="coerce")
    if df["date"].isna().any():
        bad = df.loc[df["date"].isna(), "Created at"].head(3).tolist()
        raise ValueError(f"{path}: unreadable 'Created at' values, e.g. {bad}")
    df["ds"] = df["date"] - pd.to_timedelta(df["date"].dt.day_of_week, unit="D")
    df["qty"] = pd.to_numeric(df["Lineitem quantity"], errors="coerce").fillna(0)
    df["unit_price"] = pd.to_numeric(df["Lineitem price"], errors="coerce")
    sku = df["Lineitem sku"].fillna("").str.strip()
    df["unique_id"] = np.where(sku != "", sku, "name:" + df["Lineitem name"].fillna("").str.strip())
    if by_location:
        df["location"] = df["Location"].fillna("unknown").str.strip()
        df["unique_id"] = df["unique_id"] + "@" + df["location"]
    df["vendor"] = df["Vendor"].fillna("unknown").str.strip()

    df["pq"] = (df["unit_price"] * df["qty"]).fillna(0.0)
    df["q_priced"] = df["qty"].where(df["unit_price"].notna(), 0.0)  # units that had a price
    weekly = df.groupby(["unique_id", "ds"]).agg(y=("qty", "sum"), pq=("pq", "sum"), q_priced=("q_priced", "sum"))
    weekly["price"] = (weekly["pq"] / weekly["q_priced"].replace(0, np.nan)).where(weekly["y"] > 0)
    weekly = weekly[["y", "price"]].reset_index()

    last_day = df["date"].max()
    last_week = last_day - pd.Timedelta(days=last_day.day_of_week)
    if drop_partial_last_week and last_day.day_of_week != 6:  # export stopped before Sunday
        last_week -= pd.Timedelta(weeks=1)
    first = weekly.groupby("unique_id")["ds"].min()
    first = first[first <= last_week]
    grid = pd.concat([pd.DataFrame({"unique_id": u, "ds": pd.date_range(f, last_week, freq="W-MON")})
                      for u, f in first.items()], ignore_index=True)
    out = grid.merge(weekly, on=["unique_id", "ds"], how="left")
    out["y"] = out["y"].fillna(0.0)
    attrs = df.groupby("unique_id")[["vendor"] + (["location"] if by_location else [])].first()
    out = out.merge(attrs, left_on="unique_id", right_index=True, how="left")
    out.attrs["adapter_stats"] = {**stats, "rows_kept": int(keep.sum()), "series": int(out["unique_id"].nunique()),
                                  "last_complete_week": str(last_week.date())}
    return out.sort_values(["unique_id", "ds"], kind="stable").reset_index(drop=True)


def load(cfg: dict) -> pd.DataFrame:
    """Adapter entry point used by src.run (config dataset.adapter_args)."""
    a = cfg["dataset"].get("adapter_args", {})
    path = Path(a.get("orders_file", Path(cfg["paths"]["raw_dir"]) / "orders_export.csv"))
    if not path.is_absolute():
        path = Path(cfg["paths"]["raw_dir"]) / path
    return load_orders(path, a.get("exclude_financial_status"), a.get("by_location", False),
                       a.get("drop_partial_last_week", True))

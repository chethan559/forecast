"""Standard input schema: what every adapter must produce and every engine call accepts.

One row per series and week:
    unique_id  (required)  series key, e.g. "sku123_store4"
    ds         (required)  week start date; must be a Monday
    y          (required)  units sold that week; numeric, >= 0, no missing values
    price      (optional)  average selling price that week (missing = no sale / unknown)
    in_stock   (optional)  True/False: was the item available? (used instead of inferring stockouts)
    <hierarchy columns>    optional grouping columns listed in config dataset.hierarchy, constant per series

Weeks must be consecutive within each series (adapters fill missing weeks with 0 sales), so every
rolling window in cleaning and features counts real weeks.
"""
import numpy as np
import pandas as pd

REQUIRED = ["unique_id", "ds", "y"]
OPTIONAL = ["price", "in_stock"]
_TRUE = {"true", "1", "yes", "y", "t"}
_FALSE = {"false", "0", "no", "n", "f"}


class SchemaError(ValueError):
    """Input data does not match the standard schema. The message says what and where."""


def _examples(values, n=3) -> str:
    vals = list(pd.unique(pd.Series(values)))[:n]
    return ", ".join(repr(v) for v in vals)


def _to_bool(s: pd.Series, name: str) -> pd.Series:
    if pd.api.types.is_bool_dtype(s):
        return s
    if s.isna().any():
        raise SchemaError(f"{name}: {int(s.isna().sum())} missing values in `in_stock`; use True/False for every row")
    txt = s.astype(str).str.strip().str.lower()
    bad = ~txt.isin(_TRUE | _FALSE)
    if bad.any():
        raise SchemaError(f"{name}: `in_stock` must be True/False (or 1/0, yes/no); found {_examples(s[bad])}")
    return txt.isin(_TRUE)


def validate(df: pd.DataFrame, cfg: dict | None = None, name: str = "input") -> pd.DataFrame:
    """Check `df` against the schema and return a clean, sorted copy. Raises SchemaError on problems."""
    if not isinstance(df, pd.DataFrame):
        raise SchemaError(f"{name}: expected a pandas DataFrame, got {type(df).__name__}")
    hierarchy = list((cfg or {}).get("dataset", {}).get("hierarchy", []))
    item = (cfg or {}).get("dataset", {}).get("item_column")
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise SchemaError(f"{name}: missing required column(s) {missing}. Columns present: {list(df.columns)}")
    missing_h = [c for c in hierarchy + ([item] if item else []) if c not in df.columns]
    if missing_h:
        raise SchemaError(f"{name}: config dataset.hierarchy / item_column lists {missing_h}, "
                          f"but the data has no such column(s). Columns present: {list(df.columns)}")
    if df.empty:
        raise SchemaError(f"{name}: no rows")

    out = df.copy()
    if out["unique_id"].isna().any():
        raise SchemaError(f"{name}: {int(out['unique_id'].isna().sum())} rows have no unique_id")
    out["unique_id"] = out["unique_id"].astype(str)

    ds = pd.to_datetime(out["ds"], errors="coerce")
    if ds.isna().any():
        raise SchemaError(f"{name}: `ds` has values that are not dates: {_examples(out.loc[ds.isna(), 'ds'])}")
    ds = ds.dt.tz_localize(None) if getattr(ds.dt, "tz", None) is not None else ds
    not_monday = ds.dt.day_of_week != 0
    if not_monday.any():
        raise SchemaError(f"{name}: `ds` must be the Monday that starts each week; {int(not_monday.sum())} rows are "
                          f"not Mondays, e.g. {_examples(ds[not_monday].dt.date)}. Aggregate to Monday weeks first.")
    out["ds"] = ds.dt.normalize()

    y = pd.to_numeric(out["y"], errors="coerce")
    if y.isna().any():
        bad = out.loc[y.isna(), "y"]
        raise SchemaError(f"{name}: `y` must be numeric with no missing values; {len(bad)} bad rows, e.g. {_examples(bad)}")
    if (y < 0).any():
        raise SchemaError(f"{name}: `y` has {int((y < 0).sum())} negative values (e.g. {_examples(y[y < 0])}); "
                          "returns/refunds must be removed by the adapter")
    out["y"] = y.astype(float)

    if "price" in out.columns:
        p = pd.to_numeric(out["price"], errors="coerce")
        bad = p.isna() & out["price"].notna()
        if bad.any():
            raise SchemaError(f"{name}: `price` has non-numeric values, e.g. {_examples(out.loc[bad, 'price'])}")
        if (p < 0).any():
            raise SchemaError(f"{name}: `price` has {int((p < 0).sum())} negative values")
        out["price"] = p.astype(float)
    if "in_stock" in out.columns:
        out["in_stock"] = _to_bool(out["in_stock"], name)

    dup = out.duplicated(["unique_id", "ds"])
    if dup.any():
        ex = out.loc[dup, ["unique_id", "ds"]].head(3).to_dict("records")
        raise SchemaError(f"{name}: {int(dup.sum())} duplicate (unique_id, ds) rows, e.g. {ex}. One row per series per week.")

    for col in hierarchy + ([item] if item else []):
        n = out.groupby("unique_id")[col].nunique(dropna=False)
        if (n > 1).any():
            raise SchemaError(f"{name}: hierarchy column `{col}` changes within a series "
                              f"(e.g. {_examples(n.index[n > 1])}); it must be constant per unique_id")

    out = out.sort_values(["unique_id", "ds"], kind="stable").reset_index(drop=True)
    gap = out.groupby("unique_id")["ds"].diff().dropna() != pd.Timedelta(weeks=1)
    if gap.any():
        ids = out.loc[gap[gap].index, "unique_id"]
        raise SchemaError(f"{name}: weeks are not consecutive in {ids.nunique()} series (e.g. {_examples(ids)}). "
                          "Fill missing weeks with y = 0 so every series has one row per week.")

    front = REQUIRED + hierarchy + ([item] if item and item not in hierarchy else []) + [c for c in OPTIONAL if c in out]
    return out[front + [c for c in out.columns if c not in front]]


def summary(df: pd.DataFrame) -> dict:
    """Short description of a validated frame (for logs and reports)."""
    return {"rows": len(df), "series": int(df["unique_id"].nunique()),
            "first_week": str(df["ds"].min().date()), "last_week": str(df["ds"].max().date()),
            "weeks": int(df["ds"].nunique()), "total_units": float(df["y"].sum()),
            "zero_share": float(np.mean(df["y"] == 0)),
            "has_price": "price" in df.columns, "has_in_stock": "in_stock" in df.columns}

"""VN1 adapter: wide CSV (one column per week) -> long DataFrame in the pipeline's standard schema.

Standard schema (every adapter must produce this; cleaning only relies on it):
    unique_id, client, warehouse, product, ds (datetime, Monday), y   (sales)
    unique_id, client, warehouse, product, ds, price                  (price)
"""
from pathlib import Path

import pandas as pd

ID_COLS = ["Client", "Warehouse", "Product"]
KEY = ["unique_id", "ds"]


def _wide_to_long(path: str | Path, value_name: str) -> pd.DataFrame:
    wide = pd.read_csv(path)
    missing = [c for c in ID_COLS if c not in wide.columns]
    assert not missing, f"{path}: missing id columns {missing}"
    week_cols = [c for c in wide.columns if c not in ID_COLS]
    n_series, n_weeks = len(wide), len(week_cols)

    long = wide.melt(id_vars=ID_COLS, value_vars=week_cols, var_name="ds", value_name=value_name)
    long = long.rename(columns={"Client": "client", "Warehouse": "warehouse", "Product": "product"})
    long["ds"] = pd.to_datetime(long["ds"], format="%Y-%m-%d")
    long["unique_id"] = (
        long["client"].astype(str) + "_" + long["warehouse"].astype(str) + "_" + long["product"].astype(str)
    )
    long[value_name] = long[value_name].astype("float64")
    long = long[["unique_id", "client", "warehouse", "product", "ds", value_name]]
    long = long.sort_values(KEY, kind="stable").reset_index(drop=True)

    _validate(long, n_series, n_weeks, path)
    return long


def _validate(long: pd.DataFrame, n_series: int, n_weeks: int, path) -> None:
    assert long["unique_id"].nunique() == n_series, f"{path}: duplicate series ids"
    assert len(long) == n_series * n_weeks, f"{path}: rows != series x weeks"
    assert not long.duplicated(KEY).any(), f"{path}: duplicate (unique_id, ds)"
    weeks = pd.DatetimeIndex(long["ds"].drop_duplicates().sort_values())
    assert (weeks.day_of_week == 0).all(), f"{path}: not all dates are Mondays"
    if len(weeks) > 1:
        assert (weeks.to_series().diff().dropna() == pd.Timedelta(days=7)).all(), f"{path}: weeks not consecutive"


def load_vn1_sales(path: str | Path) -> pd.DataFrame:
    return _wide_to_long(path, "y")


def load_vn1_price(path: str | Path) -> pd.DataFrame:
    return _wide_to_long(path, "price")


def merge_sales_price(sales: pd.DataFrame, price: pd.DataFrame | None) -> pd.DataFrame:
    """Left-join price onto sales for one phase. Phase 2 has no price file -> price stays NaN."""
    if price is None:
        out = sales.copy()
        out["price"] = float("nan")
        return out
    out = sales.merge(price[KEY + ["price"]], on=KEY, how="left", validate="one_to_one")
    assert len(out) == len(sales)
    return out


def load_phase(raw_dir: str | Path, sales_file: str, price_file: str | None = None) -> pd.DataFrame:
    raw_dir = Path(raw_dir)
    sales = load_vn1_sales(raw_dir / sales_file)
    price = load_vn1_price(raw_dir / price_file) if price_file else None
    return merge_sales_price(sales, price)


def concat_phases(*phases: pd.DataFrame) -> pd.DataFrame:
    """Stack consecutive phases and re-check the combined panel is still a clean weekly grid."""
    out = pd.concat(phases, ignore_index=True).sort_values(KEY, kind="stable").reset_index(drop=True)
    n_series = out["unique_id"].nunique()
    _validate(out, n_series, out["ds"].nunique(), "concat")
    return out


def load(cfg: dict, phases=("phase0",)) -> pd.DataFrame:
    """Adapter entry point used by src.run: the configured phases (default: Phase 0 = training data)."""
    f, raw = cfg["paths"]["files"], cfg["paths"]["raw_dir"]
    phases = cfg["dataset"].get("adapter_args", {}).get("phases", list(phases))
    frames = [load_phase(raw, f[f"{p}_sales"], f.get(f"{p}_price")) for p in phases]
    return frames[0] if len(frames) == 1 else concat_phases(*frames)

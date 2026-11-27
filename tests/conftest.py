import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
RAW = ROOT / "data" / "raw" / "vn1"


def make_panel(series: dict, start="2023-01-02", price: dict | None = None) -> pd.DataFrame:
    """Build a long sales(+price) frame from {uid: [y...]} for synthetic tests."""
    rows = []
    for uid, ys in series.items():
        ds = pd.date_range(start, periods=len(ys), freq="W-MON")
        ps = (price or {}).get(uid, [10.0 if y > 0 else float("nan") for y in ys])
        c, w, p = uid.split("_")
        for d, y, pr in zip(ds, ys, ps):
            rows.append({"unique_id": uid, "client": int(c), "warehouse": int(w), "product": int(p),
                         "ds": d, "y": float(y), "price": pr})
    return pd.DataFrame(rows)


@pytest.fixture(scope="session")
def raw_dir():
    if not (RAW / "Phase 0 - Sales.csv").exists():
        pytest.skip("raw VN1 files not present")
    return RAW


@pytest.fixture(scope="session")
def small_raw(tmp_path_factory, raw_dir):
    """First 80 series of every raw file, written to a temp dir (fast end-to-end tests)."""
    d = tmp_path_factory.mktemp("raw_small")
    for f in raw_dir.glob("*.csv"):
        pd.read_csv(f, nrows=80).to_csv(d / f.name, index=False)
    return d


def synthetic_weekly(n_series=30, n_weeks=150, start="2021-01-04", hierarchy=("store",), seed=0,
                     with_in_stock=False) -> pd.DataFrame:
    """Standard-schema weekly data: seasonal Poisson demand, staggered starts, a price column."""
    import numpy as np
    rng = np.random.default_rng(seed)
    weeks = pd.date_range(start, periods=n_weeks, freq="W-MON")
    rows = []
    for i in range(n_series):
        base, first = rng.uniform(2, 40), int(rng.integers(0, 40))
        for j, d in enumerate(weeks):
            y = float(rng.poisson(base * (1 + 0.4 * np.sin(2 * np.pi * j / 52)))) if j >= first else 0.0
            r = {"unique_id": f"s{i:02d}", "ds": d, "y": y, "price": 10.0 + i % 5 if y > 0 else np.nan}
            if "store" in hierarchy:
                r["store"] = f"st{i % 3}"
            if with_in_stock:
                r["in_stock"] = bool(rng.random() > 0.05)
            rows.append(r)
    return pd.DataFrame(rows)


def synthetic_engine_config(tmp_path, **overrides):
    """Generic config (configs/defaults.yaml) with all paths inside tmp_path."""
    from src.config import load_config
    base = {"dataset": {"name": "synthetic", "hierarchy": ["store"], "group_level": "store"},
            "paths": {d: str(tmp_path / d) for d in ("raw_dir", "processed_dir", "reports_dir", "outputs_dir", "artifacts_dir")},
            "level_shift": {"group_keys": ["store"]}, "backtest": {"folds": {"X": "2023-06-26"}}}
    for k, v in overrides.items():
        base.setdefault(k, {}).update(v) if isinstance(v, dict) else base.__setitem__(k, v)
    return load_config(ROOT / "configs" / "defaults.yaml", base)


def write_shopify_export(path, start="2022-01-03", days=600, seed=0):
    """A synthetic Shopify orders export in the standard CSV layout (one row per line item;
    order-level fields only on the first row of each order, as Shopify exports them)."""
    import numpy as np
    rng = np.random.default_rng(seed)
    skus = [("TEE-BLK", "Black tee", 35.0, "Acme"), ("TEE-WHT", "White tee", 35.0, "Acme"),
            ("CAP-01", "Cap", 25.0, "Acme"), ("MUG-01", "Mug", 18.0, "HomeCo"), ("", "Gift card", 50.0, "Acme"),
            ("SOCK-3", "Socks 3-pack", 15.0, "HomeCo")]
    rows, n = [], 1000
    for day in pd.date_range(start, periods=days, freq="D"):
        for _ in range(int(rng.poisson(6 * (1.5 if day.month in (11, 12) else 1.0)))):
            n += 1
            status = rng.choice(["paid", "paid", "paid", "paid", "refunded", "partially_refunded", "voided", "pending"])
            cancelled = f"{day.date()} 18:00:00 +1100" if rng.random() < 0.03 else ""
            items = rng.choice(len(skus), size=int(rng.integers(1, 3)), replace=False)
            for k, idx in enumerate(items):
                sku, name, price, vendor = skus[idx]
                rows.append({"Name": f"#{n}", "Email": "", "Financial Status": status if k == 0 else "",
                             "Created at": f"{day.date()} 10:{k:02d}:00 +1100" if k == 0 else "",
                             "Cancelled at": cancelled if k == 0 else "", "Lineitem quantity": int(rng.integers(1, 4)),
                             "Lineitem name": name, "Lineitem price": price, "Lineitem sku": sku, "Vendor": vendor})
    pd.DataFrame(rows).to_csv(path, index=False)
    return path

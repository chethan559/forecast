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

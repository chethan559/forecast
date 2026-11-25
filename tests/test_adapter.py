import numpy as np
import pandas as pd
import pytest

from src.adapters.vn1 import concat_phases, load_phase, load_vn1_price, load_vn1_sales, merge_sales_price


def test_round_trip_matches_wide_csv(small_raw):
    path = small_raw / "Phase 0 - Sales.csv"
    wide = pd.read_csv(path)
    long = load_vn1_sales(path)
    assert list(long.columns) == ["unique_id", "client", "warehouse", "product", "ds", "y"]
    assert len(long) == len(wide) * (wide.shape[1] - 3)
    back = long.pivot(index=["client", "warehouse", "product"], columns="ds", values="y")
    back.columns = back.columns.strftime("%Y-%m-%d")
    orig = wide.set_index(["Client", "Warehouse", "Product"])
    back = back.loc[list(orig.index), list(orig.columns)]
    np.testing.assert_array_equal(back.to_numpy(), orig.to_numpy(dtype=float))


def test_unique_id_format_and_mondays(small_raw):
    long = load_vn1_price(small_raw / "Phase 1 - Price.csv")
    r = long.iloc[0]
    assert r["unique_id"] == f"{r['client']}_{r['warehouse']}_{r['product']}"
    assert (long["ds"].dt.day_of_week == 0).all()
    assert not long.duplicated(["unique_id", "ds"]).any()


def test_rejects_gap_in_weeks(tmp_path):
    pd.DataFrame({"Client": [0], "Warehouse": [1], "Product": [2],
                  "2023-01-02": [1.0], "2023-01-16": [2.0]}).to_csv(tmp_path / "s.csv", index=False)
    with pytest.raises(AssertionError, match="consecutive"):
        load_vn1_sales(tmp_path / "s.csv")


def test_rejects_non_monday(tmp_path):
    pd.DataFrame({"Client": [0], "Warehouse": [1], "Product": [2],
                  "2023-01-03": [1.0]}).to_csv(tmp_path / "s.csv", index=False)
    with pytest.raises(AssertionError, match="Mondays"):
        load_vn1_sales(tmp_path / "s.csv")


def test_rejects_duplicate_series(tmp_path):
    pd.DataFrame({"Client": [0, 0], "Warehouse": [1, 1], "Product": [2, 2],
                  "2023-01-02": [1.0, 2.0]}).to_csv(tmp_path / "s.csv", index=False)
    with pytest.raises(AssertionError):
        load_vn1_sales(tmp_path / "s.csv")


def test_merge_and_concat(small_raw):
    p0 = load_phase(small_raw, "Phase 0 - Sales.csv", "Phase 0 - Price.csv")
    p1 = load_phase(small_raw, "Phase 1 - Sales.csv", "Phase 1 - Price.csv")
    both = concat_phases(p0, p1)
    assert both["ds"].nunique() == 170 + 13
    assert len(both) == 80 * 183
    no_price = merge_sales_price(load_vn1_sales(small_raw / "Phase 2 - Sales.csv"), None)
    assert no_price["price"].isna().all()

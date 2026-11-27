import numpy as np
import pandas as pd
import pytest

from src.adapters import long_csv, m5, shopify_orders
from src.config import load_config
from src.schema import validate
from tests.conftest import ROOT, write_shopify_export


# ---------------------------------------------------------------- M5
@pytest.fixture
def m5_dir(tmp_path):
    days = pd.date_range("2011-01-29", periods=23, freq="D")  # starts on a Saturday, like M5
    cal = pd.DataFrame({"date": days.strftime("%Y-%m-%d"), "d": [f"d_{i + 1}" for i in range(23)],
                        "wm_yr_wk": [11101 + (i // 7) for i in range(23)]})  # Walmart weeks Sat..Fri
    cal.to_csv(tmp_path / "calendar.csv", index=False)
    sales = pd.DataFrame({"id": ["A_S1_e", "A_S2_e", "B_S1_e"], "item_id": ["A", "A", "B"], "dept_id": ["D1"] * 3,
                          "cat_id": ["C"] * 3, "store_id": ["S1", "S2", "S1"], "state_id": ["CA"] * 3})
    for i in range(23):
        sales[f"d_{i + 1}"] = [1, 2, 0 if i < 9 else 3]
    sales.to_csv(tmp_path / "sales_train_evaluation.csv", index=False)
    pd.DataFrame({"store_id": ["S1", "S1", "S1"], "item_id": ["A", "A", "A"], "wm_yr_wk": [11101, 11102, 11103],
                  "sell_price": [7.0, 14.0, 14.0]}).to_csv(tmp_path / "sell_prices.csv", index=False)
    return tmp_path


def test_m5_daily_to_monday_weeks(m5_dir):
    df = m5.load_m5(m5_dir)
    weeks = sorted(df["ds"].unique())
    # 2011-01-29 (Sat) .. 2011-02-20 (Sun): full Monday weeks are 01-31, 02-07, 02-14 (first 2 days dropped)
    assert [str(pd.Timestamp(w).date()) for w in weeks] == ["2011-01-31", "2011-02-07", "2011-02-14"]
    a1 = df[df["unique_id"] == "A_S1"].set_index("ds")
    assert (a1["y"] == 7).all()
    # Week of 01-31: Mon-Fri in Walmart week 11101 (7.0), Sat-Sun in 11102 (14.0) -> (5*7 + 2*14) / 7 = 9
    assert a1.loc["2011-01-31", "price"] == pytest.approx(9.0)
    assert df.loc[df["unique_id"] == "A_S2", "price"].isna().all()  # no price rows for that store
    b = df[df["unique_id"] == "B_S1"].set_index("ds")["y"]
    assert b.loc["2011-01-31"] == 0 and b.loc["2011-02-07"] == 3 * 7  # sales start on d_10 = Mon 2011-02-07


def test_m5_output_validates_with_its_config(m5_dir):
    cfg = load_config(ROOT / "configs" / "m5.yaml", {"paths": {"raw_dir": str(m5_dir)}})
    df = m5.load(cfg)
    out = validate(df, cfg)
    assert {"state_id", "store_id", "cat_id", "dept_id", "item_id", "price"} <= set(out.columns)


# ---------------------------------------------------------------- Shopify
def _export(tmp_path):
    rows = [  # order-level fields only on the first line of each order, as in Shopify exports
        {"Name": "#1", "Financial Status": "paid", "Created at": "2024-01-01 09:00:00 +1100", "Cancelled at": "",
         "Lineitem quantity": 2, "Lineitem name": "Tee", "Lineitem price": 30, "Lineitem sku": "TEE", "Vendor": "Acme"},
        {"Name": "#1", "Financial Status": "", "Created at": "", "Cancelled at": "",
         "Lineitem quantity": 1, "Lineitem name": "Cap", "Lineitem price": 20, "Lineitem sku": "CAP", "Vendor": "Acme"},
        {"Name": "#2", "Financial Status": "refunded", "Created at": "2024-01-02 09:00:00 +1100", "Cancelled at": "",
         "Lineitem quantity": 5, "Lineitem name": "Tee", "Lineitem price": 30, "Lineitem sku": "TEE", "Vendor": "Acme"},
        {"Name": "#3", "Financial Status": "paid", "Created at": "2024-01-03 09:00:00 +1100",
         "Cancelled at": "2024-01-03 10:00:00 +1100", "Lineitem quantity": 7, "Lineitem name": "Tee",
         "Lineitem price": 30, "Lineitem sku": "TEE", "Vendor": "Acme"},
        {"Name": "#4", "Financial Status": "partially_refunded", "Created at": "2024-01-04 09:00:00 +1100",
         "Cancelled at": "", "Lineitem quantity": 9, "Lineitem name": "Tee", "Lineitem price": 30,
         "Lineitem sku": "TEE", "Vendor": "Acme"},
        {"Name": "#5", "Financial Status": "paid", "Created at": "2024-01-15 23:30:00 +1100", "Cancelled at": "",
         "Lineitem quantity": 1, "Lineitem name": "Tee", "Lineitem price": 24, "Lineitem sku": "TEE", "Vendor": "Acme"},
        {"Name": "#5", "Financial Status": "", "Created at": "", "Cancelled at": "",
         "Lineitem quantity": 3, "Lineitem name": "Gift card", "Lineitem price": 50, "Lineitem sku": "", "Vendor": "Acme"},
        {"Name": "#6", "Financial Status": "paid", "Created at": "2024-01-28 12:00:00 +1100", "Cancelled at": "",
         "Lineitem quantity": 1, "Lineitem name": "Tee", "Lineitem price": 30, "Lineitem sku": "TEE", "Vendor": "Acme"},
        {"Name": "#7", "Financial Status": "voided", "Created at": "2024-01-30 12:00:00 +1100", "Cancelled at": "",
         "Lineitem quantity": 4, "Lineitem name": "Cap", "Lineitem price": 20, "Lineitem sku": "CAP", "Vendor": "Acme"},
    ]
    p = tmp_path / "orders_export.csv"
    pd.DataFrame(rows).to_csv(p, index=False)
    return p


def test_shopify_orders_to_weekly_units(tmp_path):
    df = shopify_orders.load_orders(_export(tmp_path))
    tee = df[df["unique_id"] == "TEE"].set_index("ds")
    # week of 2024-01-01: order #1 only (refunded #2, cancelled #3, partially refunded #4 excluded)
    assert tee.loc["2024-01-01", "y"] == 2
    assert tee.loc["2024-01-08", "y"] == 0           # gap filled with 0
    assert tee.loc["2024-01-15", "y"] == 1 and tee.loc["2024-01-15", "price"] == 24
    assert tee.loc["2024-01-22", "y"] == 1           # Sunday 2024-01-28 belongs to the week of 01-22
    # last row is 2024-01-30 (Tuesday): that partial week is dropped
    assert df["ds"].max() == pd.Timestamp("2024-01-22")
    assert "name:Gift card" in set(df["unique_id"])  # no SKU -> line item name
    assert df.loc[df["unique_id"] == "CAP", "y"].sum() == 1  # voided order excluded
    assert df.attrs["adapter_stats"]["rows_cancelled"] == 1


def test_shopify_keep_partial_refunds_and_by_location(tmp_path):
    df = shopify_orders.load_orders(_export(tmp_path), exclude_financial_status=["refunded", "voided"])
    assert df.loc[(df["unique_id"] == "TEE") & (df["ds"] == "2024-01-01"), "y"].item() == 2 + 9


def test_shopify_output_validates_with_example_config(tmp_path):
    write_shopify_export(tmp_path / "orders_export.csv", days=120)
    cfg = load_config(ROOT / "configs" / "shopify_example.yaml", {"paths": {"raw_dir": str(tmp_path)}})
    out = validate(shopify_orders.load(cfg), cfg)
    assert out["unique_id"].nunique() == 6 and "vendor" in out


def test_not_a_shopify_export(tmp_path):
    pd.DataFrame({"a": [1]}).to_csv(tmp_path / "x.csv", index=False)
    with pytest.raises(ValueError, match="not a Shopify orders export"):
        shopify_orders.load_orders(tmp_path / "x.csv")


# ---------------------------------------------------------------- generic long CSV
def test_long_csv_daily_to_weekly(tmp_path):
    days = pd.date_range("2024-01-01", periods=21, freq="D")
    pd.DataFrame({"sku": "A", "date": days, "units": 1.0, "shop": "x"}).to_csv(tmp_path / "d.csv", index=False)
    cfg = load_config(ROOT / "configs" / "defaults.yaml", {
        "dataset": {"hierarchy": ["store"], "adapter_args": {
            "file": "d.csv", "aggregate_to_monday": True,
            "columns": {"unique_id": "sku", "ds": "date", "y": "units", "store": "shop"}}},
        "paths": {"raw_dir": str(tmp_path)}})
    out = validate(long_csv.load(cfg), cfg)
    assert out["y"].tolist() == [7.0, 7.0, 7.0] and set(out["store"]) == {"x"}

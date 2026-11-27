import pandas as pd
import pytest

from src.config import get_config
from src.schema import SchemaError, summary, validate
from tests.conftest import synthetic_weekly

CFG = {"dataset": {"hierarchy": ["store"], "item_column": None}}


def test_valid_frame_passes_and_is_sorted():
    df = synthetic_weekly(5, 20).sample(frac=1, random_state=0)
    out = validate(df, CFG)
    assert list(out.columns[:4]) == ["unique_id", "ds", "y", "store"]
    assert out.equals(out.sort_values(["unique_id", "ds"]).reset_index(drop=True))
    assert summary(out)["series"] == 5


@pytest.mark.parametrize("mutate, message", [
    (lambda d: d.drop(columns="y"), "missing required column"),
    (lambda d: d.assign(ds=d["ds"] + pd.Timedelta(days=1)), "Monday"),
    (lambda d: d.assign(y=d["y"] - 1000), "negative"),
    (lambda d: d.assign(y=d["y"].where(d.index != 3)), "no missing values"),
    (lambda d: d.assign(ds=d["ds"].where(d.index != 3, "not a date")), "not dates"),
    (lambda d: pd.concat([d, d.iloc[[0]]]), "duplicate"),
    (lambda d: d.drop(index=[5]), "Fill missing weeks"),
    (lambda d: d.drop(columns="store"), "hierarchy"),
    (lambda d: d.assign(store=d.index.astype(str)), "changes within a series"),
    (lambda d: d.assign(price="abc"), "non-numeric"),
    (lambda d: d.assign(in_stock="maybe"), "in_stock"),
])
def test_clear_errors(mutate, message):
    df = synthetic_weekly(3, 12)
    with pytest.raises(SchemaError, match=message):
        validate(mutate(df), CFG)


def test_in_stock_is_parsed_to_bool():
    df = synthetic_weekly(2, 6).assign(in_stock=["yes", "no", "1", "0", "True", "false"] * 2)
    out = validate(df, CFG)
    assert out["in_stock"].dtype == bool and out["in_stock"].sum() == 6


def test_vn1_adapter_output_matches_schema(small_raw):
    from src.adapters.vn1 import load_phase
    df = load_phase(small_raw, "Phase 0 - Sales.csv", "Phase 0 - Price.csv")
    out = validate(df, get_config())
    assert {"client", "warehouse", "product", "price"} <= set(out.columns)

"""End-to-end: Phase 2 values must have zero influence on anything the cleaning stage writes."""
import shutil

import numpy as np
import pandas as pd

from src.config import get_config
from src.run_cleaning import build_outputs

CLEANING_OUTPUTS = ["phase0_clean.parquet", "phase01_clean.parquet", "series_table.parquet",
                    "series_table_phase01.parquet", "price_shift_events.parquet", "phase1_actuals.parquet"]


def test_phase2_values_do_not_affect_cleaning(small_raw, tmp_path):
    alt_raw = tmp_path / "raw_alt"
    shutil.copytree(small_raw, alt_raw)
    p2 = pd.read_csv(alt_raw / "Phase 2 - Sales.csv")
    rng = np.random.default_rng(0)
    p2.iloc[:, 3:] = rng.integers(0, 1000, size=p2.iloc[:, 3:].shape)  # scramble held-out values
    p2.to_csv(alt_raw / "Phase 2 - Sales.csv", index=False)

    cfg = get_config()
    build_outputs(cfg, raw_dir=small_raw, out_dir=tmp_path / "a", make_report=False)
    build_outputs(cfg, raw_dir=alt_raw, out_dir=tmp_path / "b", make_report=False)
    for name in CLEANING_OUTPUTS:
        pd.testing.assert_frame_equal(pd.read_parquet(tmp_path / "a" / name), pd.read_parquet(tmp_path / "b" / name))
    # sanity: the test really changed Phase 2
    assert not pd.read_parquet(tmp_path / "a" / "phase2_actuals.parquet").equals(
        pd.read_parquet(tmp_path / "b" / "phase2_actuals.parquet"))


def test_cleaned_outputs_stop_before_phase2(small_raw, tmp_path):
    build_outputs(get_config(), raw_dir=small_raw, out_dir=tmp_path, make_report=False)
    p2_start = pd.read_parquet(tmp_path / "phase2_actuals.parquet")["ds"].min()
    for name in ["phase0_clean.parquet", "phase01_clean.parquet"]:
        assert pd.read_parquet(tmp_path / name)["ds"].max() < p2_start
    st = pd.read_parquet(tmp_path / "series_table_phase01.parquet")
    assert st["last_sale_date"].max() < p2_start

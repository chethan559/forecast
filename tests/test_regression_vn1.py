"""VN1 must stay reproducible: the final config scores exactly what was reported.

Phase 1 (train on Phase 0, origin 2023-10-02) and Phase 2 (re-clean Phase 0 + 1, origin 2024-01-01).
Slow (~5 min): run with  pytest -m slow
"""
import pandas as pd
import pytest

from src.config import get_config
from src.evaluation import load_phase0
from src.final_model import forecast_candidates
from src.metrics import vn1_score
from tests.conftest import ROOT

PROCESSED = ROOT / "data" / "processed"
EXPECTED = {  # reported in experiments.md / reports/phase1.md / reports/phase2.md
    "phase1": {"lgbm": 0.5711712612959301, "blend_0.25": 0.5815564957185151},
    "phase2": {"lgbm": 0.48788770583004526, "blend_0.25": 0.784429304979525},
}
pytestmark = pytest.mark.slow


def _need(*files):
    for f in files:
        if not (PROCESSED / f).exists():
            pytest.skip(f"{f} missing: run python -m src.run_cleaning first")


def _score(fc, actual):
    return vn1_score(fc, actual, forecast_col="forecast")["score"]


def test_phase1_scores_unchanged():
    _need("phase0_clean.parquet", "phase1_actuals.parquet")
    cfg = get_config()
    c = forecast_candidates(load_phase0(PROCESSED), "2023-10-02", cfg, PROCESSED)
    actual = pd.read_parquet(PROCESSED / "phase1_actuals.parquet")
    for name, expected in EXPECTED["phase1"].items():
        assert _score(c["forecasts"][name], actual) == pytest.approx(expected, abs=1e-9), name


def test_phase2_scores_unchanged(tmp_path):
    _need("phase2_actuals.parquet")
    from src.run_phase2 import TRAIN_FILE, build_training_data
    cfg = get_config()
    weekly = build_training_data(cfg, work_dir=tmp_path)
    c = forecast_candidates(weekly, "2024-01-01", cfg, tmp_path, TRAIN_FILE)
    actual = pd.read_parquet(PROCESSED / "phase2_actuals.parquet")
    for name, expected in EXPECTED["phase2"].items():
        assert _score(c["forecasts"][name], actual) == pytest.approx(expected, abs=1e-9), name

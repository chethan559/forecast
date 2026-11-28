from pathlib import Path

import pytest

from src.config import get_config, load_config
from tests.conftest import ROOT


def test_vn1_yaml_holds_the_vn1_decisions():
    c = get_config()
    assert c["dataset"]["hierarchy"] == ["client", "warehouse"] and c["dataset"]["calendar"] == "us"
    assert c["smoothing"]["switch_date"] == "2022-08-29"
    assert c["price_rules"]["ignore"] == [{"column": "client", "value": 0, "before": "2022-08-29"}]
    assert c["level_shift"]["restart_after_shift"] and c["lgbm"]["forecast_cap_k"] == 3.0
    assert c["features"]["optional_features"] == ["uplift"] and not c["lgbm"]["use_ids"]
    assert isinstance(c["paths"]["processed_dir"], Path) and c["paths"]["processed_dir"].is_absolute()


def test_defaults_and_overrides():
    c = load_config(ROOT / "configs" / "m5.yaml", {"features": {"scale_floor": 0.5}})
    assert c["features"]["scale_floor"] == 0.5 and c["features"]["scale_method"] == "max13_52"  # from defaults
    assert c["smoothing"]["enabled"] is True and c["smoothing"]["switch_date"] == "2014-03-17"
    assert c["level_shift"]["group_keys"] == c["dataset"]["hierarchy"]  # generic default


def test_group_keys_default_to_hierarchy():
    c = load_config(ROOT / "configs" / "defaults.yaml", {"dataset": {"hierarchy": ["store"]}})
    assert c["level_shift"]["group_keys"] == ["store"]


@pytest.mark.parametrize("override, msg", [
    ({"dataset": {"hierarchy": ["store"], "group_level": "region"}}, "group_level"),
    ({"dataset": {"calendar": "uk"}}, "calendar"),
    ({"dataset": {"hierarchy": ["store"]}, "level_shift": {"group_keys": ["region"]}}, "group_keys"),
    ({"price_rules": {"ignore": [{"column": "client"}]}}, "missing"),
])
def test_invalid_configs(override, msg):
    with pytest.raises(ValueError, match=msg):
        load_config(ROOT / "configs" / "defaults.yaml", override)


def test_every_shipped_config_loads():
    for p in (ROOT / "configs").glob("*.yaml"):
        load_config(p)

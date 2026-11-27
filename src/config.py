"""Configuration: configs/defaults.yaml + configs/<dataset>.yaml, merged into one plain dict.

Every threshold, date and dataset-specific rule lives in YAML so the same code runs on any dataset.
    load_config("configs/m5.yaml")          # any dataset
    get_config()                            # VN1 (configs/vn1.yaml), kept for existing code and tests
    get_config({"lgbm": {"use_ids": True}}) # nested overrides on top

Keys ending in `_dir` and `paths.*` strings are resolved relative to the project root into Paths.
"""
from copy import deepcopy
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = PROJECT_ROOT / "configs"
DEFAULT_DATASET = CONFIG_DIR / "vn1.yaml"


def _merge(base: dict, upd: dict) -> dict:
    for k, v in upd.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _merge(base[k], v)
        else:
            base[k] = deepcopy(v)
    return base


def _resolve_paths(node, key=""):
    """Turn path-like strings into absolute Paths (relative to the project root)."""
    if isinstance(node, dict):
        return {k: (_resolve_paths(v, k) if k != "files" else v) for k, v in node.items()}
    if isinstance(node, str) and key.endswith("_dir"):
        p = Path(node)
        return p if p.is_absolute() else PROJECT_ROOT / p
    return node


def _read(path) -> dict:
    with open(path) as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"{path}: expected a mapping at the top level")
    return data


def load_config(path=DEFAULT_DATASET, overrides: dict | None = None) -> dict:
    """defaults.yaml <- dataset yaml <- overrides. Returns a fresh dict (safe to mutate)."""
    path = Path(path)
    if not path.is_absolute() and not path.exists():
        path = PROJECT_ROOT / path
    cfg = _merge(_read(CONFIG_DIR / "defaults.yaml"), _read(path))
    if overrides:
        _merge(cfg, overrides)
    cfg = _resolve_paths(cfg)
    cfg["config_path"] = str(path)
    # Sensible derived defaults.
    ls = cfg["level_shift"]
    if not ls.get("group_keys"):
        ls["group_keys"] = list(cfg["dataset"]["hierarchy"])
    for section in ("baselines", "backtest"):
        cfg[section].setdefault("output_dir", cfg["paths"]["outputs_dir"] / section)
    _check(cfg)
    return cfg


def _check(cfg: dict) -> None:
    d = cfg["dataset"]
    for key in ("item_column", "group_level"):
        v = d.get(key)
        if v is not None and v not in d["hierarchy"] and key == "group_level":
            raise ValueError(f"dataset.group_level={v!r} must be one of dataset.hierarchy {d['hierarchy']}")
    for k in cfg["level_shift"]["group_keys"]:
        if k not in d["hierarchy"]:
            raise ValueError(f"level_shift.group_keys has {k!r}, which is not in dataset.hierarchy {d['hierarchy']}")
    for rule in cfg["price_rules"]["ignore"]:
        missing = {"column", "value", "before"} - set(rule)
        if missing:
            raise ValueError(f"price_rules.ignore entry {rule} is missing {sorted(missing)}")
    if cfg["dataset"]["calendar"] not in ("us", "au"):
        raise ValueError(f"dataset.calendar must be 'us' or 'au', got {cfg['dataset']['calendar']!r}")


def get_config(overrides: dict | None = None) -> dict:
    """VN1 config (configs/vn1.yaml) with optional nested overrides."""
    return load_config(DEFAULT_DATASET, overrides)

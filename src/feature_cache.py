"""Per-origin cache of UNSCALED features (make_raw_features), so the backtest builds each origin once.

Layout: <processed_dir>/features_cache/<key>/origin=YYYY-MM-DD.parquet
The key hashes the feature-related config, the feature/cleaning source code and the input file's
size + mtime, so a change in any of them starts a fresh cache instead of reusing stale features.
"""
import hashlib
import inspect
import json
from pathlib import Path

import pandas as pd

from src import calendar, cleaning, features
from src.evaluation import ALLOWED_INPUT

CONFIG_SECTIONS = ["features", "classification", "stockout", "smoothing", "price_shift", "relative_price",
                   "level_shift", "price_rules", "dataset"]
APPLY_TIME_ONLY = ["scale_method", "optional_features"]


def data_fingerprint(df: pd.DataFrame) -> str:
    """Content hash of a weekly frame (stable across processes): identifies the data in the cache key."""
    cols = sorted(df.columns)
    h = pd.util.hash_pandas_object(df[cols], index=False).to_numpy()
    return hashlib.sha1(h.tobytes() + json.dumps(cols).encode()).hexdigest()[:16]


def cache_key(cfg: dict, processed_dir, input_name: str = ALLOWED_INPUT, fingerprint: str | None = None) -> str:
    if fingerprint is None:
        stat = (Path(processed_dir) / input_name).stat()
        ident = [input_name, stat.st_size, int(stat.st_mtime)]
    else:
        ident = ["fingerprint", fingerprint]
    sections = {k: dict(cfg[k]) for k in CONFIG_SECTIONS}
    for k in APPLY_TIME_ONLY:  # chosen when features are loaded, not stored in the cache
        sections["features"].pop(k, None)
    payload = {
        "cfg": sections,
        "code": [inspect.getsource(features), inspect.getsource(cleaning), inspect.getsource(calendar)],
        "input": ident,
    }
    return hashlib.sha1(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()[:12]


def cache_dir(cfg: dict, processed_dir, input_name: str = ALLOWED_INPUT, fingerprint: str | None = None) -> Path:
    return Path(processed_dir) / "features_cache" / cache_key(cfg, processed_dir, input_name, fingerprint)


def _path(d: Path, origin) -> Path:
    return d / f"origin={pd.Timestamp(origin).date()}.parquet"


def build_cache(weekly: pd.DataFrame, origins, cfg: dict, processed_dir, input_name: str = ALLOWED_INPUT,
                fingerprint: str | None = None) -> Path:
    """Compute and store raw features for every origin not cached yet. Returns the cache directory.
    `input_name` = the cleaned file in processed_dir that `weekly` came from (part of the cache key)."""
    d = cache_dir(cfg, processed_dir, input_name, fingerprint)
    d.mkdir(parents=True, exist_ok=True)
    todo = [o for o in sorted(set(pd.Timestamp(o) for o in origins)) if not _path(d, o).exists()]
    for o in todo:
        fr = features.make_raw_features(weekly, [o], cfg["features"]["horizons"], cfg)
        tmp = _path(d, o).with_suffix(".tmp")
        fr.to_parquet(tmp, index=False)
        tmp.rename(_path(d, o))  # atomic: a crash never leaves a half-written origin behind
    return d


def load_origins(d: Path, origins, columns: list[str] | None = None) -> pd.DataFrame:
    frames = [pd.read_parquet(_path(d, o), columns=columns) for o in origins]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=columns or [])

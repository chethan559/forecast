"""LightGBM for the global direct multi-horizon model.

Objective tweedie: suits non-negative, zero-heavy demand (a point mass at 0 plus a skewed tail).
The model learns the SCALED target; predictions are multiplied back by `scale`.
"""
import lightgbm as lgb
import numpy as np
import pandas as pd

ID_FEATURES = ("client", "warehouse")


def select_features(features: list[str], use_ids: bool) -> list[str]:
    return list(features) if use_ids else [f for f in features if f not in ID_FEATURES]


def fit(train: pd.DataFrame, features: list[str], params: dict, weight_by_scale: bool = True,
        clip_target: float | None = None) -> lgb.Booster:
    """weight_by_scale: weight each row by its series scale, so the loss counts units the way the
    VN1 score does (a 1,000/week series matters more than a 1/week one).
    clip_target: cap the scaled target (limits rows where a near-zero scale inflates it)."""
    train = train[train["target"].notna()]
    label = train["target"].astype(float)
    if clip_target is not None:
        label = label.clip(upper=clip_target)
    weight = train["scale"].astype(float) if weight_by_scale else None
    ds = lgb.Dataset(train[features], label=label, weight=weight, free_raw_data=True)
    return lgb.train(params, ds)  # default 100 rounds, default learning rate / leaves


def predict(model: lgb.Booster, valid: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    fc = valid[["unique_id", "target_date"]].rename(columns={"target_date": "ds"}).copy()
    fc["forecast"] = np.clip(model.predict(valid[features]), 0, None) * valid["scale"].to_numpy(float)
    return fc


def importance(model: lgb.Booster, features: list[str]) -> pd.DataFrame:
    imp = pd.DataFrame({"feature": features, "gain": model.feature_importance("gain"),
                        "splits": model.feature_importance("split")}).sort_values("gain", ascending=False)
    imp["gain_share"] = imp["gain"] / imp["gain"].sum()
    return imp


def fit_predict(train, valid, features, params, weight_by_scale: bool = False, clip_target=None):
    """Fit + predict + importance in one call (used by the feature-stage smoke test)."""
    model = fit(train, features, params, weight_by_scale, clip_target)
    return model, predict(model, valid, features), importance(model, features)

"""VN1 competition score.

score = (sum|F - A| + |sum(F - A)|) / sum(A)
       = accuracy part (MAE-like)   + bias part (total over/under forecast), both scaled by total demand.
Lower is better; 0 = perfect. Forecasting all zeros scores exactly 2.0.
"""
import numpy as np
import pandas as pd

KEY = ["unique_id", "ds"]


def vn1_score(forecast_df: pd.DataFrame, actual_df: pd.DataFrame,
              forecast_col: str = "y_hat", actual_col: str = "y") -> dict:
    """Join forecast and actuals on (unique_id, ds); rows with NaN actuals are ignored.

    A missing forecast for an existing actual is an error (it would silently improve the score).
    """
    act = actual_df[KEY + [actual_col]].dropna(subset=[actual_col])
    df = act.merge(forecast_df[KEY + [forecast_col]], on=KEY, how="left", validate="one_to_one")
    if df[forecast_col].isna().any():
        raise ValueError(f"{int(df[forecast_col].isna().sum())} actuals have no forecast")

    err = df[forecast_col].to_numpy(float) - df[actual_col].to_numpy(float)
    total = df[actual_col].sum()
    if total <= 0:
        raise ValueError("sum of actuals must be positive")
    accuracy = np.abs(err).sum() / total
    bias = abs(err.sum()) / total
    return {"score": accuracy + bias, "accuracy": accuracy, "bias": bias, "n": len(df)}


def vn1_score_by_group(forecast_df: pd.DataFrame, actual_df: pd.DataFrame, groups: pd.Series,
                       forecast_col: str = "y_hat", actual_col: str = "y") -> pd.DataFrame:
    """VN1 score inside each group, plus how much each group contributes to the total.

    groups: Series indexed by unique_id -> group label (e.g. demand_type).
    Columns:
      score / accuracy / bias : the VN1 score computed on that group alone
      signed_bias             : sum(F - A) / sum(A); > 0 means over-forecasting
      unit_share              : group's share of actual units
      abs_error_share         : group's share of total sum|F - A| (where the error comes from)
    The group scores do not add up to the total score (the bias term doesn't decompose), but
    abs_error_share does sum to 1.
    """
    act = actual_df[KEY + [actual_col]].dropna(subset=[actual_col])
    df = act.merge(forecast_df[KEY + [forecast_col]], on=KEY, how="left", validate="one_to_one")
    if df[forecast_col].isna().any():
        raise ValueError(f"{int(df[forecast_col].isna().sum())} actuals have no forecast")
    df["group"] = df["unique_id"].map(groups)
    df["err"] = df[forecast_col] - df[actual_col]
    df["abs_err"] = df["err"].abs()
    g = df.groupby("group", observed=True)
    out = pd.DataFrame({
        "n_series": g["unique_id"].nunique(),
        "actual_units": g[actual_col].sum(),
        "forecast_units": g[forecast_col].sum(),
        "abs_error": g["abs_err"].sum(),
        "signed_error": g["err"].sum(),
    })
    denom = out["actual_units"].replace(0, np.nan)  # a group with no sales has no defined score
    out["accuracy"] = out["abs_error"] / denom
    out["bias"] = out["signed_error"].abs() / denom
    out["score"] = out["accuracy"] + out["bias"]
    out["signed_bias"] = out["signed_error"] / denom
    out["unit_share"] = out["actual_units"] / out["actual_units"].sum()
    out["abs_error_share"] = out["abs_error"] / out["abs_error"].sum()
    return out

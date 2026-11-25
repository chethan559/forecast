"""Post-processing applied to any forecast after the model."""
import numpy as np
import pandas as pd

WEEK = pd.Timedelta(weeks=1)


def cap_forecasts(fc: pd.DataFrame, weekly: pd.DataFrame, origin, k: float, window: int = 52):
    """Cap each forecast at k x the series' max weekly sales in the last `window` weeks up to the origin.

    A safety net against runaway predictions (e.g. a model multiplying an already-high level).
    Data <= origin only. A series with no sales in the window is capped at 0.
    Returns (capped forecast, stats dict).
    """
    origin = pd.Timestamp(origin)
    hist = weekly[(weekly["ds"] <= origin) & (weekly["ds"] > origin - window * WEEK)]
    cap = (k * hist.groupby("unique_id")["y"].max()).rename("cap")
    out = fc.merge(cap, left_on="unique_id", right_index=True, how="left")
    out["cap"] = out["cap"].fillna(0.0)
    capped = out["forecast"] > out["cap"]
    stats = {"n_capped": int(capped.sum()), "share_capped": float(capped.mean()),
             "series_capped": int(out.loc[capped, "unique_id"].nunique()),
             "units_removed_share": float((out["forecast"] - out["cap"]).clip(lower=0).sum() / max(out["forecast"].sum(), 1e-9))}
    out["forecast"] = np.minimum(out["forecast"], out["cap"])
    return out[fc.columns], stats

"""Simple baseline forecasts. Every method has the same contract:

    method(train_df, h=13, ...) -> DataFrame[unique_id, ds, forecast]

train_df is long format (unique_id, ds, y), one row per series-week. The forecast covers the h weeks
right after the LAST training date of the whole panel (not of each series), so every series gets
exactly h rows on the same dates. Forecasts are flat or copied values; no fitting.

Weeks before a series' first sale are ignored (they are "not listed yet", not zero demand). The
cleaned data are already trimmed, but each method re-applies the rule so it is safe on raw data too.
"""
import numpy as np
import pandas as pd


# ----------------------------------------------------------------------------- helpers
def _prepare(train_df: pd.DataFrame) -> pd.DataFrame:
    """Keep unique_id/ds/y, sorted, and drop each series' weeks before its first positive sale."""
    df = train_df[["unique_id", "ds", "y"]].sort_values(["unique_id", "ds"], kind="stable")
    has_sold = df["y"].gt(0).groupby(df["unique_id"]).cummax()
    return df.loc[has_sold].reset_index(drop=True)


def _all_ids(train_df: pd.DataFrame) -> pd.Index:
    # Series that never sold still need a forecast (they get 0).
    return pd.Index(train_df["unique_id"].unique()).sort_values()


def future_dates(train_df: pd.DataFrame, h: int) -> pd.DatetimeIndex:
    """The h weekly dates right after the last training week."""
    last = train_df["ds"].max()
    return pd.date_range(last + pd.Timedelta(weeks=1), periods=h, freq=pd.Timedelta(weeks=1))


def _flat(train_df: pd.DataFrame, level: pd.Series, h: int) -> pd.DataFrame:
    """Repeat one level per series over the h future weeks. Missing series -> 0."""
    ids = _all_ids(train_df)
    level = level.reindex(ids).fillna(0.0).clip(lower=0)
    dates = future_dates(train_df, h)
    return pd.DataFrame({
        "unique_id": np.repeat(ids.to_numpy(), h),
        "ds": np.tile(dates.to_numpy(), len(ids)),
        "forecast": np.repeat(level.to_numpy(float), h),
    })


def _last_window_mean(df: pd.DataFrame, end: pd.Timestamp, window: int) -> pd.Series:
    """Mean of y over the last `window` calendar weeks up to `end`, using only weeks after first sale.

    A series with fewer post-first-sale weeks than `window` is averaged over the weeks it has
    (e.g. a 5-week-old series uses 5 weeks), rather than padding with zeros it never had.
    """
    start = end - pd.Timedelta(weeks=window - 1)
    recent = df[df["ds"].between(start, end)]
    return recent.groupby("unique_id")["y"].mean()


# ----------------------------------------------------------------------------- methods
def zero(train_df: pd.DataFrame, h: int = 13) -> pd.DataFrame:
    """All zeros. Sanity check: the VN1 score of this must be exactly 2.0."""
    return _flat(train_df, pd.Series(dtype=float), h)


def moving_avg(train_df: pd.DataFrame, h: int = 13, window: int = 13) -> pd.DataFrame:
    """Flat forecast = mean of the last `window` weeks (post first sale)."""
    df = _prepare(train_df)
    return _flat(train_df, _last_window_mean(df, train_df["ds"].max(), window), h)


def seasonal_naive(train_df: pd.DataFrame, h: int = 13, season_length: int = 52,
                   fallback_window: int = 13) -> pd.DataFrame:
    """Forecast for week t = actual of week t - 52 (same week last year).

    Needs the series to have sold at least `season_length` weeks before the end of training, so that
    every week t-52 (for t in the horizon) is inside its history. Otherwise -> moving_avg(fallback_window).
    Note 52 weeks = 364 days, so the 'same week' drifts by a day per year; fine for weekly data.
    """
    df = _prepare(train_df)
    end = train_df["ds"].max()
    out = moving_avg(train_df, h, fallback_window)

    first_sale = df.groupby("unique_id")["ds"].min()
    history_weeks = (end - first_sale).dt.days // 7 + 1
    eligible = history_weeks.index[history_weeks >= season_length]

    lagged = df[df["unique_id"].isin(eligible)].copy()
    lagged["ds"] = lagged["ds"] + pd.Timedelta(weeks=season_length)  # move last year's weeks forward
    lagged = lagged.rename(columns={"y": "snaive"})
    out = out.merge(lagged, on=["unique_id", "ds"], how="left")
    out["forecast"] = out["snaive"].fillna(out["forecast"])
    return out[["unique_id", "ds", "forecast"]]


def moving_avg_inactive_zero(train_df: pd.DataFrame, h: int = 13, window: int = 13,
                             inactive_weeks: int = 13) -> pd.DataFrame:
    """moving_avg(window), but 0 for series with no sales in the last `inactive_weeks` weeks.

    With window == inactive_weeks this is identical to moving_avg: a series with no sales in the
    last 13 weeks already has a 13-week mean of 0. It only differs when window > inactive_weeks.
    """
    df = _prepare(train_df)
    end = train_df["ds"].max()
    level = _last_window_mean(df, end, window)
    recent_units = df[df["ds"] > end - pd.Timedelta(weeks=inactive_weeks)].groupby("unique_id")["y"].sum()
    inactive = recent_units.reindex(level.index).fillna(0).eq(0)
    return _flat(train_df, level.where(~inactive, 0.0), h)


def run_all(train_df: pd.DataFrame, cfg: dict) -> dict[str, pd.DataFrame]:
    """Every baseline from the config, keyed by a short method name."""
    b = cfg["baselines"]
    h = b["h"]
    out = {"zero": zero(train_df, h)}
    for w in b["moving_avg_windows"]:
        out[f"moving_avg_{w}"] = moving_avg(train_df, h, w)
    out["seasonal_naive"] = seasonal_naive(train_df, h, b["season_length"], b["fallback_window"])
    out["moving_avg_13_inactive_zero"] = moving_avg_inactive_zero(train_df, h, 13, b["inactive_weeks"])
    return out


def blend(fc_a: pd.DataFrame, fc_b: pd.DataFrame, w: float) -> pd.DataFrame:
    """w * fc_a + (1 - w) * fc_b, matched on (unique_id, ds). Both must cover the same rows."""
    m = fc_a.merge(fc_b, on=["unique_id", "ds"], suffixes=("_a", "_b"), validate="one_to_one")
    assert len(m) == len(fc_a) == len(fc_b), "blend inputs cover different rows"
    m["forecast"] = w * m["forecast_a"] + (1 - w) * m["forecast_b"]
    return m[["unique_id", "ds", "forecast"]]


def seasonal_index(train_df: pd.DataFrame, h: int = 13, window: int = 13, season_length: int = 52,
                   clip: tuple[float, float] = (0.25, 4.0), group_col: str | None = "client") -> pd.DataFrame:
    """moving_avg(window) x a client-level seasonal ratio taken from last year.

    For target week W and client c:
        ratio = client units in week W - 52  /  mean client units/week over the `window` weeks
                ending at (origin - 52)
    i.e. "how did last year's same week compare with last year's level at the same point".
    The moving average gives this year's level; the ratio gives the shape. Client totals pool many
    series, so the shape is much less noisy than a single series' seasonal naive.
    `group_col` = the grouping column (VN1: client); None = all series form one group.
    Ratio = 1 where last year is missing or zero; clipped to `clip`.
    """
    if group_col is None or group_col not in train_df.columns:
        train_df = train_df.assign(_group=0)
        group_col = "_group"
    train_df = train_df.rename(columns={group_col: "client"}) if group_col != "client" else train_df
    end = train_df["ds"].max()
    out = moving_avg(train_df, h, window).merge(
        train_df.groupby("unique_id")["client"].first().rename("client").reset_index(), on="unique_id", how="left")

    weeks = pd.date_range(end - (season_length + window + h) * pd.Timedelta(weeks=1), end, freq=pd.Timedelta(weeks=1))
    tot = (train_df[train_df["ds"].isin(weeks)].groupby(["client", "ds"])["y"].sum()
           .unstack("ds").reindex(columns=weeks, fill_value=0).fillna(0))
    ly_origin = end - season_length * pd.Timedelta(weeks=1)
    base = tot.loc[:, (tot.columns > ly_origin - window * pd.Timedelta(weeks=1)) & (tot.columns <= ly_origin)].mean(axis=1)

    ratios = []
    for ds in future_dates(train_df, h):
        ly = ds - season_length * pd.Timedelta(weeks=1)
        num = tot[ly] if ly in tot.columns else pd.Series(np.nan, index=tot.index)
        r = (num / base.where(base > 0)).fillna(1.0).clip(*clip)
        ratios.append(pd.DataFrame({"client": tot.index, "ds": ds, "ratio": r.to_numpy()}))
    out = out.merge(pd.concat(ratios), on=["client", "ds"], how="left")
    out["forecast"] = out["forecast"] * out["ratio"].fillna(1.0)
    return out[["unique_id", "ds", "forecast"]]

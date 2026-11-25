"""Data cleaning for weekly demand panels.

Input: long DataFrames in the adapter schema (unique_id, client, warehouse, product, ds, y / price).
Principle: FLAG, don't fix. Every step adds columns; raw values are kept (y is never changed,
price_raw sits next to price_clean). The only rows removed are leading zeros before a series'
first sale (step a), and their count is recorded per series so it is fully traceable.

Every step is a separate function so it can be tested and re-used on other sources (e.g. Shopify).
All thresholds come from the config dict (src/config.py).
"""
import numpy as np
import pandas as pd

KEY = ["unique_id", "ds"]


# ----------------------------------------------------------------------------- helpers
def _sorted(df: pd.DataFrame) -> pd.DataFrame:
    return df.sort_values(KEY, kind="stable").reset_index(drop=True)


def _run_ids(df: pd.DataFrame, values: pd.Series) -> pd.Series:
    """Label consecutive runs of equal `values` within each series (a new series always starts a new run)."""
    new_series = df["unique_id"].ne(df["unique_id"].shift())
    changed = values.ne(values.shift())
    return (new_series | changed).cumsum()


def _last_date(df: pd.DataFrame) -> pd.Timestamp:
    return df["ds"].max()


# ----------------------------------------------------------------------------- (a) trim
def trim_leading_zeros(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Drop weeks before each series' first positive sale.

    Why: VN1 series are padded with zeros before the product existed (or before it was listed at
    this warehouse). Those zeros are not "demand was zero", they are "no product yet", and would
    make every series look intermittent and drag forecasts towards zero.

    Returns (trimmed_df, per-series table with first_sale_date and n_leading_zero_weeks).
    Series that never sold are dropped from the weekly table but kept in the per-series table
    with first_sale_date = NaT, so nothing disappears silently.
    """
    df = _sorted(df)
    has_sold = df["y"].gt(0).groupby(df["unique_id"]).cummax()  # True from the first sale onwards
    info = pd.DataFrame({"series_start": df.groupby("unique_id")["ds"].min()})
    info["first_sale_date"] = df.loc[df["y"].gt(0)].groupby("unique_id")["ds"].min().reindex(info.index)
    info["n_leading_zero_weeks"] = ((info["first_sale_date"] - info["series_start"]).dt.days // 7).astype("Int64")
    info["never_sold"] = info["first_sale_date"].isna()
    return df.loc[has_sold].reset_index(drop=True), info.drop(columns="series_start")


# ----------------------------------------------------------------------------- (b) smoothed era
def weekly_repeat_rate(df: pd.DataFrame) -> pd.Series:
    """Per week: share of positive sales values that exactly equal the same series' previous week.

    In genuinely weekly count data this is low and stable (only low-volume series repeat by chance).
    If older data are monthly totals spread over weeks, consecutive weeks are identical far more often.
    """
    df = _sorted(df)
    prev = df.groupby("unique_id")["y"].shift()
    pos = df["y"].gt(0) & prev.notna()
    same = pos & df["y"].eq(prev)
    rate = same.groupby(df["ds"]).sum() / pos.groupby(df["ds"]).sum().replace(0, np.nan)
    return rate.dropna()


def estimate_weekly_switch_date(df: pd.DataFrame, min_segment: int = 13) -> dict:
    """Find the single date where the weekly repeat rate changes level the most.

    Method: classic one-changepoint search. For every candidate split we fit one mean before and
    one mean after, and keep the split with the lowest total squared error. The first week of
    the second segment is the switch date (first week of "true weekly" data).
    """
    rate = weekly_repeat_rate(df)
    x = rate.to_numpy()
    n = len(x)
    if n < 2 * min_segment:
        return {"switch_date": None, "rate": rate}
    best_k, best_sse = None, np.inf
    for k in range(min_segment, n - min_segment + 1):
        a, b = x[:k], x[k:]
        sse = ((a - a.mean()) ** 2).sum() + ((b - b.mean()) ** 2).sum()
        if sse < best_sse:
            best_k, best_sse = k, sse
    return {
        "switch_date": rate.index[best_k],
        "rate_before": float(x[:best_k].mean()),
        "rate_after": float(x[best_k:].mean()),
        "rate": rate,
    }


def flag_smoothed_weeks(df: pd.DataFrame, cfg: dict) -> tuple[pd.DataFrame, dict]:
    """Flag runs of >= min_run_length identical consecutive positive values, and the smoothed era.

    Columns added (values are NOT changed):
      identical_run_len : length of the run of identical values this week belongs to
      identical_run     : week is in a run of >= N identical positive values (pure pattern detector)
      is_smoothed_era   : week is before the estimated switch to true weekly data
      is_smoothed       : identical_run AND is_smoothed_era  (the flag to use downstream)
    Runs after the switch date are kept in identical_run: they are mostly chance repeats of
    small counts (1,1,1) and their rate is the "noise floor" of the detector.
    """
    c = cfg["smoothing"]
    df = _sorted(df)
    run = _run_ids(df, df["y"])
    run_len = df.groupby(run)["y"].transform("size")
    df["identical_run_len"] = run_len.astype("int32")
    df["identical_run"] = df["y"].gt(0) & run_len.ge(c["min_run_length"])

    if c.get("switch_date"):
        info = {"switch_date": pd.Timestamp(c["switch_date"]), "method": "forced in config"}
    else:
        info = estimate_weekly_switch_date(df, c["switch_min_segment_weeks"])
        info["method"] = "changepoint on weekly repeat rate"
    sd = info["switch_date"]
    df["is_smoothed_era"] = df["ds"] < sd if sd is not None else False
    df["is_smoothed"] = df["identical_run"] & df["is_smoothed_era"]
    return df, info


# ----------------------------------------------------------------------------- (c) price
def check_price_missing_pattern(df: pd.DataFrame) -> dict:
    """Verify the VN1 rule 'price is NaN exactly when sales are 0'. Returns counts, never raises."""
    nan, zero_sales = df["price"].isna(), df["y"].eq(0)
    return {
        "nan_and_zero_sales": int((nan & zero_sales).sum()),
        "nan_but_sales_pos": int((nan & ~zero_sales).sum()),
        "price_but_zero_sales": int((~nan & zero_sales).sum()),
        "holds_exactly": bool((nan == zero_sales).all()),
    }


def clean_price(df: pd.DataFrame) -> pd.DataFrame:
    """Build a usable price column without touching the raw one.

    price_raw      : original value
    price_observed : a price was recorded this week (VN1 only records one when something sold)
    price_zero     : recorded price is 0 while units sold > 0 -> almost certainly a data error,
                     treated as missing
    price_clean    : last valid price carried FORWARD within the series. Never backfilled:
                     a backfill would copy a future price into the past (leakage).
                     Weeks before the first valid price stay NaN.
    price_filled   : price_clean came from an earlier week
    """
    df = _sorted(df).rename(columns={"price": "price_raw"})
    df["price_observed"] = df["price_raw"].notna()
    df["price_zero"] = df["price_raw"].eq(0) & df["y"].gt(0)
    valid = df["price_raw"].where(df["price_observed"] & ~df["price_zero"] & df["price_raw"].gt(0))
    df["price_valid"] = valid  # helper used by later steps: observed, positive prices only
    df["price_clean"] = valid.groupby(df["unique_id"]).ffill()
    df["price_filled"] = valid.isna() & df["price_clean"].notna()
    return df


# ----------------------------------------------------------------------------- (d) price level shift
def _locate_shift(log_p: np.ndarray) -> int:
    """Index in a short window where log price best splits into two levels (min squared error)."""
    idx = np.flatnonzero(~np.isnan(log_p))
    v = log_p[idx]
    best_i, best_sse = idx[-1], np.inf
    for j in range(1, len(v)):
        a, b = v[:j], v[j:]
        sse = ((a - a.mean()) ** 2).sum() + ((b - b.mean()) ** 2).sum()
        if sse < best_sse:
            best_i, best_sse = idx[j], sse
    return int(best_i)


def flag_price_level_shift(df: pd.DataFrame, cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Detect structural price jumps (e.g. a change of pack size or currency, not a promotion).

    Rule: m_now = median of valid prices in the trailing `window` weeks; m_before = the same median
    `window` weeks earlier (two back-to-back, non-overlapping windows, past data only).
    A shift is detected at week t when m_now / m_before > threshold or < 1/threshold.
    Medians ignore one-off promo weeks, so only lasting level changes trigger.

    Because the trailing median only flips after most of the window has the new price, detection
    lags the real jump by a few weeks. We therefore also locate the actual shift week, searching
    only the 2*window weeks up to the detection week (still no future data relative to detection).

    Columns:
      price_shift          : True on the located (back-dated) shift week. Descriptive only: it uses
                             up to 2*window weeks after that week, so do NOT use it as a model feature.
      price_shift_detected : True on the week the trailing rule fires (causal, known in real time).
      price_regime         : 0, 1, 2... increments at each DETECTED shift. add_relative_price resets
                             its window here, which keeps relative_price strictly past-only.
    """
    c = cfg["price_shift"]
    w, thr = c["window"], c["ratio_threshold"]
    df = _sorted(df)
    g = df.groupby("unique_id")["price_valid"]
    m_now = g.rolling(w, min_periods=c["min_periods"]).median().reset_index(level=0, drop=True)
    m_before = m_now.groupby(df["unique_id"]).shift(w)
    ratio = m_now / m_before
    fired = ratio.gt(thr) | ratio.lt(1 / thr)

    df["price_shift"] = False
    df["price_shift_detected"] = False
    events = []
    log_p = np.log(df["price_valid"].to_numpy())
    pos_in_series = df.groupby("unique_id").cumcount().to_numpy()
    for uid, rows in df.index[fired].groupby(df.loc[fired, "unique_id"]).items():
        next_allowed = -1
        for r in rows:
            if r < next_allowed:  # same jump keeps the ratio high for ~window weeks: count it once
                continue
            start = r - min(pos_in_series[r], 2 * w - 1)
            loc = start + _locate_shift(log_p[start:r + 1])
            df.at[loc, "price_shift"] = True
            df.at[r, "price_shift_detected"] = True
            events.append({"unique_id": uid, "shift_date": df.at[loc, "ds"],
                           "detected_date": df.at[r, "ds"], "ratio": float(ratio.iat[r])})
            next_allowed = r + w
    df["price_regime"] = df.groupby("unique_id")["price_shift_detected"].cumsum().astype("int16")
    ev = pd.DataFrame(events, columns=["unique_id", "shift_date", "detected_date", "ratio"])
    return df, ev


# ----------------------------------------------------------------------------- (e) relative price
def add_relative_price(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """relative_price = price_clean / median of valid prices in the PAST `window` weeks.

    'Past' means weeks t-window .. t-1 (the current week is excluded, so a discount this week
    shows up as < 1 instead of being half-absorbed into its own baseline). Windows are trailing,
    never centred, so the feature can be computed the same way at forecast time.
    The window restarts when a price level shift is DETECTED (price_regime), otherwise every week
    after a 3x jump would look like a 3x price increase for the next 13 weeks. We reset at the
    detection week, not the back-dated shift week, because only the detection is known in real time;
    the few weeks between the jump and its detection keep a high relative_price, as they would live.
    NaN when there is no past valid price in the current regime (e.g. first week of a series).
    """
    c = cfg["relative_price"]
    df = _sorted(df)
    keys = [df["unique_id"], df["price_regime"]]
    past = df["price_valid"].groupby(keys).shift(1)
    base = (past.groupby(keys).rolling(c["window"], min_periods=c["min_periods"]).median()
            .reset_index(level=[0, 1], drop=True).sort_index())
    df["price_baseline"] = base
    df["relative_price"] = df["price_clean"] / base
    return df


# ----------------------------------------------------------------------------- (f) stockouts
def flag_suspected_stockouts(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Flag zero runs that look like 'could not sell' rather than 'nobody wanted it'.

    No inventory data exist, so we infer: a run of >= min_zero_run zero weeks, in a series that sold
    >= min_pre_mean units/week on average over the pre_window weeks before the run. A busy item that
    suddenly sells nothing for a month is more likely out of stock than unwanted.
    Runs still open at the end of the data are skipped by default (could be a delisting).
    Flag only: the zeros stay in y. Models can later mask or down-weight them.
    """
    c = cfg["stockout"]
    df = _sorted(df)
    is_zero = df["y"].eq(0)
    run = _run_ids(df, is_zero)
    run_len = df.groupby(run)["y"].transform("size")
    df["zero_run_len"] = run_len.where(is_zero, 0).astype("int32")

    # Trailing mean of the pre_window weeks strictly before each week (shift(1) = past only).
    pre_mean = (df.groupby("unique_id")["y"].shift(1)
                .groupby(df["unique_id"]).rolling(c["pre_window"], min_periods=c["min_pre_periods"]).mean()
                .reset_index(level=0, drop=True))
    run_pre_mean = pre_mean.groupby(run).transform("first")  # value at the first week of the run
    is_last_row = df["unique_id"].ne(df["unique_id"].shift(-1))
    run_is_open = is_last_row.groupby(run).transform("max")  # run reaches the end of the series

    df["is_suspected_stockout"] = (
        is_zero
        & run_len.ge(c["min_zero_run"])
        & (run_len.le(c["max_zero_run"]) if c.get("max_zero_run") else True)
        & run_pre_mean.ge(c["min_pre_mean"])
        & (c["include_trailing_run"] | ~run_is_open)
    )
    return df


# ----------------------------------------------------------------------------- (g) classify
def _sb_stats(df: pd.DataFrame, cfg: dict, suffix: str = "") -> pd.DataFrame:
    """ADI and CV2 (Syntetos-Boylan) per series.

    ADI  = weeks / weeks with demand  (average gap between sales; 1 = sells every week)
    CV2  = (std / mean)^2 of the non-zero demand sizes (how variable the sizes are)
    smooth: ADI<1.32 & CV2<0.49 | erratic: ADI<1.32 & CV2>=0.49
    intermittent: ADI>=1.32 & CV2<0.49 | lumpy: ADI>=1.32 & CV2>=0.49
    """
    c = cfg["classification"]
    nz = df.loc[df["y"].gt(0)].groupby("unique_id")["y"]
    out = pd.DataFrame({
        "n_weeks": df.groupby("unique_id").size(),
    })
    out["n_nonzero"] = nz.size().reindex(out.index).fillna(0).astype(int)
    out["adi"] = out["n_weeks"] / out["n_nonzero"].replace(0, np.nan)
    mean, std = nz.mean().reindex(out.index), nz.std(ddof=0).reindex(out.index)
    out["cv2"] = (std / mean) ** 2
    intermittent = out["adi"].ge(c["adi_threshold"])
    variable = out["cv2"].ge(c["cv2_threshold"])
    out["demand_type"] = np.select(
        [~intermittent & ~variable, ~intermittent & variable, intermittent & ~variable, intermittent & variable],
        ["smooth", "erratic", "intermittent", "lumpy"], default="no_sales")
    out.loc[out["n_nonzero"].eq(0), "demand_type"] = "no_sales"
    return out[["adi", "cv2", "demand_type"]].add_suffix(suffix)


def classify_series(df: pd.DataFrame, cfg: dict, switch_date=None) -> pd.DataFrame:
    """One row per series: history length, ADI/CV2 demand type, new/inactive flags, volume rank.

    Uses only the rows passed in (the caller decides the training cut-off).
    demand_type uses cfg.classification.from_date (default: whole trimmed history); when the
    weekly switch date is known we also report demand_type_weekly_era (weekly-era rows only),
    because the smoothed era hides week-to-week variability and makes CV2 look smaller.
    """
    c = cfg["classification"]
    end = _last_date(df)
    g = df.groupby("unique_id")
    t = g.agg(client=("client", "first"), warehouse=("warehouse", "first"), product=("product", "first"),
              history_weeks=("ds", "size"), total_units=("y", "sum"))
    last_sale = df.loc[df["y"].gt(0)].groupby("unique_id")["ds"].max()
    t["last_sale_date"] = last_sale.reindex(t.index)

    sub = df if not c.get("from_date") else df[df["ds"] >= pd.Timestamp(c["from_date"])]
    t = t.join(_sb_stats(sub, cfg))
    if switch_date is not None:
        t = t.join(_sb_stats(df[df["ds"] >= switch_date], cfg, "_weekly_era"))

    t["is_new"] = t["history_weeks"] < c["new_series_weeks"]
    recent_cut = end - pd.Timedelta(weeks=c["inactive_weeks"] - 1)
    recent_units = df.loc[df["ds"] >= recent_cut].groupby("unique_id")["y"].sum().reindex(t.index).fillna(0)
    t["is_inactive"] = recent_units.eq(0)

    t["volume_rank"] = t["total_units"].rank(ascending=False, method="first").astype(int)
    t = t.sort_values("volume_rank")
    t["volume_share"] = t["total_units"] / t["total_units"].sum()
    t["cum_volume_share"] = t["volume_share"].cumsum()
    return t.sort_index()


# ----------------------------------------------------------------------------- pipeline
WEEKLY_COLUMNS = [
    "unique_id", "client", "warehouse", "product", "ds", "y",
    "identical_run_len", "identical_run", "is_smoothed_era", "is_smoothed",
    "price_raw", "price_observed", "price_zero", "price_clean", "price_filled",
    "price_shift", "price_shift_detected", "price_regime", "price_baseline", "relative_price",
    "zero_run_len", "is_suspected_stockout",
    "level_shift", "level_shift_up", "level_regime", "group_level_shift",
]


def run_cleaning(sales_df: pd.DataFrame, price_df: pd.DataFrame | None, config: dict,
                 return_diagnostics: bool = False):
    """Chain steps a-g. Returns (weekly_clean, series_table) [, diagnostics dict for the report]."""
    limit = pd.Timestamp(config["max_allowed_date"])
    for name, d in [("sales", sales_df), ("price", price_df)]:
        if d is not None and d["ds"].max() > limit:
            raise ValueError(f"{name} data run past max_allowed_date={limit.date()}: held-out data must "
                             "never enter cleaning")

    if "price" in sales_df.columns:  # already merged by the adapter
        df = sales_df.copy()
    elif price_df is not None:
        df = sales_df.merge(price_df[KEY + ["price"]], on=KEY, how="left", validate="one_to_one")
    else:
        df = sales_df.assign(price=np.nan)
    price_check = check_price_missing_pattern(df)

    df, first_sale = trim_leading_zeros(df)                       # a
    df, switch = flag_smoothed_weeks(df, config)                   # b
    df = clean_price(df)                                           # c
    df, shifts = flag_price_level_shift(df, config)                # d
    df = add_relative_price(df, config)                            # e
    df = flag_suspected_stockouts(df, config)                      # f
    df = detect_level_shifts(df, config)                           # h (series level)
    level_events = level_shift_group_events(df, config)            # h (warehouse / client level)
    gsd = group_shift_dates(df, level_events, config)
    df["group_level_shift"] = pd.MultiIndex.from_frame(df[["unique_id", "ds"]]).isin(
        pd.MultiIndex.from_frame(gsd.rename(columns={"shift_ds": "ds"})))
    series = classify_series(df, config, switch.get("switch_date"))  # g

    # Per-series roll-up of every weekly flag so series_table is self-contained.
    g = df.groupby("unique_id")
    roll = pd.DataFrame({
        "n_smoothed_weeks": g["is_smoothed"].sum(),
        "n_identical_run_weeks": g["identical_run"].sum(),
        "n_price_zero_weeks": g["price_zero"].sum(),
        "n_suspected_stockout_weeks": g["is_suspected_stockout"].sum(),
        "n_stockout_runs": df.loc[df["is_suspected_stockout"] & ~df.groupby("unique_id")["is_suspected_stockout"]
                                  .shift(fill_value=False)].groupby("unique_id").size(),
        "trailing_zero_weeks": g["zero_run_len"].last(),
    }).fillna({"n_stockout_runs": 0})
    first_shift = shifts.groupby("unique_id").agg(price_shift_date=("shift_date", "min"),
                                                  n_price_shifts=("shift_date", "size"),
                                                  first_shift_ratio=("ratio", "first"))
    series = (first_sale.join(series, how="left").join(roll).join(first_shift))
    series["n_price_shifts"] = series["n_price_shifts"].fillna(0).astype(int)
    series["has_price_shift"] = series["n_price_shifts"].gt(0)
    series["has_suspected_stockout"] = series["n_suspected_stockout_weeks"].fillna(0).gt(0)
    ls = df[df["level_shift"]].groupby("unique_id")["ds"]
    series = series.join(ls.min().rename("level_shift_date")).join(ls.size().rename("n_level_shifts"))
    series["n_level_shifts"] = series["n_level_shifts"].fillna(0).astype(int)
    series = series.join(gsd.groupby("unique_id")["shift_ds"].min().rename("group_level_shift_date"))
    series = series.reset_index()

    if not return_diagnostics:
        return df[WEEKLY_COLUMNS], series
    rate = switch.pop("rate", None)
    diag = {"switch": switch, "price_check": price_check, "price_shift_events": shifts,
            "level_shift_events": level_events,
            "repeat_rate": rate, "data_end": _last_date(df)}
    return df[WEEKLY_COLUMNS], series, diag


# ----------------------------------------------------------------------------- (h) level shifts
def detect_level_shifts(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Flag sudden, lasting changes in a series' sales level (past data only).

    recent   = mean y over the last `short_window` weeks (t-3 .. t)
    previous = mean y over the `long_window` weeks before those (t-16 .. t-4)
    up   : recent > ratio x previous, and recent >= min_level
    down : recent < previous / ratio, and previous >= min_level
    The higher level must be >= min_level so 0 -> 1 unit noise on slow sellers doesn't count.
    A level shift keeps the condition true for a few weeks; only the first firing week counts as the
    event (no firing in the previous cooldown_weeks). Needs short + long weeks of history.

    Columns: level_active (enough history to test), level_shift (event week), level_shift_up,
    level_regime (0, 1, 2... increments at each event).
    """
    c = cfg["level_shift"]
    df = _sorted(df)
    g = df.groupby("unique_id")["y"]
    recent = g.rolling(c["short_window"], min_periods=c["short_window"]).mean().reset_index(level=0, drop=True)
    prev = (g.shift(c["short_window"]).groupby(df["unique_id"])
            .rolling(c["long_window"], min_periods=c["long_window"]).mean().reset_index(level=0, drop=True))
    up = (recent > c["ratio"] * prev) & (recent >= c["min_level"])
    down = (recent * c["ratio"] < prev) & (prev >= c["min_level"])
    fired = up | down
    fired_before = (fired.groupby(df["unique_id"]).shift(1, fill_value=False).astype(float)
                    .groupby(df["unique_id"]).rolling(c["cooldown_weeks"], min_periods=1).max()
                    .reset_index(level=0, drop=True))
    df["level_active"] = prev.notna()
    df["level_shift"] = fired & fired_before.eq(0)
    df["level_shift_up"] = df["level_shift"] & up
    df["level_regime"] = df.groupby("unique_id")["level_shift"].cumsum().astype("int16")
    return df


def level_shift_group_events(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Weeks where most of a warehouse's (or client's) active series shift level together.

    Needs the columns from detect_level_shifts. 'Active' = series with enough history to be tested
    that week. A group event is a structural change (a warehouse going live, volume moved between
    sites), not normal demand: the whole group moves at once.
    """
    c = cfg["level_shift"]
    out = []
    for key in c["group_keys"]:
        t = df.groupby([key, "ds"]).agg(n_active=("level_active", "sum"), n_shifted=("level_shift", "sum"),
                                         n_up=("level_shift_up", "sum")).reset_index()
        t["share"] = t["n_shifted"] / t["n_active"].replace(0, np.nan)
        ev = t[(t["share"] > c["group_share"]) & (t["n_active"] >= c["min_group_series"])]
        out.append(ev.rename(columns={key: "group_id"}).assign(group_type=key,
                                                              direction=np.where(ev["n_up"] * 2 >= ev["n_shifted"], "up", "down")))
    cols = ["group_type", "group_id", "ds", "n_active", "n_shifted", "share", "direction"]
    return pd.concat(out, ignore_index=True)[cols] if out else pd.DataFrame(columns=cols)


def group_shift_dates(df: pd.DataFrame, events: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """(unique_id, shift_ds) for every group event of the series' warehouse or client.

    Group events apply to every series of the group (even one whose own level did not cross the
    threshold that exact week): the cause is the group, not the series.
    """
    if events.empty:
        return pd.DataFrame(columns=["unique_id", "shift_ds"])
    parts = []
    ids = df.drop_duplicates("unique_id")
    for key in cfg["level_shift"]["group_keys"]:
        ev = events[events["group_type"] == key][["group_id", "ds"]].rename(columns={"group_id": key, "ds": "shift_ds"})
        parts.append(ids[["unique_id", key]].merge(ev, on=key)[["unique_id", "shift_ds"]])
    return pd.concat(parts).drop_duplicates().sort_values(["unique_id", "shift_ds"]).reset_index(drop=True)

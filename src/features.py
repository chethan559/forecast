"""Features for a global, direct, multi-horizon model.

One row per (unique_id, origin T, horizon h). Target = y at T + h. Every feature is computed from
weeks <= T only. To make that hard to break:
  * make_features() keeps only RAW input columns (ids, ds, y, price_raw). Flags stored in the
    cleaned file (is_smoothed, is_suspected_stockout, price regimes, series_table stats) were
    computed on the whole of Phase 0, so they partly use weeks after T. They are recomputed here,
    per origin, on data <= T with the same cleaning functions.
  * _features_at_origin() first slices the data to ds <= T; only _targets() looks after T.
  * tests/test_features.py changes every week after T and checks no feature changes.

Scaling: two options, picked by config features.scale_method (see apply_scaling):
  mean13   = mean of the last 13 weeks at T (fallback: mean since first sale; floor 0.1)
  max13_52 = max(mean of last 13 weeks, mean of last 52 weeks since first sale); floor 0.1
Target and every sales-based feature are divided by it, so one model can learn from series that
sell 1/week and 1,000/week. Unscaled copies are kept with a `_raw` suffix (debugging only; they
are not in MODEL_FEATURES).
"""
import numpy as np
import pandas as pd

from src.cleaning import (add_relative_price, clean_price, detect_level_shifts, flag_price_level_shift,
                          flag_smoothed_weeks, flag_suspected_stockouts, group_shift_dates,
                          level_shift_group_events)

BASE_COLUMNS = ["unique_id", "client", "warehouse", "product", "ds", "y", "price_raw"]
FLAG_HISTORY_WEEKS = 80  # enough look-back for 52-week flag shares + 13-week stockout pre-window + runs
WEEK = pd.Timedelta(weeks=1)


# ----------------------------------------------------------------------------- feature registry
def feature_specs(cfg: dict) -> list[dict]:
    """Name, group and plain-English definition of every model feature (drives reports/features.md)."""
    f = cfg["features"]
    specs = []

    def add(name, group, definition, scaled=False, optional=None):
        specs.append({"name": name, "group": group, "definition": definition, "scaled": scaled,
                      "optional": optional})

    for k in f["lags"]:
        add(f"lag_{k}", "recent demand", f"y at week T-{k - 1} (lag 1 = origin week)", True)
    for w in f["roll_windows"]:
        add(f"roll_mean_{w}", "recent demand", f"mean y over the last {w} weeks (post first sale)", True)
        add(f"roll_std_{w}", "recent demand", f"std of y over the last {w} weeks", True)
    add(f"roll_max_{f['max_window']}", "recent demand", f"max y over the last {f['max_window']} weeks", True)
    add("trend_4_13", "trend", "mean4 / mean13 (NaN if mean13 = 0)")
    add("trend_13_52", "trend", "mean13 / mean52 (NaN if mean52 = 0)")
    add("zero_share_13", "intermittency", "share of zero weeks in the last 13 weeks")
    add("zero_share_52", "intermittency", "share of zero weeks in the last 52 weeks")
    add("weeks_since_last_sale", "intermittency", "weeks from the last positive sale to T (0 = sold at T)")
    add("age_weeks", "age", "weeks from first sale to T")
    add("seas_lag_52", "seasonality", "y at target week - 52 (NaN if before first sale)", True)
    add("seas_mean_3", "seasonality", "mean y over target week - 53 .. - 51 (available weeks only)", True)
    add("tw_week_of_year", "calendar", "ISO week number of the target week")
    add("tw_month", "calendar", "month of the target week (Monday date)")
    add("tw_thanksgiving_week", "calendar", "target week contains US Thanksgiving (and Black Friday)")
    add("tw_cyber_week", "calendar", "target week starts on Cyber Monday")
    add("tw_christmas_week", "calendar", "target week contains 25 December")
    # Optional seasonality features (off unless listed in config features.optional_features).
    add("ly_uplift_series", "seasonality (optional)",
        "series: mean y over target week -53..-51 / mean y over the 13 weeks ending at origin - 52 "
        "(last year's target weeks vs last year's level at the same point; NaN if no history then)", optional="uplift")
    add("ly_uplift_client", "seasonality (optional)",
        "same ratio on the client's total units (pools all its series, so less noisy)", optional="uplift")
    add("tw_weeks_to_thanksgiving", "calendar (optional)",
        "signed weeks from the target week to the nearest Thanksgiving week (clipped to ±13)", optional="holiday_distance")
    add("tw_weeks_to_christmas", "calendar (optional)",
        "signed weeks from the target week to the nearest Christmas week (clipped to ±13)", optional="holiday_distance")
    add("prod_mean_13", "hierarchy", "same product summed over all its warehouses: mean weekly units, last 13 weeks", True)
    add("client_trend_4_13", "hierarchy", "client total units: mean4 / mean13")
    add("client_share_13", "hierarchy", "series units / client units over the last 13 weeks")
    add("relative_price", "price", "price at T / median valid price of the 13 weeks before (past only, reset at detected shifts)")
    add("price_observed_share_13", "price", "share of the last 13 weeks with a trusted recorded price")
    add("client", "categorical", "client id (categorical)")
    add("warehouse", "categorical", "warehouse id (categorical)")
    add("demand_type", "categorical", "Syntetos-Boylan class from ADI/CV2 over the last 52 weeks at T")
    add("smoothed_share_13", "data quality", "share of last 13 weeks in an identical run before the weekly switch")
    add("smoothed_share_52", "data quality", "same over the last 52 weeks")
    add("stockout_share_13", "data quality", "share of last 13 weeks flagged as suspected stockout (recomputed at T)")
    add("stockout_share_52", "data quality", "same over the last 52 weeks")
    add("log_scale", "scaling", "log(scale): the series' volume level at T")
    add("h", "horizon", "weeks ahead (1-13)")
    return specs


def model_features(cfg: dict) -> list[str]:
    """Features the model uses: all core features + optional groups enabled in config.
    (The cache always stores every feature; this list decides which ones a model sees.)"""
    on = set(cfg["features"].get("optional_features", []))
    return [s["name"] for s in feature_specs(cfg) if s["optional"] is None or s["optional"] in on]


def scaled_features(cfg: dict) -> list[str]:
    return [s["name"] for s in feature_specs(cfg) if s["scaled"]]


CATEGORICAL = ["client", "warehouse", "demand_type"]
ID_COLUMNS = ["unique_id", "origin", "h", "target_date", "product"]
TARGET_COLUMNS = ["target", "target_raw"]  # never model inputs


# ----------------------------------------------------------------------------- helpers
def safe_div(a, b):
    """a / b, NaN where b is 0 or missing (a tree model can treat NaN as its own case)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    out = np.full_like(a, np.nan)
    ok = np.isfinite(b) & (b != 0)
    out[ok] = a[ok] / b[ok]
    return out


def us_holiday_weeks(years) -> dict[str, set]:
    """Monday dates of Thanksgiving week, Cyber week and Christmas week for each year."""
    out = {"thanksgiving": set(), "cyber": set(), "christmas": set()}
    for y in years:
        nov1 = pd.Timestamp(year=y, month=11, day=1)
        thanksgiving = nov1 + pd.Timedelta(days=(3 - nov1.day_of_week) % 7 + 21)  # 4th Thursday
        tg_monday = thanksgiving - pd.Timedelta(days=3)
        out["thanksgiving"].add(tg_monday)
        out["cyber"].add(tg_monday + WEEK)  # the week that starts on Cyber Monday
        xmas = pd.Timestamp(year=y, month=12, day=25)
        out["christmas"].add(xmas - pd.Timedelta(days=xmas.day_of_week))
    return out


def _nan_window(W: np.ndarray, k: int):
    """Last k columns plus the count of observed (post first sale) weeks in them."""
    sub = W[:, -k:]
    return sub, np.sum(~np.isnan(sub), axis=1)


def _row_stat(sub: np.ndarray, fn) -> np.ndarray:
    """Apply a nan-aware numpy reducer row-wise; rows that are all NaN give NaN (no warnings)."""
    out = np.full(sub.shape[0], np.nan)
    ok = ~np.all(np.isnan(sub), axis=1)
    if ok.any():
        out[ok] = fn(sub[ok], axis=1)
    return out


def _trim(hist: pd.DataFrame) -> pd.DataFrame:
    """Drop weeks before each series' first sale (no-op on already-trimmed data)."""
    has_sold = hist["y"].gt(0).groupby(hist["unique_id"]).cummax()
    return hist.loc[has_sold]


# ----------------------------------------------------------------------------- level-shift restart
def group_shift_dates_upto(weekly: pd.DataFrame, T, cfg: dict) -> pd.DataFrame:
    """(unique_id, shift_ds) of warehouse/client level-shift events detected from data <= T."""
    hist = _trim(weekly[weekly["ds"] <= pd.Timestamp(T)][["unique_id", "client", "warehouse", "ds", "y"]])
    d = detect_level_shifts(hist.copy(), cfg)
    return group_shift_dates(d, level_shift_group_events(d, cfg), cfg)


def _restart_after_group_shift(hist: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Keep only weeks from the latest group level-shift event of each series (then re-trim).

    After a warehouse-wide jump, the old weeks describe a different business; using them for lags,
    means, scale and seasonality mixes two levels. Detection uses data <= T only.
    """
    dates = group_shift_dates_upto(hist, hist["ds"].max(), cfg)
    if dates.empty:
        return hist
    start = hist["unique_id"].map(dates.groupby("unique_id")["shift_ds"].max())
    return _trim(hist[start.isna() | (hist["ds"] >= start)])


def drop_rows_crossing_shift(train: pd.DataFrame, dates: pd.DataFrame) -> pd.DataFrame:
    """Drop training rows whose (origin, target_date] window contains a group level-shift event.

    Such rows pair a pre-shift history with a post-shift target (the fold B blow-up). Their targets
    are known at training time, so dropping them uses no information from the forecast period.
    """
    if dates.empty:
        return train
    m = train[["unique_id", "origin", "target_date"]].reset_index().merge(dates, on="unique_id")
    bad = m.loc[(m["shift_ds"] > m["origin"]) & (m["shift_ds"] <= m["target_date"]), "index"].unique()
    return train.drop(index=bad)


# ----------------------------------------------------------------------------- per-origin features
def _price_at_origin(recent: pd.DataFrame, T: pd.Timestamp, cfg: dict) -> pd.DataFrame:
    """relative_price at T and trusted-price share over 13 weeks, from data <= T only."""
    d = clean_price(recent[["unique_id", "client", "ds", "y", "price_raw"]].rename(columns={"price_raw": "price"}))
    # Untrusted prices (client 0 before its cut-off date) become missing, then re-forward-fill.
    ignore = pd.Series(False, index=d.index)
    for client, date in cfg["price_shift"].get("ignore_prices_before", {}).items():
        ignore |= d["client"].eq(client) & d["ds"].lt(pd.Timestamp(date))
    d.loc[ignore, "price_valid"] = np.nan
    d["price_trusted"] = d["price_valid"].notna()
    d["price_clean"] = d["price_valid"].groupby(d["unique_id"]).ffill()
    d, _ = flag_price_level_shift(d, cfg)
    d = add_relative_price(d, cfg)
    at_t = d[d["ds"].eq(T)].set_index("unique_id")["relative_price"]
    share = d[d["ds"] > T - 13 * WEEK].groupby("unique_id")["price_trusted"].mean()
    return pd.DataFrame({"relative_price": at_t, "price_observed_share_13": share})


def _flags_at_origin(recent: pd.DataFrame, T: pd.Timestamp, cfg: dict) -> pd.DataFrame:
    """Smoothed / stockout shares, with both flags recomputed on data <= T.

    Recomputing matters: in the cleaned file a zero run that started before T and ended after T is
    a 'closed' run (so possibly a stockout), but at T nobody knows yet whether it will end.
    """
    c = {**cfg, "smoothing": {**cfg["smoothing"], "switch_date": cfg["features"]["switch_date"]}}
    d, _ = flag_smoothed_weeks(recent[["unique_id", "ds", "y"]].copy(), c)
    d = flag_suspected_stockouts(d, c)
    out = {}
    for k in (13, 52):
        w = d[d["ds"] > T - k * WEEK].groupby("unique_id")
        out[f"smoothed_share_{k}"] = w["is_smoothed"].mean()
        out[f"stockout_share_{k}"] = w["is_suspected_stockout"].mean()
    return pd.DataFrame(out)


def _demand_type(W52: np.ndarray, cfg: dict) -> np.ndarray:
    """Syntetos-Boylan class from the last 52 weeks (post first sale) at the origin."""
    c = cfg["classification"]
    n = np.sum(~np.isnan(W52), axis=1)
    nz = np.where(W52 > 0, W52, np.nan)
    n_nz = np.sum(~np.isnan(nz), axis=1)
    adi = safe_div(n, n_nz)
    cv2 = safe_div(_row_stat(nz, np.nanstd), _row_stat(nz, np.nanmean)) ** 2
    intermittent, variable = adi >= c["adi_threshold"], cv2 >= c["cv2_threshold"]
    out = np.select([~intermittent & ~variable, ~intermittent & variable, intermittent & ~variable],
                    ["smooth", "erratic", "intermittent"], default="lumpy")
    return np.where(n_nz == 0, "no_sales", out)


def _features_at_origin(base: pd.DataFrame, T: pd.Timestamp, horizons, cfg: dict) -> pd.DataFrame:
    f = cfg["features"]
    L = f["season_length"]
    hist = _trim(base[base["ds"] <= T])  # <-- the only data features may use
    if cfg["level_shift"].get("restart_after_shift"):
        hist = _restart_after_group_shift(hist, cfg)
    ids = pd.Index(hist["unique_id"].unique()).sort_values()  # series with first sale <= T
    if len(ids) == 0:
        return pd.DataFrame()
    meta = hist.groupby("unique_id")[["client", "warehouse", "product"]].first().reindex(ids)

    # Wide matrix of the last L+1 weeks: column j is week T - (L - j); last column = T.
    # NaN = before first sale (not zero demand).
    # EXTRA extra weeks before that, for last year's level (13 weeks ending at T - L).
    EXTRA = 13
    weeks = pd.date_range(T - (L + EXTRA) * WEEK, T, freq=WEEK)
    window = hist[hist["ds"] >= weeks[0]]
    W_ext = window.pivot(index="unique_id", columns="ds", values="y").reindex(index=ids, columns=weeks).to_numpy(float)
    W = W_ext[:, EXTRA:]  # weeks T-L .. T (column j = week T - (L - j))

    s = pd.DataFrame(index=ids)
    for k in f["lags"]:
        s[f"lag_{k}"] = W[:, L - (k - 1)]
    means = {}
    for w in f["roll_windows"]:
        sub, _ = _nan_window(W, w)
        means[w] = _row_stat(sub, np.nanmean)
        s[f"roll_mean_{w}"] = means[w]
        s[f"roll_std_{w}"] = _row_stat(sub, np.nanstd)
    s[f"roll_max_{f['max_window']}"] = _row_stat(_nan_window(W, f["max_window"])[0], np.nanmax)
    s["trend_4_13"] = safe_div(means[4], means[13]) if 4 in means and 13 in means else np.nan
    s["trend_13_52"] = safe_div(means[13], means[52]) if 13 in means and 52 in means else np.nan
    for k in (13, 52):
        sub, n = _nan_window(W, k)
        s[f"zero_share_{k}"] = safe_div(np.sum(sub == 0, axis=1), n)

    g = hist.groupby("unique_id")
    first_sale = g["ds"].min().reindex(ids)
    last_sale = hist[hist["y"] > 0].groupby("unique_id")["ds"].max().reindex(ids)
    s["age_weeks"] = ((T - first_sale).dt.days // 7).to_numpy()
    s["weeks_since_last_sale"] = ((T - last_sale).dt.days // 7).to_numpy()

    # Two candidate scales (chosen later by apply_scaling; both use data <= T only):
    #  mean13  : mean of the last 13 weeks; if 0, mean since first sale; floor.
    #  max13_52: max(mean of last 13 weeks, mean of last 52 weeks since first sale); floor.
    #            A series that went quiet keeps a realistic scale, so a restart doesn't become a
    #            scaled target of 1,000+.
    m13 = _row_stat(_nan_window(W, f["scale_window"])[0], np.nanmean)
    m52 = _row_stat(_nan_window(W, 52)[0], np.nanmean)
    since_first = (g["y"].sum() / g["y"].size()).reindex(ids).to_numpy()
    floor = f["scale_floor"]
    s["scale_mean13"] = np.maximum(np.nan_to_num(np.where(m13 > 0, m13, since_first), nan=0.0), floor)
    s["scale_max13_52"] = np.maximum(np.nan_to_num(np.fmax(m13, m52), nan=0.0), floor)

    # Hierarchy (at T). Sums treat pre-first-sale weeks as 0 units.
    sum13 = np.nansum(W[:, -13:], axis=1)
    client = meta["client"].to_numpy()
    s["prod_mean_13"] = pd.Series(sum13 / 13, index=ids).groupby(meta["product"]).transform("sum")
    cw = pd.DataFrame(np.nan_to_num(W[:, -13:]), index=ids).groupby(client).sum()  # client x 13 weeks
    client_trend = pd.Series(safe_div(cw.iloc[:, -4:].mean(axis=1), cw.mean(axis=1)), index=cw.index)
    s["client_trend_4_13"] = client_trend.reindex(client).to_numpy()
    s["client_share_13"] = safe_div(sum13, cw.sum(axis=1).reindex(client).to_numpy())

    # Recomputed flags and price features use a bounded recent history (all <= T).
    recent = hist[hist["ds"] > T - FLAG_HISTORY_WEEKS * WEEK]
    s = s.join(_price_at_origin(recent, T, cfg)).join(_flags_at_origin(recent, T, cfg))
    s["demand_type"] = _demand_type(W[:, -f["demand_type_window"]:], cfg)
    s = s.join(meta)

    # Expand to one row per horizon and add target-week features.
    hs = np.asarray(list(horizons))
    rows = s.loc[np.repeat(ids.to_numpy(), len(hs))].reset_index().rename(columns={"index": "unique_id"})
    rows["h"] = np.tile(hs, len(ids))
    rows["origin"] = T
    rows["target_date"] = T + pd.to_timedelta(rows["h"] * 7, unit="D")
    # Seasonal lag: target week - 52 is column h of W; the 3-week mean uses columns h-1..h+1.
    ri = np.repeat(np.arange(len(ids)), len(hs))
    hh = rows["h"].to_numpy()
    # (h + 1 <= L for every h used here, so these columns are always <= T.)
    assert hh.max() + 1 <= L, "horizon too long for the seasonal look-back"
    rows["seas_lag_52"] = W[ri, hh]
    three = np.stack([W[ri, hh + d] for d in (-1, 0, 1)], axis=1)
    rows["seas_mean_3"] = _row_stat(three, np.nanmean)

    td = rows["target_date"]
    rows["tw_week_of_year"] = td.dt.isocalendar().week.astype(int).to_numpy()
    rows["tw_month"] = td.dt.month
    hol = us_holiday_weeks(range(td.dt.year.min(), td.dt.year.max() + 1))
    rows["tw_thanksgiving_week"] = td.isin(hol["thanksgiving"]).astype(int)
    rows["tw_cyber_week"] = td.isin(hol["cyber"]).astype(int)
    rows["tw_christmas_week"] = td.isin(hol["christmas"]).astype(int)

    # Signed distance (weeks) to the nearest Thanksgiving / Christmas week, clipped.
    hol_wide = us_holiday_weeks(range(td.dt.year.min() - 1, td.dt.year.max() + 2))
    for key, col in [("thanksgiving", "tw_weeks_to_thanksgiving"), ("christmas", "tw_weeks_to_christmas")]:
        anchors = np.array(sorted(hol_wide[key]), dtype="datetime64[ns]")
        d = (anchors[None, :] - td.to_numpy()[:, None]).astype("timedelta64[D]").astype(int) // 7
        rows[col] = np.clip(d[np.arange(len(d)), np.abs(d).argmin(axis=1)], -13, 13)

    # Last-year uplift: target weeks last year (T+h-53..T+h-51) vs last year's level (T-64..T-52).
    # Every week used is <= T (h + 1 <= L), so no future data.
    ly_level = _row_stat(W_ext[:, 1:EXTRA + 1], np.nanmean)  # W_ext column j = week T-(L+EXTRA-j): 1..13 = T-64..T-52
    rows["ly_uplift_series"] = safe_div(rows["seas_mean_3"].to_numpy(float), ly_level[ri])
    C = pd.DataFrame(np.nan_to_num(W_ext), index=ids).groupby(client).sum().to_numpy()  # client x weeks
    cidx = pd.Index(sorted(set(client))).get_indexer(client)
    c_level = C[:, 1:EXTRA + 1].mean(axis=1)
    c_num = np.stack([C[:, EXTRA + hh_ + d] for hh_ in hs for d in (-1, 0, 1)], axis=1).reshape(len(C), len(hs), 3).mean(axis=2)
    rows["ly_uplift_client"] = safe_div(c_num[cidx[ri], np.tile(np.arange(len(hs)), len(ids))], c_level[cidx[ri]])

    # Sales-based features are stored unscaled under `<name>_raw`; apply_scaling() makes the scaled ones.
    rows = rows.rename(columns={c: f"{c}_raw" for c in scaled_features(cfg)})
    tgt = _targets(base, T, ids, hs)
    return rows.merge(tgt, on=["unique_id", "h"], how="left")


def _targets(base: pd.DataFrame, T: pd.Timestamp, ids: pd.Index, hs: np.ndarray) -> pd.DataFrame:
    """y at T + h. The ONLY function that reads weeks after the origin. NaN if not in the data."""
    fut = base[(base["ds"] > T) & (base["ds"] <= T + int(hs.max()) * WEEK) & base["unique_id"].isin(ids)]
    out = fut[["unique_id", "ds", "y"]].rename(columns={"y": "target_raw"})
    out["h"] = ((out["ds"] - T).dt.days // 7).astype(int)
    return out[out["h"].isin(hs)][["unique_id", "h", "target_raw"]]


# ----------------------------------------------------------------------------- public
SCALE_METHODS = ("mean13", "max13_52")


def make_raw_features(weekly_clean: pd.DataFrame, origins, horizons=range(1, 14), config: dict | None = None) -> pd.DataFrame:
    """Unscaled features: one row per (unique_id, origin, h), with both candidate scales and target_raw.

    This is what the feature cache stores. Only the raw input columns are used; any precomputed
    flags in the input are ignored on purpose (they were computed with future data).
    """
    if config is None:
        from src.config import get_config
        config = get_config()
    missing = [c for c in BASE_COLUMNS if c not in weekly_clean.columns]
    if missing:
        raise ValueError(f"missing input columns: {missing}")
    base = weekly_clean[BASE_COLUMNS].sort_values(["unique_id", "ds"], kind="stable").reset_index(drop=True)

    frames = []
    for T in origins:
        fr = _features_at_origin(base, pd.Timestamp(T), horizons, config)
        num = fr.select_dtypes("number").columns.difference(["h", "client", "warehouse", "product"])
        frames.append(fr.astype({c: "float32" for c in num}))  # halves memory
    out = pd.concat(frames, ignore_index=True)
    out["h"] = out["h"].astype("int16")
    out["demand_type"] = out["demand_type"].astype(str)
    return out


def apply_scaling(raw: pd.DataFrame, config: dict, method: str | None = None, keep_raw: bool = True) -> pd.DataFrame:
    """Pick a scale, then add scale, log_scale, the scaled sales features and the scaled target.

    Adds columns to `raw` in place and returns it (always use the return value). keep_raw=False drops the
    `<name>_raw` columns afterwards to save memory (the model never uses them).
    """
    method = method or config["features"]["scale_method"]
    if method not in SCALE_METHODS:
        raise ValueError(f"unknown scale method {method!r}; use one of {SCALE_METHODS}")
    raw["scale"] = raw[f"scale_{method}"]
    raw["log_scale"] = np.log(raw["scale"]).astype("float32")
    for col in scaled_features(config):
        raw[col] = (raw[f"{col}_raw"] / raw["scale"]).astype("float32")
    if "target_raw" in raw:
        raw["target"] = (raw["target_raw"] / raw["scale"]).astype("float32")
    if not keep_raw:
        raw = raw.drop(columns=[f"{c}_raw" for c in scaled_features(config)])
    return raw


def set_categories(df: pd.DataFrame, categories: dict | None = None) -> dict:
    """Convert categoricals in place with FIXED category lists (train and predict must match).

    categories=None: derive them from df (use on training rows) and return them for reuse.
    Unseen values at predict time become NaN, which LightGBM treats as missing.
    """
    if categories is None:
        categories = {"client": sorted(pd.unique(df["client"].astype(int))),
                      "warehouse": sorted(pd.unique(df["warehouse"].astype(int))),
                      "demand_type": ["smooth", "erratic", "intermittent", "lumpy", "no_sales"]}
    for c, cats in categories.items():
        df[c] = pd.Categorical(df[c].astype(int) if c != "demand_type" else df[c].astype(str), categories=cats)
    return categories


def make_features(weekly_clean: pd.DataFrame, origins, horizons=range(1, 14), config: dict | None = None,
                  scale_method: str | None = None) -> pd.DataFrame:
    """Scaled feature matrix (raw features + apply_scaling + categoricals), one row per (unique_id, origin, h)."""
    if config is None:
        from src.config import get_config
        config = get_config()
    out = apply_scaling(make_raw_features(weekly_clean, origins, horizons, config), config, scale_method)
    for c in CATEGORICAL:
        out[c] = out[c].astype("category")
    raw = [f"{c}_raw" for c in scaled_features(config)]
    all_feats = [sp["name"] for sp in feature_specs(config)]  # includes optional ones (model picks via model_features)
    cols = dict.fromkeys(ID_COLUMNS + all_feats + ["scale", "scale_mean13", "scale_max13_52"]
                         + raw + TARGET_COLUMNS)  # h is both id and feature
    return out[list(cols)]

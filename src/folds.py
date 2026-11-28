"""Automatic rolling backtest folds (config backtest.auto_folds), used when backtest.folds is empty.

Origins step back from the end of the training data: T = last - h, last - h - step, ...
Each fold forecasts the h weeks after its origin. A fold whose window contains a major holiday
(calendar distance_events, e.g. Thanksgiving / Christmas) is kept only if the same window one year
earlier is clean weekly data (inside the data and on or after the smoothing switch date), because
the seasonal features read last year's values. Otherwise it is skipped and the next one back is used.
Every fold also needs enough history for training and conformal calibration (2 x h weeks of
training origins after the training start).
"""
import pandas as pd

from src import calendar

WEEK = pd.Timedelta(weeks=1)


def auto_folds(weekly: pd.DataFrame, cfg: dict, switch_date=None) -> tuple[dict, list[dict]]:
    """Returns ({name: origin}, log) with folds named F1 (oldest) .. Fn. `log` explains every candidate."""
    a = cfg["backtest"].get("auto_folds") or {}
    n, step = int(a.get("n", 4)), int(a.get("step_weeks", 13))
    h = max(cfg["features"]["horizons"])
    season = cfg["features"]["season_length"]
    first, last = weekly["ds"].min(), weekly["ds"].max()
    sd = pd.Timestamp(switch_date) if switch_date is not None else None
    min_origin = first + season * WEEK + 2 * h * WEEK  # training start + targets + calibration window
    events = calendar.distance_events(cfg["dataset"]["calendar"])
    years = range(first.year - 1, last.year + 2)
    holiday_weeks = {w for e, ws in calendar.event_weeks(cfg["dataset"]["calendar"], years).items() if e in events for w in ws}

    chosen, log = [], []
    T = last - h * WEEK
    while len(chosen) < n and T >= min_origin:
        window = [T + k * WEEK for k in range(1, h + 1)]
        holiday = any(w in holiday_weeks for w in window)
        prev_start = window[0] - season * WEEK
        clean_prev = prev_start >= first and (sd is None or prev_start >= sd)
        keep = (not holiday) or clean_prev
        log.append({"origin": T.date(), "window": f"{window[0].date()} .. {window[-1].date()}", "holiday": holiday,
                    "previous_year_clean": clean_prev, "kept": keep})
        if keep:
            chosen.append(T)
        T -= step * WEEK
    if len(chosen) < n:
        log.append({"note": f"only {len(chosen)} of {n} folds fit after {min_origin.date()} (training start + {2 * h} weeks)"})
    chosen = sorted(chosen)
    return {f"F{i + 1}": t for i, t in enumerate(chosen)}, log

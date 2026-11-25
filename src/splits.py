"""Origins for training and validating the global direct model. Dates only: no data is read here.

An origin T is the last week the model is allowed to see. Each origin yields one row per series
and horizon h, with target y at T + h.
"""
import pandas as pd


def _origins(start, end, step_weeks: int) -> list[pd.Timestamp]:
    return list(pd.date_range(pd.Timestamp(start), pd.Timestamp(end), freq=pd.Timedelta(weeks=step_weeks)))


def inner_validation_split(cfg: dict) -> dict:
    """Train origins every 2 weeks from origin_start; validate at inner_valid_origin.

    Training targets must stay <= the validation origin, otherwise training rows would contain the
    very weeks we validate on. So a training origin T is kept only if T + max(h) <= validation origin
    (this drops the last 13 weeks of origins).
    """
    s = cfg["splits"]
    valid_origin = pd.Timestamp(s["inner_valid_origin"])
    h_max = max(cfg["features"]["horizons"])
    candidates = _origins(s["origin_start"], valid_origin, s["origin_step_weeks"])
    train = [t for t in candidates if t + pd.Timedelta(weeks=h_max) <= valid_origin]
    return {"train_origins": train, "valid_origin": valid_origin,
            "valid_target_start": valid_origin + pd.Timedelta(weeks=1),
            "valid_target_end": valid_origin + pd.Timedelta(weeks=h_max)}


def phase1_split(cfg: dict) -> dict:
    """For the Phase 1 run (later): same rule, with the forecast origin at the end of Phase 0.

    Training origins run up to 2023-07-03, so every training target is <= 2023-10-02 (Phase 0).
    The forecast origin's targets are Phase 1 weeks, which this code never reads.
    """
    s = cfg["splits"]
    forecast_origin = pd.Timestamp(s["phase1_forecast_origin"])
    h_max = max(cfg["features"]["horizons"])
    candidates = _origins(s["origin_start"], forecast_origin, s["origin_step_weeks"])
    train = [t for t in candidates if t + pd.Timedelta(weeks=h_max) <= forecast_origin]
    return {"train_origins": train, "forecast_origin": forecast_origin}

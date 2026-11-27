"""Holiday calendars as Monday-dated weeks. The dataset config picks one (dataset.calendar: us | au).

Every event maps to a set of Monday dates: the weeks (Monday to Sunday) in which the event's sales
happen. Features flag the TARGET week (tw_<event>_week) and, optionally, the signed distance to the
nearest occurrence of the main events (tw_weeks_to_<event>).

(Module name: imported as `src.calendar`, so it never shadows Python's built-in `calendar`.)
"""
import pandas as pd

DAY = pd.Timedelta(days=1)
WEEK = pd.Timedelta(weeks=1)


def monday(d: pd.Timestamp) -> pd.Timestamp:
    """Monday of the week (Mon..Sun) that contains d."""
    d = pd.Timestamp(d).normalize()
    return d - d.day_of_week * DAY


def nth_weekday(year: int, month: int, weekday: int, n: int) -> pd.Timestamp:
    """n-th given weekday (Mon=0 .. Sun=6) of a month, e.g. 4th Thursday of November."""
    first = pd.Timestamp(year=year, month=month, day=1)
    return first + ((weekday - first.day_of_week) % 7) * DAY + (n - 1) * WEEK


def thanksgiving(year: int) -> pd.Timestamp:
    return nth_weekday(year, 11, 3, 4)  # 4th Thursday of November


# Each event: (name, description, function year -> list of Monday dates)
def _us(year):
    tg = monday(thanksgiving(year))
    return {
        "thanksgiving": [tg],                                   # Thanksgiving + Black Friday week
        "cyber": [tg + WEEK],                                   # the week that starts on Cyber Monday
        "christmas": [monday(pd.Timestamp(year=year, month=12, day=25))],
    }


def _au(year):
    bf = thanksgiving(year) + DAY                               # AU retailers follow the US Black Friday date
    june = pd.date_range(f"{year}-06-01", f"{year}-06-30", freq="D")
    return {
        "black_friday": [monday(bf)],
        "cyber_monday": [monday(bf) + WEEK],                    # the week that starts on Cyber Monday
        "click_frenzy": [monday(nth_weekday(year, 11, 1, 2))],  # Main Event: 2nd Tuesday of November (2022-24)
        "christmas": [monday(pd.Timestamp(year=year, month=12, day=25))],
        "boxing_day": [monday(pd.Timestamp(year=year, month=12, day=26))],
        "eofy": sorted({monday(d) for d in june}),              # end-of-financial-year sales: weeks touching June
        "mothers_day": [monday(nth_weekday(year, 5, 6, 2))],    # week ending on the 2nd Sunday of May
        "fathers_day": [monday(nth_weekday(year, 9, 6, 1))],    # week ending on the 1st Sunday of September
    }


CALENDARS = {
    "us": {"events": _us, "distance_events": ["thanksgiving", "christmas"],
           "descriptions": {
               "thanksgiving": "target week contains US Thanksgiving (and Black Friday)",
               "cyber": "target week starts on Cyber Monday",
               "christmas": "target week contains 25 December"}},
    "au": {"events": _au, "distance_events": ["black_friday", "christmas"],
           "descriptions": {
               "black_friday": "target week contains Black Friday (day after US Thanksgiving)",
               "cyber_monday": "target week starts on Cyber Monday",
               "click_frenzy": "target week contains the Click Frenzy Main Event (2nd Tuesday of November)",
               "christmas": "target week contains 25 December",
               "boxing_day": "target week contains 26 December (Boxing Day sales)",
               "eofy": "target week overlaps June (end-of-financial-year sales)",
               "mothers_day": "target week ends on Mother's Day (2nd Sunday of May)",
               "fathers_day": "target week ends on Father's Day (1st Sunday of September)"}},
}


def _get(name: str) -> dict:
    if name not in CALENDARS:
        raise ValueError(f"unknown calendar {name!r}; choose one of {sorted(CALENDARS)}")
    return CALENDARS[name]


def event_names(name: str) -> list[str]:
    return list(_get(name)["descriptions"])


def event_description(name: str, event: str) -> str:
    return _get(name)["descriptions"][event]


def distance_events(name: str) -> list[str]:
    return list(_get(name)["distance_events"])


def event_weeks(name: str, years) -> dict[str, set]:
    """{event: set of Monday dates} for the given years."""
    out = {e: set() for e in event_names(name)}
    for y in years:
        for e, weeks in _get(name)["events"](int(y)).items():
            out[e].update(weeks)
    return out

"""Data quality report: reports/data_quality.md + PNG charts in reports/figures/.

Everything here is computed from the Phase 0 (training) cleaning output only.
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

# Validated categorical palette (fixed order) + recessive ink for axes/text.
BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN, VIOLET, RED = (
    "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948")
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
    "axes.spines.top": False, "axes.spines.right": False, "font.size": 9,
    "axes.titlesize": 10, "axes.titlecolor": INK, "axes.titleweight": "bold", "lines.linewidth": 1.5,
    "legend.frameon": False,
})


def _pct(n, d):
    return f"{n:,} ({100 * n / d:.1f}%)" if d else f"{n:,}"


def _save(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def _vline(ax, date, label=None):
    ax.axvline(date, color=INK2, lw=1, ls="--")
    if label:
        ax.annotate(label, (date, 1), xycoords=("data", "axes fraction"), xytext=(4, -12),
                    textcoords="offset points", color=INK2, fontsize=8)


# ----------------------------------------------------------------------------- example picking
def pick_examples(series: pd.DataFrame) -> list[tuple[str, str]]:
    """12 deterministic examples: 2 per category, the highest-volume series that fits each."""
    s = series.dropna(subset=["history_weeks"]).sort_values("volume_rank")
    old = s[~s["is_new"]]
    smoothed_share = s["n_smoothed_weeks"] / s["history_weeks"]
    picks, used = [], set()

    def take(cands, label, k=2):
        for uid in cands["unique_id"]:
            if uid not in used and k > 0:
                picks.append((uid, label)); used.add(uid); k -= 1

    take(old[old["demand_type"].eq("smooth") & ~old["has_price_shift"]
             & ~old["has_suspected_stockout"]].iloc[20:], "smooth")
    take(old[old["demand_type"].eq("intermittent") & old["history_weeks"].ge(120)].iloc[50:], "intermittent")
    take(s.assign(sh=smoothed_share).query("history_weeks >= 150").sort_values("sh", ascending=False), "smoothed era")
    take(s[s["has_price_shift"]], "price shift")
    take(s[s["has_suspected_stockout"]].sort_values(["n_suspected_stockout_weeks", "volume_rank"],
                                                    ascending=[False, True]).query("volume_rank < 3000"),
         "suspected stockout")
    take(s[s["is_new"] & ~s["is_inactive"]], "new")
    return picks


def plot_examples(clean: pd.DataFrame, picks, switch_date, path):
    fig, axes = plt.subplots(4, 3, figsize=(13, 11), sharex=True)
    for ax, (uid, label) in zip(axes.flat, picks):
        d = clean[clean["unique_id"].eq(uid)]
        if label == "price shift":  # the interesting thing here is the price, not the units
            ax.plot(d["ds"], d["price_clean"], color=VIOLET, lw=1.2, label="price (ffilled)")
            obs = d[d["price_observed"]]
            ax.scatter(obs["ds"], obs["price_raw"], s=6, color=VIOLET, zorder=3)
            for sd in d.loc[d["price_shift"], "ds"]:
                ax.axvline(sd, color=RED, lw=1)
            ax.set_ylabel("price")
        else:
            ax.plot(d["ds"], d["y"], color=BLUE, lw=1.2)
            sm = d[d["is_smoothed"]]
            ax.scatter(sm["ds"], sm["y"], s=8, color=ORANGE, zorder=3, label="is_smoothed")
            so = d[d["is_suspected_stockout"]]
            for ds in so["ds"]:
                ax.axvspan(ds - pd.Timedelta(days=3), ds + pd.Timedelta(days=3), color=RED, alpha=0.18, lw=0)
            ax.set_ylabel("units / week")
        ax.axvline(switch_date, color=INK2, lw=0.8, ls="--")
        ax.set_title(f"{label}: {uid}", loc="left", fontsize=9)
        ax.set_ylim(bottom=0)
        ax.tick_params(axis="x", labelrotation=30)
    handles = [plt.Line2D([], [], color=BLUE, label="units"),
               plt.Line2D([], [], marker="o", ls="", color=ORANGE, label="is_smoothed week"),
               plt.Rectangle((0, 0), 1, 1, color=RED, alpha=0.18, label="suspected stockout"),
               plt.Line2D([], [], color=VIOLET, label="price (price-shift panels)"),
               plt.Line2D([], [], color=RED, label="detected price shift"),
               plt.Line2D([], [], color=INK2, ls="--", label="weekly-data switch")]
    fig.legend(handles=handles, loc="lower center", ncol=6, fontsize=8)
    fig.suptitle("Example series (Phase 0, after leading-zero trim)", x=0.01, ha="left",
                 fontweight="bold", color=INK)
    fig.tight_layout(rect=(0, 0.03, 1, 0.98))
    fig.savefig(path, dpi=130)
    plt.close(fig)


def _events_table(ev):
    if ev.empty:
        return "(none)"
    lines = ["| level | id | week | active series | shifted | share | direction |", "|---|---|---|---|---|---|---|"]
    for r in ev.sort_values(["ds", "group_type"]).itertuples():
        lines.append(f"| {r.group_type} | {r.group_id} | {r.ds.date()} | {r.n_active} | {r.n_shifted} | {r.share:.0%} | {r.direction} |")
    return "\n".join(lines)


def _with_valid(clean):
    """Rebuild the 'valid observed price' helper column (not saved in the weekly output)."""
    d = clean[["unique_id", "ds", "price_raw", "price_observed", "price_zero"]].copy()
    d["price_valid"] = d["price_raw"].where(d["price_observed"] & ~d["price_zero"] & d["price_raw"].gt(0))
    return d


# ----------------------------------------------------------------------------- main report
def write_report(clean, series, diag, raw_phase0, cfg, reports_dir):
    reports_dir = Path(reports_dir)
    fig_dir = reports_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    sw = diag["switch"]
    switch_date = sw["switch_date"]
    rate = diag["repeat_rate"]
    n_series, n_rows = len(series), len(clean)
    raw_rows = len(raw_phase0)

    # --- evidence that the old era is monthly totals spread over weeks:
    # in the smoothed era a value should change mostly when a new month starts.
    new_month = pd.Series(rate.index.month != (rate.index - pd.Timedelta(days=7)).month, index=rate.index)
    pre = rate.index < switch_date
    rate_newmonth_pre = rate[pre & new_month].mean()
    rate_midmonth_pre = rate[pre & ~new_month].mean()
    rate_newmonth_post = rate[~pre & new_month].mean()
    rate_midmonth_post = rate[~pre & ~new_month].mean()

    # --- chart: repeat rate + switch
    fig, ax = plt.subplots(figsize=(10, 3.4))
    ax.plot(rate.index, rate.values, color=BLUE)
    ax.hlines([sw["rate_before"]], rate.index[0], switch_date, color=ORANGE, lw=1.5)
    ax.hlines([sw["rate_after"]], switch_date, rate.index[-1], color=ORANGE, lw=1.5)
    _vline(ax, switch_date, f"switch {switch_date.date()}")
    ax.set_title("Share of positive weekly sales identical to the previous week", loc="left")
    ax.set_ylabel("repeat rate")
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    _save(fig, fig_dir / "repeat_rate.png")

    # --- chart: total weekly units (raw, untrimmed: trimming only drops zeros so totals match)
    tot = raw_phase0.groupby("ds")["y"].sum()
    fig, ax = plt.subplots(figsize=(10, 3.4))
    ax.plot(tot.index, tot.values, color=BLUE)
    _vline(ax, switch_date, "weekly-data switch")
    ax.set_title("Total units sold per week, all series (Phase 0)", loc="left")
    ax.set_ylabel("units")
    ax.set_ylim(bottom=0)
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v / 1000:.0f}k"))
    _save(fig, fig_dir / "total_weekly_units.png")

    # --- chart: active series per week (series past their first sale)
    active = clean.groupby("ds").size()
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.plot(active.index, active.values, color=BLUE)
    _vline(ax, switch_date)
    ax.set_title("Series with history (after leading-zero trim) per week", loc="left")
    ax.set_ylim(bottom=0)
    _save(fig, fig_dir / "active_series.png")

    # --- chart: history length
    hist = series["history_weeks"].dropna()
    fig, ax = plt.subplots(figsize=(7, 3.2))
    ax.hist(hist, bins=np.arange(0, hist.max() + 6, 5), color=BLUE, edgecolor=SURFACE, linewidth=1)
    ax.axvline(cfg["classification"]["new_series_weeks"], color=INK2, ls="--", lw=1)
    ax.set_title("Weeks of history per series (after trim)", loc="left")
    ax.set_xlabel("weeks"); ax.set_ylabel("series")
    _save(fig, fig_dir / "history_length.png")

    # --- chart: volume concentration (cumulative share of units vs share of series)
    cum = series.sort_values("volume_rank")["cum_volume_share"].to_numpy()
    xs = np.arange(1, len(cum) + 1) / len(cum)
    fig, ax = plt.subplots(figsize=(5.5, 4))
    ax.plot(xs, cum, color=BLUE)
    ax.plot([0, 1], [0, 1], color=GRID, lw=1)
    for q in (0.01, 0.10, 0.20):
        v = cum[int(np.ceil(q * len(cum))) - 1]
        ax.scatter([q], [v], color=BLUE, s=16, zorder=3)
        ax.annotate(f"top {q:.0%}: {v:.0%}", (q, v), xytext=(8, -4), textcoords="offset points", color=INK)
    ax.set_title("Volume concentration", loc="left")
    ax.set_xlabel("share of series (ranked by units)"); ax.set_ylabel("share of units")
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    _save(fig, fig_dir / "volume_concentration.png")

    # --- chart: price-shift timing, client 0 vs others (two panels, same scale = no dual axis)
    ev = diag["price_shift_events"].copy()
    ev["client0"] = ev["unique_id"].str.split("_").str[0].eq("0")
    months = pd.period_range(clean["ds"].min(), clean["ds"].max(), freq="M")
    by_m = ev.groupby([ev["shift_date"].dt.to_period("M"), "client0"]).size().unstack(fill_value=0)
    by_m = by_m.reindex(months, fill_value=0).reindex(columns=[True, False], fill_value=0)
    fig, axes = plt.subplots(2, 1, figsize=(10, 4.2), sharex=True, sharey=True)
    for ax, col, name, colr in [(axes[0], True, "client 0", VIOLET), (axes[1], False, "all other clients", BLUE)]:
        ax.bar(by_m.index.to_timestamp(), by_m[col], width=24, color=colr)
        _vline(ax, switch_date)
        ax.set_title(f"Detected price level shifts per month: {name}", loc="left")
    _save(fig, fig_dir / "price_shift_timing.png")

    # --- examples
    picks = pick_examples(series)
    plot_examples(clean, picks, switch_date, fig_dir / "example_series.png")

    # --- numbers for the markdown
    lead = series["n_leading_zero_weeks"]
    has_lead = lead.gt(0)
    pc = diag["price_check"]
    w_pre = clean["is_smoothed_era"]
    t = series
    top1 = t.loc[t["volume_rank"] <= int(np.ceil(0.01 * n_series)), "volume_share"].sum()
    top10 = t.loc[t["volume_rank"] <= int(np.ceil(0.10 * n_series)), "volume_share"].sum()
    top20 = t.loc[t["volume_rank"] <= int(np.ceil(0.20 * n_series)), "volume_share"].sum()
    mix = pd.DataFrame({
        "full history": t["demand_type"].value_counts(),
        "weekly era only": t["demand_type_weekly_era"].value_counts(),
    }).fillna(0).astype(int)
    mix_units = t.groupby("demand_type")["total_units"].sum() / t["total_units"].sum()
    shifted = t[t["has_price_shift"]]
    shift_client0 = shifted["client"].eq(0).sum()
    ev_aug_sep22 = ev["shift_date"].between("2022-08-01", "2022-09-30").sum()
    so = t[t["has_suspected_stockout"]]
    inactive_new = (t["is_inactive"] & t["is_new"]).sum()
    q = hist.quantile([0.1, 0.25, 0.5, 0.75, 0.9]).astype(int)
    rel = clean["relative_price"]

    # Stockout weeks by the length of the zero run they sit in (long runs = off-season / dormancy?).
    so_len = clean.loc[clean["is_suspected_stockout"], "zero_run_len"]
    cap = cfg["stockout"].get("max_zero_run")
    edges, labels = ([0, 8, cap], ["4-8 weeks", f"9-{cap} weeks"]) if cap else (
        [0, 8, 26, np.inf], ["4-8 weeks", "9-26 weeks", ">26 weeks"])
    so_bins = pd.cut(so_len, edges, labels=labels).value_counts().sort_index()

    # Same price-shift rule with min_periods=1, to compare with the looser count seen in earlier EDA.
    from src.cleaning import flag_price_level_shift
    loose_cfg = {**cfg, "price_shift": {**cfg["price_shift"], "min_periods": 1}}
    _, loose_ev = flag_price_level_shift(_with_valid(clean), loose_cfg)
    n_loose = loose_ev["unique_id"].nunique()

    # Weeks before the switch whose repeat rate is already at the weekly floor (partly weekly months?).
    floor_weeks_pre = int((rate[pre] < sw["rate_after"] + 0.02).sum())

    def flag_row(name, weeks, series_n, note=""):
        return f"| `{name}` | {_pct(int(weeks), n_rows) if weeks is not None else '-'} | {_pct(int(series_n), n_series)} | {note} |"

    lines = [
        "# Data quality report: VN1 (Phase 0 training data)",
        "",
        f"Generated by `python -m src.run_cleaning`. Training data: {raw_phase0['ds'].min().date()} to "
        f"{raw_phase0['ds'].max().date()}, {n_series:,} series, {raw_phase0['ds'].nunique()} weeks "
        f"({raw_rows:,} raw rows, {n_rows:,} after trimming leading zeros). "
        "Phase 1 and Phase 2 values are not used anywhere in this report.",
        "",
        "## 1. Flags at a glance",
        "",
        f"Week percentages are out of {n_rows:,} trimmed weeks; series percentages out of {n_series:,} series.",
        "",
        "| flag | weeks | series with ≥1 | note |",
        "|---|---|---|---|",
        f"| leading zeros trimmed | {_pct(int(lead.sum()), raw_rows)} of raw rows | {_pct(int(has_lead.sum()), n_series)} "
        f"| median {lead[has_lead].median():.0f} weeks per affected series; {int(t['never_sold'].sum())} series never sold |",
        flag_row("identical_run", clean["identical_run"].sum(), t["n_identical_run_weeks"].gt(0).sum(),
                 f"≥{cfg['smoothing']['min_run_length']} identical positive weeks, any date"),
        flag_row("is_smoothed", clean["is_smoothed"].sum(), t["n_smoothed_weeks"].gt(0).sum(),
                 f"identical run before {switch_date.date()}"),
        flag_row("is_smoothed_era", w_pre.sum(), clean.loc[w_pre, 'unique_id'].nunique(), f"any week before {switch_date.date()}"),
        flag_row("price_observed", clean["price_observed"].sum(), clean.loc[clean['price_observed'], 'unique_id'].nunique(),
                 "price recorded (only happens when y > 0)"),
        flag_row("price_zero", clean["price_zero"].sum(), t["n_price_zero_weeks"].gt(0).sum(), "price = 0 while y > 0: treated as missing"),
        flag_row("price_filled", clean["price_filled"].sum(), clean.loc[clean['price_filled'], 'unique_id'].nunique(),
                 "price_clean carried forward from an earlier week"),
        flag_row("price_shift", clean["price_shift"].sum(), len(shifted),
                 f"{len(ev)} shift events; {shift_client0} of {len(shifted)} series are client 0"),
        flag_row("is_suspected_stockout", clean["is_suspected_stockout"].sum(), len(so),
                 f"{int(t['n_stockout_runs'].sum()):,} zero runs"),
        flag_row("level_shift", clean["level_shift"].sum(), t["n_level_shifts"].gt(0).sum(),
                 f"4-week mean > {cfg['level_shift']['ratio']:g}x or < 1/{cfg['level_shift']['ratio']:g} of the previous 13 weeks"),
        flag_row("group_level_shift", clean["group_level_shift"].sum(), t["group_level_shift_date"].notna().sum(),
                 f"{len(diag['level_shift_events'])} warehouse/client events: > {cfg['level_shift']['group_share']:.0%} "
                 "of a group's active series shift in the same week"),
        flag_row("is_new", None, t["is_new"].sum(), f"< {cfg['classification']['new_series_weeks']} weeks of history"),
        flag_row("is_inactive", None, t["is_inactive"].sum(),
                 f"no sales in last {cfg['classification']['inactive_weeks']} weeks; {inactive_new} are also new"),
        "",
        f"Price missingness check: price is NaN in {pc['nan_and_zero_sales']:,} weeks, all with zero sales; "
        f"NaN with sales > 0: {pc['nan_but_sales_pos']}; price present with zero sales: {pc['price_but_zero_sales']}. "
        f"**The rule 'price is NaN exactly when sales are 0' {'holds exactly' if pc['holds_exactly'] else 'does NOT hold'}.** "
        "So price is a *transaction* price (revenue / units), not a list price: there is no price for weeks without sales.",
        "",
        f"`relative_price` (price ÷ median of past 13 weeks): median {rel.median():.3f}, "
        f"{(rel < 0.8).mean():.1%} of weeks < 0.8, {(rel > 1.25).mean():.1%} > 1.25, "
        f"{int((rel > 5).sum()):,} weeks > 5 (extreme price outliers left as they are).",
        "",
        "Suspected-stockout weeks by the length of their zero run: "
        + ", ".join(f"{k} {_pct(int(v), len(so_len))}" for k, v in so_bins.items())
        + (f". Zero runs longer than {cap} weeks are not flagged (`stockout.max_zero_run`): they look more like an "
           "off-season or dormancy than a stockout." if cap else
           ". Runs over 26 weeks are more likely an off-season (seasonal items) or dormancy than a stockout; "
           "`stockout.max_zero_run` in the config can exclude them."),
        "",
        "Group level-shift events (structural: a whole warehouse/client changes level at once):",
        "",
        _events_table(diag["level_shift_events"]),
        "",
        "## 2. When does the data become truly weekly?",
        "",
        f"**Estimated switch date: {switch_date.date()}** (first week of genuine weekly data). "
        f"Weeks before it: {int(w_pre.sum()):,} ({w_pre.mean():.1%}) of the trimmed training rows.",
        "",
        "How it was found:",
        "",
        "1. For each week, take all series with positive sales this week and last week, and compute the share "
        "whose value is *exactly* the same as last week (the *repeat rate*).",
        "2. Genuine weekly counts repeat only by chance (mostly 1→1 on slow sellers). Monthly totals spread "
        "evenly over the weeks repeat by construction (7, 7, 7, 7).",
        f"3. A one-changepoint search (best split of the repeat-rate series into two constant levels, least squares) "
        f"puts the break at {switch_date.date()}: repeat rate {sw['rate_before']:.1%} before vs {sw['rate_after']:.1%} after.",
        f"4. Cross-check: before the switch, weeks that start a new calendar month repeat the previous value "
        f"{rate_newmonth_pre:.1%} of the time vs {rate_midmonth_pre:.1%} for mid-month weeks. After it the two are "
        f"{rate_newmonth_post:.1%} and {rate_midmonth_post:.1%}. Values change when the month changes, which is the "
        "signature of monthly totals split across weeks.",
        "",
        "![repeat rate](figures/repeat_rate.png)",
        "",
        f"Caveat: the repeat rate before the switch is ~2× the weekly floor, not 100%. Only part of each series is "
        f"visibly smoothed (where runs are long enough), and after the switch `identical_run` still fires on "
        f"{clean.loc[~w_pre, 'identical_run'].mean():.1%} of weeks, which is the chance-repeat noise floor. "
        f"Before the switch it fires on {clean.loc[w_pre, 'identical_run'].mean():.1%}. "
        f"Also, {floor_weeks_pre} pre-switch weeks (mostly Feb-Jun 2021 and Oct 2021) already sit at the weekly "
        "floor, so the old era is a mix rather than uniformly smoothed. `is_smoothed` (run-level) is the more "
        "precise flag; `is_smoothed_era` is the blunt date cut.",
        "",
        "## 3. Demand types (Syntetos-Boylan, ADI 1.32 / CV² 0.49)",
        "",
        "| demand type | series (full history) | series (weekly era only) | share of units (full) |",
        "|---|---|---|---|",
    ]
    for dt in ["smooth", "erratic", "intermittent", "lumpy", "no_sales"]:
        if dt in mix.index:
            r = mix.loc[dt]
            lines.append(f"| {dt} | {_pct(int(r['full history']), n_series)} | "
                         f"{_pct(int(r['weekly era only']), n_series)} | {mix_units.get(dt, 0):.1%} |")
    lines += [
        "",
        "The smoothed era hides week-to-week variation (CV² looks smaller), so classifications on the full history "
        "lean smoother than on the weekly era alone. `no_sales` in the weekly-era column means the series had "
        "history but sold nothing after the switch.",
        "",
        "## 4. Volume concentration",
        "",
        f"Top 1% of series ({int(np.ceil(0.01 * n_series))}) sell **{top1:.1%}** of units; top 10% sell **{top10:.1%}**; "
        f"top 20% sell {top20:.1%}. The VN1 score weights errors by volume, so the head drives it.",
        "",
        "![volume concentration](figures/volume_concentration.png)",
        "",
        "## 5. History length",
        "",
        f"Weeks of history after trimming: 10th pct {q[0.1]}, 25th {q[0.25]}, median {q[0.5]}, 75th {q[0.75]}, "
        f"90th {q[0.9]} (max possible {raw_phase0['ds'].nunique()}). {_pct(int(t['is_new'].sum()), n_series)} "
        f"series have under {cfg['classification']['new_series_weeks']} weeks.",
        "",
        "![history length](figures/history_length.png)",
        "",
        "![active series](figures/active_series.png)",
        "",
        "## 6. Total weekly units",
        "",
        "![total units](figures/total_weekly_units.png)",
        "",
        "## 7. Price level shifts",
        "",
        f"{len(shifted)} series ({len(shifted) / n_series:.1%}) have ≥1 shift ({len(ev)} events; "
        f"{(ev['ratio'] > 1).mean():.0%} upward). {shift_client0} of them are client 0. "
        f"{ev_aug_sep22} events fall in Aug-Sep 2022, the busiest two months. "
        f"Many client-0 series flip up and down repeatedly (series with ≥3 shifts: "
        f"{int(t['n_price_shifts'].ge(3).sum())}); the flipping stops at the weekly switch. That points to a "
        "price-unit/aggregation artefact in the old era rather than real repricing.",
        "",
        f"Detection rule: trailing {cfg['price_shift']['window']}-week median vs the previous {cfg['price_shift']['window']} weeks, "
        f"ratio > {cfg['price_shift']['ratio_threshold']} or < 1/{cfg['price_shift']['ratio_threshold']}, "
        f"each window needs ≥{cfg['price_shift']['min_periods']} observed prices. "
        f"With `min_periods=1` the count rises to {n_loose} series (close to the ~758 seen in earlier EDA), but then a "
        "single odd price on a slow seller can trigger a 'shift'. `min_periods=4` is the default here. "
        f"The trailing rule fires a median of {((ev['detected_date'] - ev['shift_date']).dt.days / 7).median():.0f} "
        "weeks after the located jump; `relative_price` resets at the detection week (causal), `price_shift` "
        "marks the back-dated jump week (descriptive only).",
        "",
        "![price shift timing](figures/price_shift_timing.png)",
        "",
        "## 8. Example series",
        "",
        "Two per category: " + ", ".join(f"{lab} `{uid}`" for uid, lab in picks) + ".",
        "",
        "![examples](figures/example_series.png)",
        "",
        "## 9. Implications for modelling",
        "",
        f"- **Training window.** {w_pre.mean():.0%} of trimmed training weeks are before {switch_date.date()} and "
        "look like monthly totals spread across weeks. That history is fine for level and yearly seasonality, but "
        "it hides weekly noise. Compare training on everything with down-weighting or dropping the smoothed era.",
        f"- **Start dates differ.** {has_lead.mean():.0%} of series had leading zeros (not yet listed) that were trimmed. "
        f"{t['is_new'].mean():.0%} of series have under a year of history, so a yearly-seasonal model per series "
        "is not possible for them. Pooled/global models or hierarchy-level seasonality are needed.",
        f"- **Intermittency dominates.** {(t['demand_type'].isin(['intermittent', 'lumpy'])).mean():.0%} of series are "
        "intermittent or lumpy, but most units come from smooth/erratic series. Use Croston/TSB-style or zero-aware "
        "baselines for the tail, and judge the overall score on the head.",
        f"- **Inactive series.** {t['is_inactive'].sum():,} series sold nothing in the last 13 weeks. A zero or "
        "near-zero forecast is a strong baseline for them; check how many of them restart in Phase 1 before deciding.",
        f"- **Stockouts.** {len(so):,} series have suspected stockout weeks. Treating those zeros as true demand "
        "biases forecasts down. Options: mask them from the loss, or impute them for training only.",
        "- **Price features.** Price only exists in weeks with sales, so `price_clean` is forward-filled and "
        "`relative_price` is safe to use as a past-only feature. Raw price levels are not comparable across a "
        "detected shift; use `relative_price` or `price_regime` instead.",
        "- **Metric.** The VN1 score adds a bias term, so a systematic under-forecast (e.g. from stockout zeros) "
        "is penalised twice. Track `accuracy` and `bias` separately.",
        "",
    ]
    (reports_dir / "data_quality.md").write_text("\n".join(lines))

# scripts/04_trends.py
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

IN_MERGED = Path("data_processed/merged_daily_mean.csv")
IN_DPBI = Path("data_processed/dpbi.csv")
OUT_DATA = Path("data_processed")
OUT_FIGS = Path("figures")
OUT_DATA.mkdir(parents=True, exist_ok=True)
OUT_FIGS.mkdir(parents=True, exist_ok=True)

POLLUTANTS = ["SO2", "NO2", "O3", "PM25"]

# ---------- numeric trend helpers (for summary table) ----------

def linear_slope(series: pd.Series) -> float:
    """
    Return slope per day using simple least-squares on ordinal dates.
    Even if the visual is rolling mean, this gives a numerical trend measure.
    """
    y = series.dropna()
    if len(y) < 10:
        return np.nan
    x = (y.index - y.index[0]).days.values.astype(float)  # days since start
    m, b = np.polyfit(x, y.values.astype(float), 1)
    return m  # units per day

def kendall_tau(series: pd.Series):
    """
    Return Kendall's tau if scipy is available; else (None, None).
    This is optional; if scipy is not installed the values will be None.
    """
    try:
        from scipy.stats import kendalltau
        y = series.dropna().values
        if len(y) < 10:
            return (None, None)
        x = np.arange(len(y))
        tau, p = kendalltau(x, y)
        return (float(tau), float(p))
    except Exception:
        return (None, None)

# ---------- plotting helpers for rolling means ----------

def rolling_mean(s: pd.Series, window: int, min_periods: int) -> pd.Series:
    """Simple rolling mean wrapper."""
    return s.rolling(window=window, min_periods=min_periods, center=True).mean()

def plot_dpbi_rolling_full(dpbi: pd.Series, window: int = 90, min_periods: int = 30):
    """
    Plot full-period DPBI with a smooth rolling mean trend.
    """
    smooth = rolling_mean(dpbi, window=window, min_periods=min_periods)

    fig, ax = plt.subplots(figsize=(10, 4))
    dpbi.plot(ax=ax, alpha=0.4, label="Daily DPBI")
    smooth.plot(ax=ax, linewidth=2, label=f"{window}-day rolling mean")
    ax.set_title("DPBI trend (rolling mean)")
    ax.set_xlabel("Date")
    ax.set_ylabel("z-score units")
    ax.legend()
    plt.tight_layout()
    fig.savefig(OUT_FIGS / "04_dpbi_trend_smooth.png", dpi=180)
    plt.close(fig)

def plot_dpbi_by_year_panels(dpbi: pd.Series, window: int = 30, min_periods: int = 7):
    """
    Make a multi-panel figure with one subplot per year, stacked vertically.
    Each panel shows daily DPBI and a within-year rolling mean.
    """
    # ensure sorted
    dpbi = dpbi.sort_index()
    years = sorted(dpbi.index.year.unique())

    n_years = len(years)
    if n_years == 0:
        return

    fig, axes = plt.subplots(n_years, 1, figsize=(10, 2.5 * n_years), sharex=False)
    if n_years == 1:
        axes = [axes]  # make iterable

    for ax, year in zip(axes, years):
        s_year = dpbi[dpbi.index.year == year]
        smooth_year = rolling_mean(s_year, window=window, min_periods=min_periods)

        s_year.plot(ax=ax, alpha=0.4, label="Daily DPBI")
        smooth_year.plot(ax=ax, linewidth=2, label=f"{window}-day mean")

        ax.set_title(f"DPBI — {year}")
        ax.set_ylabel("z-score")
        ax.legend(loc="upper right")

    axes[-1].set_xlabel("Date")
    plt.tight_layout()
    fig.savefig(OUT_FIGS / "04_dpbi_trend_year_panels.png", dpi=180)
    plt.close(fig)

# ---------- main script ----------

def main():
    # load merged daily pollutants and DPBI
    merged = pd.read_csv(IN_MERGED, parse_dates=["Datetime"], index_col="Datetime")
    dpbi = pd.read_csv(IN_DPBI, parse_dates=["Datetime"], index_col="Datetime")["DPBI"]

    # numeric trend summary for pollutants + DPBI (still useful for report)
    cols = [c for c in POLLUTANTS if c in merged.columns]
    df = merged[cols].join(dpbi, how="outer")

    rows = []
    for c in cols + ["DPBI"]:
        s = df[c]
        m = linear_slope(s)
        tau, p = kendall_tau(s)
        rows.append({"series": c, "slope_per_day": m, "kendall_tau": tau, "kendall_p": p})

    out = pd.DataFrame(rows)
    out.to_csv(OUT_DATA / "trend_summary.csv", index=False)

    # rolling-mean based visualizations
    plot_dpbi_rolling_full(dpbi)
    plot_dpbi_by_year_panels(dpbi)

    print("Wrote:")
    print(" - data_processed/trend_summary.csv")
    print(" - figures/04_dpbi_trend_smooth.png")
    print(" - figures/04_dpbi_trend_year_panels.png")

if __name__ == "__main__":
    main()

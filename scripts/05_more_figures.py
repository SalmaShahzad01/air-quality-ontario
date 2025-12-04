# scripts/05_more_figures.py

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

IN_MERGED = Path("data_processed/merged_daily_mean.csv")
IN_DPBI = Path("data_processed/dpbi.csv")
OUT_FIGS = Path("figures")
OUT_FIGS.mkdir(parents=True, exist_ok=True)

POLLUTANTS = ["SO2", "NO2", "O3", "PM25"]


def plot_monthly_means(df: pd.DataFrame, cols):
    monthly = df[cols].groupby(df.index.month).mean()
    ax = monthly.plot(figsize=(10, 4))
    ax.set_title("Monthly mean concentrations by pollutant (all years)")
    ax.set_xlabel("Month")
    ax.set_ylabel("ppb / µg m⁻³")
    plt.tight_layout()
    plt.savefig(OUT_FIGS / "05_monthly_means.png", dpi=180)
    plt.close()


def plot_no2_weekday(df: pd.DataFrame):
    if "NO2" not in df.columns:
        return
    wd = df["NO2"].groupby(df.index.dayofweek).mean()
    fig, ax = plt.subplots(figsize=(8, 4))
    wd.plot(kind="bar", ax=ax)
    ax.set_title("NO₂ by weekday (0=Mon ... 6=Sun)")
    ax.set_xlabel("Weekday")
    ax.set_ylabel("NO₂ (mean)")
    plt.tight_layout()
    plt.savefig(OUT_FIGS / "05_no2_weekday.png", dpi=180)
    plt.close()


def plot_dpbi_histogram():
    if not IN_DPBI.exists():
        return
    dpbi = pd.read_csv(IN_DPBI, parse_dates=["Datetime"], index_col="Datetime")["DPBI"]
    fig, ax = plt.subplots(figsize=(8, 4))
    dpbi.dropna().plot(kind="hist", bins=40, ax=ax)
    ax.set_title("DPBI distribution")
    ax.set_xlabel("z-score units")
    plt.tight_layout()
    plt.savefig(OUT_FIGS / "05_dpbi_hist.png", dpi=180)
    plt.close()


def plot_monthly_composition(df: pd.DataFrame, cols):
    daily_sum = df[cols].sum(axis=1)
    share = df[cols].div(daily_sum, axis=0).dropna(how="all")
    mshare = share.groupby(share.index.month).mean()
    ax = mshare.plot(kind="area", stacked=True, figsize=(10, 4))
    ax.set_title("Average monthly pollutant shares (relative composition)")
    ax.set_xlabel("Month")
    ax.set_ylabel("Share")
    plt.tight_layout()
    plt.savefig(OUT_FIGS / "05_monthly_shares.png", dpi=180)
    plt.close()


def plot_pollutants_by_year_panels(df: pd.DataFrame, cols):
    """
    Make a figure with one subplot per year, stacked vertically.
    Each panel shows all pollutants for that year so shapes/levels
    can be compared across years.
    """
    df = df[cols].sort_index()
    years = sorted(df.index.year.unique())
    n_years = len(years)
    if n_years == 0:
        return

    fig, axes = plt.subplots(n_years, 1, figsize=(10, 2.5 * n_years), sharex=False)
    if n_years == 1:
        axes = [axes]

    for ax, year in zip(axes, years):
        df_year = df[df.index.year == year]
        df_year.plot(ax=ax)
        ax.set_title(f"Daily pollutant means — {year}")
        ax.set_ylabel("ppb / µg m⁻³")
        ax.legend(loc="upper right")

    axes[-1].set_xlabel("Date")
    plt.tight_layout()
    plt.savefig(OUT_FIGS / "05_pollutants_by_year_panels.png", dpi=180)
    plt.close()


def main():
    df = pd.read_csv(IN_MERGED, parse_dates=["Datetime"], index_col="Datetime")
    cols = [c for c in POLLUTANTS if c in df.columns]
    df = df[cols].sort_index()

    # 1) Monthly averages across all years
    plot_monthly_means(df, cols)

    # 2) Weekday pattern for NO2
    plot_no2_weekday(df)

    # 3) DPBI histogram
    plot_dpbi_histogram()

    # 4) Monthly composition shares
    plot_monthly_composition(df, cols)

    # 5) NEW: pollutants by year (stacked panels)
    plot_pollutants_by_year_panels(df, cols)

    print("Wrote:")
    print(" - figures/05_monthly_means.png")
    print(" - figures/05_no2_weekday.png (if NO2 present)")
    print(" - figures/05_dpbi_hist.png")
    print(" - figures/05_monthly_shares.png")
    print(" - figures/05_pollutants_by_year_panels.png")


if __name__ == "__main__":
    main()

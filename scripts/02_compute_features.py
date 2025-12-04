# scripts/02_compute_features.py
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

IN = Path("data_processed/merged_daily_mean.csv")
OUT_DATA = Path("data_processed")
OUT_FIGS = Path("figures")
OUT_DATA.mkdir(parents=True, exist_ok=True)
OUT_FIGS.mkdir(parents=True, exist_ok=True)

WINDOW = 90
MIN_PERIODS = 30
POLLUTANTS = ["SO2", "NO2", "O3", "PM25"]

def rolling_zscore(s: pd.Series, window=WINDOW, min_periods=MIN_PERIODS) -> pd.Series:
    mu = s.rolling(window, min_periods=min_periods).mean()
    sd = s.rolling(window, min_periods=min_periods).std(ddof=0)
    z = (s - mu) / sd
    return z

def main():
    df = pd.read_csv(IN, parse_dates=["Datetime"], index_col="Datetime")

    # keep only expected columns if extra slipped in
    cols = [c for c in POLLUTANTS if c in df.columns]
    df = df[cols].sort_index()

    # rolling z-scores for each pollutant
    z = pd.DataFrame(index=df.index)
    for c in cols:
        z[c + "_z"] = rolling_zscore(df[c])

    # DPBI: mean of available z-scores
    z_cols = [c + "_z" for c in cols]
    dpbi = z[z_cols].mean(axis=1)
    dpbi.name = "DPBI"

    # save outputs
    z.to_csv(OUT_DATA / "zscores_90d.csv")
    dpbi.to_frame().to_csv(OUT_DATA / "dpbi.csv")

    # quick figure for sanity — one stacked subplot per year
    years = sorted(dpbi.index.year.unique())
    fig, axes = plt.subplots(len(years), 1, figsize=(10, 2.6 * len(years)), sharex=False, sharey=True)
    if len(years) == 1:
        axes = [axes]

    for ax, year in zip(axes, years):
        yearly = dpbi[dpbi.index.year == year]
        yearly.plot(ax=ax, label="DPBI")
        ax.set_title(f"{year}")
        ax.set_ylabel("z-score units")
        if ax is not axes[-1]:
            ax.set_xlabel("")
        else:
            ax.set_xlabel("Date")
        legend = ax.get_legend()
        if ax is not axes[0] and legend:
            legend.remove()

    fig.suptitle("Daily Pollution Burden Index (DPBI) — 90-day rolling z")
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(OUT_FIGS / "02_dpbi.png", dpi=180)
    plt.close(fig)
    print("Wrote data_processed/zscores_90d.csv, data_processed/dpbi.csv and figures/02_dpbi.png")

if __name__ == "__main__":
    main()

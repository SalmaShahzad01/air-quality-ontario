# scripts/03_extremes_sensitivity.py
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

IN = Path("data_processed/dpbi.csv")
OUT_DATA = Path("data_processed")
OUT_FIGS = Path("figures")
OUT_DATA.mkdir(parents=True, exist_ok=True)
OUT_FIGS.mkdir(parents=True, exist_ok=True)

QUANTILES = [0.90, 0.95, 0.975]

def mark_extremes(series: pd.Series, q: float) -> pd.DataFrame:
    thr = series.quantile(q)
    flags = series >= thr
    out = pd.DataFrame({
        "value": series,
        "is_high_extreme": flags,
        "threshold": thr
    })
    return out

def yearly_counts(flag_series: pd.Series) -> pd.Series:
    return flag_series.groupby(flag_series.index.year).sum()

def main():
    s = pd.read_csv(IN, parse_dates=["Datetime"], index_col="Datetime")["DPBI"]

    for q in QUANTILES:
        tag = str(int(q*1000)).rstrip('0')  # '900','950','975'
        df = mark_extremes(s, q)
        df.to_csv(OUT_DATA / f"dpbi_extremes_{tag}.csv")

        yc = yearly_counts(df["is_high_extreme"])
        yc.to_frame("count_high").to_csv(OUT_DATA / f"dpbi_extremes_{tag}_yearly_counts.csv")

        if abs(q - 0.95) < 1e-9:
            # plot markers for the main 95th percentile — stacked by year
            years = sorted(s.index.year.unique())
            fig, axes = plt.subplots(len(years), 1, figsize=(10, 2.6 * len(years)), sharex=False, sharey=True)
            if len(years) == 1:
                axes = [axes]

            for ax, year in zip(axes, years):
                mask = s.index.year == year
                yearly = s[mask]
                yearly.plot(ax=ax, label="DPBI")
                yearly[df["is_high_extreme"] & mask].plot(ax=ax, style='o', markersize=3, label="≥95th pct")
                ax.set_title(f"{year}")
                ax.set_ylabel("DPBI")
                if ax is not axes[-1]:
                    ax.set_xlabel("")
                else:
                    ax.set_xlabel("Date")
                legend = ax.get_legend()
                if ax is not axes[0] and legend:
                    legend.remove()

            fig.suptitle("DPBI with ≥95th percentile extremes")
            fig.tight_layout(rect=[0, 0, 1, 0.97])
            fig.savefig(OUT_FIGS / "03_dpbi_extremes_95.png", dpi=180)
            plt.close(fig)

    print("Wrote dpbi_extremes_*.csv, *_yearly_counts.csv and figures/03_dpbi_extremes_95.png")

if __name__ == "__main__":
    main()

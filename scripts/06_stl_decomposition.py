# scripts/06_stl_decomposition.py

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import STL

IN_DPBI = Path("data_processed/dpbi.csv")
OUT_DATA = Path("data_processed")
OUT_FIGS = Path("figures")
OUT_DATA.mkdir(parents=True, exist_ok=True)
OUT_FIGS.mkdir(parents=True, exist_ok=True)


def stl_one_series(s: pd.Series, period: int) -> pd.DataFrame:
    """
    Run STL on a single time series and return a DataFrame
    with columns: original, trend, seasonal, resid.
    """
    s = s.sort_index().dropna()
    if len(s) < period * 2:
        # too short for a stable seasonal pattern; still try with smaller period
        period = max(7, min(period, len(s) // 2))
    stl = STL(s, period=period, robust=True)
    res = stl.fit()
    return pd.DataFrame(
        {
            "value": s,
            "trend": res.trend,
            "seasonal": res.seasonal,
            "resid": res.resid,
        }
    )


def plot_stl_components(components: pd.DataFrame, title: str, outfile: Path):
    """
    Plot STL components (original, trend, seasonal, residuals) stacked vertically.
    """
    fig, axes = plt.subplots(4, 1, figsize=(10, 8), sharex=True)

    components["value"].plot(ax=axes[0])
    axes[0].set_ylabel("Value")
    axes[0].set_title(title)

    components["trend"].plot(ax=axes[1])
    axes[1].set_ylabel("Trend")

    components["seasonal"].plot(ax=axes[2])
    axes[2].set_ylabel("Seasonal")

    components["resid"].plot(ax=axes[3])
    axes[3].set_ylabel("Residual")
    axes[3].set_xlabel("Date")

    plt.tight_layout()
    fig.savefig(outfile, dpi=180)
    plt.close(fig)


def main():
    # Load DPBI
    df = pd.read_csv(IN_DPBI, parse_dates=["Datetime"], index_col="Datetime")
    dpbi = df["DPBI"].sort_index().dropna()

    if dpbi.empty:
        raise ValueError("DPBI series is empty or all NaN; run 02_compute_features.py first.")

    # ---- (A) Full-period STL (for overall seasonal pattern) ----
    full_components = stl_one_series(dpbi, period=365)
    full_components.to_csv(OUT_DATA / "dpbi_stl_components_full.csv")
    plot_stl_components(
        full_components,
        title="DPBI STL decomposition (full 2021–2024)",
        outfile=OUT_FIGS / "06_dpbi_stl_full.png",
    )

    # ---- (B) Per-year STL decomposition ----
    years = sorted(dpbi.index.year.unique())
    for year in years:
        s_year = dpbi[dpbi.index.year == year]
        if len(s_year) < 60:
            # too little data for meaningful STL; skip with a message
            print(f"Skipping STL for {year}: not enough data ({len(s_year)} points).")
            continue

        # use smaller period within a year to capture intra-year seasonality
        # here we use ~monthly/shorter pattern (e.g. 30 days)
        period = 30
        comps_year = stl_one_series(s_year, period=period)
        comps_year.to_csv(OUT_DATA / f"dpbi_stl_components_{year}.csv")

        plot_stl_components(
            comps_year,
            title=f"DPBI STL decomposition ({year})",
            outfile=OUT_FIGS / f"06_dpbi_stl_{year}.png",
        )

    print("Wrote:")
    print(" - data_processed/dpbi_stl_components_full.csv")
    print(" - figures/06_dpbi_stl_full.png")
    print(" - data_processed/dpbi_stl_components_<year>.csv")
    print(" - figures/06_dpbi_stl_<year>.png for each year with enough data")


if __name__ == "__main__":
    main()

# scripts/prepare_first_plot.py
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

RAW = Path("data_raw")
OUT_DATA = Path("data_processed")
OUT_FIGS = Path("figures")
OUT_DATA.mkdir(parents=True, exist_ok=True)
OUT_FIGS.mkdir(parents=True, exist_ok=True)

MISSING = [9999, -999, -9999]

def _detect_header_row(filepath: Path, max_scan_rows: int = 40) -> int:
    """
    Scan the top of the file to find the header row index that contains 'Date' and hour columns (H01..).
    Returns the 0-based row index to use as header (i.e., pass to pd.read_csv(header=row_idx)).
    """
    tmp = pd.read_csv(filepath, header=None, nrows=max_scan_rows, dtype=str, keep_default_na=False)
    for i in range(len(tmp)):
        row = tmp.iloc[i].astype(str).str.strip().str.lower()
        if any(cell.startswith('date') for cell in row):
            # also check if we see at least something like H01 in the same row
            if any(cell.startswith('h01') for cell in row):
                return i
    # Fallback: try the row that contains 'station id' and assume next row is header
    for i in range(len(tmp)):
        row = tmp.iloc[i].astype(str).str.strip().str.lower()
        if 'station id' in row.values:
            return i + 1
    # Final fallback to 11 (old assumption)
    return 11


def load_clean_long(filepath: Path) -> pd.DataFrame:
    """
    Robust loader for Air Quality Ontario CSV:
    - Auto-detect header row
    - Tolerate variations in 'Date' column label (e.g., 'Date ', 'DATE', 'Date (LST)')
    - Replace missing sentinels, drop duplicates
    - Drop rows with all hourly values missing
    - Return hourly long DF with Datetime index
    """
    header_row = _detect_header_row(filepath)
    df = pd.read_csv(filepath, header=header_row)
    # Normalise column names
    df.columns = df.columns.astype(str).str.strip()

    # Find the date column
    date_col = None
    for c in df.columns:
        lc = c.lower().strip()
        if lc == 'date' or lc.startswith('date'):
            date_col = c
            break
    if date_col is None:
        raise KeyError("Could not find a 'Date' column in the header. "
                       f"Detected header row: {header_row}. Columns: {list(df.columns)}")

    # Replace missing sentinels
    df.replace(MISSING, np.nan, inplace=True)
    df.drop_duplicates(inplace=True)

    # Identify hour columns like H01..H24 (case-insensitive)
    hour_cols = [c for c in df.columns if str(c).strip().upper().startswith('H')]
    if not hour_cols:
        raise KeyError("No hour columns found (expected H01..H24). "
                       f"Columns present: {list(df.columns)}")

    # Drop rows where all hourly values are missing
    df.dropna(subset=hour_cols, how='all', inplace=True)

    # Build long format
    df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
    long = df.melt(id_vars=[date_col], value_vars=hour_cols,
                   var_name='Hour', value_name='Concentration')
    long['Hour'] = long['Hour'].astype(str).str.extract(r'H(\d+)', expand=False).astype(int) - 1
    long['Datetime'] = long[date_col] + pd.to_timedelta(long['Hour'], unit='h')
    long = long.drop(columns=[date_col, 'Hour']).sort_values('Datetime').set_index('Datetime')
    return long


def to_daily_mean(df_hourly: pd.DataFrame, min_valid_hours=18) -> pd.DataFrame:
    """
    Compute daily mean only if at least 'min_valid_hours' exist for that day.
    Otherwise, set daily mean to NaN.
    """
    # count valid hourly values per day
    valid_counts = df_hourly.resample('D').count()
    daily_mean = df_hourly.resample('D').mean()

    # where count < threshold → set to NaN
    daily_mean[valid_counts < min_valid_hours] = np.nan
    return daily_mean


def main():
    files = {
        "SO2": RAW / "Sulphate_2021_2024.csv",
        "NO2": RAW / "Nitrogen_2021_2024.csv",
        "O3":  RAW / "Ozone_2021-2024.csv",
        "PM25":RAW / "PM2.5_2021_2024.csv",
    }

    daily = {}
    for name, path in files.items():
        h = load_clean_long(path)
        d = to_daily_mean(h)
        d.to_csv(OUT_DATA / f"{name}_daily_mean.csv")
        daily[name] = d.rename(columns={"Concentration": name})

    # merge daily means (outer join keeps full date range)
    merged = pd.concat(daily.values(), axis=1).sort_index()
    merged.to_csv(OUT_DATA / "merged_daily_mean.csv")

    # FIRST PLOT
    ax = merged.plot(figsize=(10,4))
    ax.set_title("Daily pollutant trends (mean) — Toronto North, 2021–2024")
    ax.set_ylabel("Concentration (ppb / µg/m³)")
    ax.set_xlabel("Date")
    plt.tight_layout()
    plt.savefig(OUT_FIGS / "01_trends_all.png", dpi=180)
    plt.close()
    print("Done. Wrote:")
    print(" - data_processed/merged_daily_mean.csv")
    print(" - figures/01_trends_all.png")

if __name__ == "__main__":
    main()

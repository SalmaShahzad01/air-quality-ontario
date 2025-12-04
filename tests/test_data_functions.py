import importlib.util
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import numpy as np
import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT / "scripts"


def load_module(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS_DIR / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


prepare = load_module("prepare_first_plot", "prepare_first_plot.py")
compute = load_module("compute_features", "02_compute_features.py")
extremes = load_module("extremes", "03_extremes_sensitivity.py")
trends = load_module("trends", "04_trends.py")
stl_module = load_module("stl_decomposition", "06_stl_decomposition.py")


def test_to_daily_mean_respects_min_hour_threshold():
    idx = pd.date_range("2021-01-01", periods=48, freq="H")
    data = pd.DataFrame({"Concentration": np.arange(48, dtype=float)}, index=idx)
    data.loc["2021-01-02 00:00":"2021-01-02 12:00", "Concentration"] = np.nan

    result = prepare.to_daily_mean(data, min_valid_hours=18)

    assert result.loc["2021-01-01", "Concentration"] == pytest.approx(11.5)
    assert pd.isna(result.loc["2021-01-02", "Concentration"])


def manual_rolling_z(series: pd.Series, window: int, min_periods: int) -> pd.Series:
    values = series.values.astype(float)
    rolling = []
    for i in range(len(values)):
        start = max(0, i - window + 1)
        current = values[start : i + 1]
        if len(current) < min_periods:
            rolling.append(np.nan)
            continue
        mu = current.mean()
        sd = np.sqrt(((current - mu) ** 2).mean())
        rolling.append((values[i] - mu) / sd if sd != 0 else np.nan)
    return pd.Series(rolling, index=series.index)


def test_rolling_zscore_matches_manual_calculation():
    idx = pd.date_range("2022-01-01", periods=6, freq="D")
    series = pd.Series([5, 7, 9, 11, 13, 15], index=idx, dtype=float)

    result = compute.rolling_zscore(series, window=3, min_periods=3)
    expected = manual_rolling_z(series, window=3, min_periods=3)

    pd.testing.assert_series_equal(result, expected)


def test_mark_extremes_uses_quantile_threshold():
    idx = pd.date_range("2023-01-01", periods=5, freq="D")
    series = pd.Series([10, 20, 30, 40, 50], index=idx, dtype=float)

    df = extremes.mark_extremes(series, q=0.8)

    assert df["threshold"].iloc[0] == pytest.approx(42.0)
    assert df["is_high_extreme"].tolist() == [False, False, False, False, True]


def test_yearly_counts_groups_flags_by_year():
    dates = pd.to_datetime(["2021-01-01", "2021-06-01", "2022-03-05", "2022-08-09"])
    flags = pd.Series([True, False, True, True], index=dates)

    counts = extremes.yearly_counts(flags)

    assert counts.loc[2021] == 1
    assert counts.loc[2022] == 2


def test_linear_slope_recovers_known_daily_change():
    idx = pd.date_range("2020-01-01", periods=40, freq="D")
    series = pd.Series(1.5 * np.arange(40) + 3.0, index=idx)

    slope = trends.linear_slope(series)

    assert slope == pytest.approx(1.5, abs=1e-6)


def test_kendall_tau_requires_sufficient_samples():
    short_series = pd.Series([1, 2, 3, 4], index=pd.date_range("2021-01-01", periods=4))
    tau, p = trends.kendall_tau(short_series)
    assert tau is None and p is None

    long_series = pd.Series(np.arange(30), index=pd.date_range("2021-01-01", periods=30))
    tau_long, p_long = trends.kendall_tau(long_series)
    assert tau_long > 0.95
    assert p_long < 1e-4


def test_stl_one_series_reconstructs_original_signal():
    idx = pd.date_range("2022-01-01", periods=180, freq="D")
    trend_component = np.linspace(0, 5, 180)
    seasonal_component = np.sin(2 * np.pi * np.arange(180) / 30)
    values = trend_component + seasonal_component
    series = pd.Series(values, index=idx)
    series.iloc[10] = np.nan  # ensure NaNs are dropped internally

    components = stl_module.stl_one_series(series, period=60)

    pd.testing.assert_index_equal(components.index, series.dropna().index)
    assert {"value", "trend", "seasonal", "resid"} <= set(components.columns)
    np.testing.assert_allclose(
        components["trend"] + components["seasonal"] + components["resid"],
        components["value"],
        atol=1e-6,
    )


def test_plot_stl_components_writes_png(tmp_path):
    idx = pd.date_range("2021-01-01", periods=5, freq="D")
    trend = np.linspace(0, 4, 5)
    seasonal = np.linspace(-1, 1, 5)
    resid = np.linspace(0.5, -0.5, 5)
    components = pd.DataFrame(
        {
            "value": trend + seasonal + resid,
            "trend": trend,
            "seasonal": seasonal,
            "resid": resid,
        },
        index=idx,
    )
    outfile = tmp_path / "stl_components.png"

    stl_module.plot_stl_components(components, "Demo STL", outfile)

    assert outfile.exists()
    assert outfile.stat().st_size > 0

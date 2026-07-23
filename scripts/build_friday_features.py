"""Build the Friday market-regime feature table."""

from __future__ import annotations

import sys
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATABASE_PATH = PROJECT_ROOT / "database" / "research.duckdb"
OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "friday_regime_features.parquet"
)

ATR_PERIOD = 14


def wilder_average(series: pd.Series, period: int) -> pd.Series:
    """Calculate a Wilder-style moving average."""

    values = pd.to_numeric(series, errors="coerce")
    result = pd.Series(np.nan, index=values.index, dtype="float64")

    valid_positions = np.flatnonzero(values.notna().to_numpy())

    if len(valid_positions) < period:
        return result

    start = int(valid_positions[0])
    seed_position = start + period - 1

    if seed_position >= len(values):
        return result

    seed_values = values.iloc[start : seed_position + 1]

    if seed_values.isna().any():
        return result

    result.iloc[seed_position] = seed_values.mean()

    for position in range(seed_position + 1, len(values)):
        current_value = values.iloc[position]

        if pd.isna(current_value):
            result.iloc[position] = result.iloc[position - 1]
        else:
            result.iloc[position] = (
                result.iloc[position - 1] * (period - 1)
                + current_value
            ) / period

    return result


def build_weekly_bars(daily: pd.DataFrame) -> pd.DataFrame:
    """Aggregate daily bars into completed trading weeks."""

    daily = daily.sort_values("date").copy()
    daily["date"] = pd.to_datetime(daily["date"])

    days_to_friday = (4 - daily["date"].dt.weekday) % 7

    daily["week_end"] = daily["date"] + pd.to_timedelta(
        days_to_friday,
        unit="D",
    )

    weekly = (
        daily.groupby("week_end", as_index=False)
        .agg(
            entry_date=("date", "max"),
            weekly_open=("open", "first"),
            weekly_high=("high", "max"),
            weekly_low=("low", "min"),
            weekly_close=("close", "last"),
            trading_sessions=("date", "size"),
        )
        .sort_values("entry_date")
        .reset_index(drop=True)
    )

    # Keep normal Friday weeks and holiday-shortened Thursday weeks.
    # Remove an incomplete Monday-through-Wednesday terminal week.
    weekly = weekly[
        weekly["entry_date"].dt.weekday >= 3
    ].reset_index(drop=True)

    return weekly


def add_ndx_features(weekly: pd.DataFrame) -> pd.DataFrame:
    """Create NDX trend, momentum, drawdown and ATR features."""

    weekly = weekly.copy()

    prior_close = weekly["weekly_close"].shift(1)

    weekly["weekly_true_range"] = pd.concat(
        [
            weekly["weekly_high"] - weekly["weekly_low"],
            (weekly["weekly_high"] - prior_close).abs(),
            (weekly["weekly_low"] - prior_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    weekly["weekly_atr14_current"] = wilder_average(
        weekly["weekly_true_range"],
        ATR_PERIOD,
    )

    # Compare the current completed week's range with ATR known
    # before the current week.
    weekly["weekly_atr14_prior"] = (
        weekly["weekly_atr14_current"].shift(1)
    )

    weekly["weekly_range"] = (
        weekly["weekly_high"] - weekly["weekly_low"]
    )

    weekly["weekly_range_atr14_ratio"] = (
        weekly["weekly_range"]
        / weekly["weekly_atr14_prior"]
    )

    weekly["sma20_week"] = (
        weekly["weekly_close"].rolling(20).mean()
    )

    weekly["sma40_week"] = (
        weekly["weekly_close"].rolling(40).mean()
    )

    weekly["ndx_vs_sma20"] = (
        weekly["weekly_close"] / weekly["sma20_week"] - 1.0
    )

    weekly["ndx_vs_sma40"] = (
        weekly["weekly_close"] / weekly["sma40_week"] - 1.0
    )

    weekly["sma20_slope_4w"] = (
        weekly["sma20_week"]
        / weekly["sma20_week"].shift(4)
        - 1.0
    )

    weekly["sma40_slope_4w"] = (
        weekly["sma40_week"]
        / weekly["sma40_week"].shift(4)
        - 1.0
    )

    weekly["ndx_momentum_4w"] = (
        weekly["weekly_close"]
        / weekly["weekly_close"].shift(4)
        - 1.0
    )

    weekly["ndx_momentum_12w"] = (
        weekly["weekly_close"]
        / weekly["weekly_close"].shift(12)
        - 1.0
    )

    weekly["prior_20w_high"] = (
        weekly["weekly_high"]
        .shift(1)
        .rolling(20)
        .max()
    )

    weekly["prior_52w_high"] = (
        weekly["weekly_high"]
        .shift(1)
        .rolling(52)
        .max()
    )

    weekly["drawdown_from_20w_high"] = (
        weekly["weekly_close"]
        / weekly["prior_20w_high"]
        - 1.0
    )

    weekly["drawdown_from_52w_high"] = (
        weekly["weekly_close"]
        / weekly["prior_52w_high"]
        - 1.0
    )

    # Synthetic rolling four-week bar.
    weekly["rolling4w_open"] = weekly["weekly_open"].shift(3)

    weekly["rolling4w_high"] = (
        weekly["weekly_high"].rolling(4).max()
    )

    weekly["rolling4w_low"] = (
        weekly["weekly_low"].rolling(4).min()
    )

    weekly["rolling4w_close"] = weekly["weekly_close"]

    weekly["rolling4w_range"] = (
        weekly["rolling4w_high"]
        - weekly["rolling4w_low"]
    )

    prior_rolling_close = weekly["rolling4w_close"].shift(1)

    weekly["rolling4w_true_range"] = pd.concat(
        [
            weekly["rolling4w_range"],
            (
                weekly["rolling4w_high"]
                - prior_rolling_close
            ).abs(),
            (
                weekly["rolling4w_low"]
                - prior_rolling_close
            ).abs(),
        ],
        axis=1,
    ).max(axis=1)

    weekly["rolling4w_atr14_current"] = wilder_average(
        weekly["rolling4w_true_range"],
        ATR_PERIOD,
    )

    weekly["rolling4w_atr14_prior"] = (
        weekly["rolling4w_atr14_current"].shift(1)
    )

    weekly["rolling4w_range_atr14_ratio"] = (
        weekly["rolling4w_range"]
        / weekly["rolling4w_atr14_prior"]
    )

    weekly["rolling4w_close_location"] = (
        (
            weekly["rolling4w_close"]
            - weekly["rolling4w_low"]
        )
        / weekly["rolling4w_range"]
    )

    weekly["below_sma20"] = (
        weekly["weekly_close"] < weekly["sma20_week"]
    )

    weekly["below_sma40"] = (
        weekly["weekly_close"] < weekly["sma40_week"]
    )

    weekly["sma20_falling"] = weekly["sma20_slope_4w"] < 0

    weekly["sma40_falling"] = weekly["sma40_slope_4w"] < 0

    return weekly


def add_spy_features(weekly: pd.DataFrame) -> pd.DataFrame:
    """Create SPY weekly momentum features."""

    weekly = weekly.copy()

    weekly["spy_momentum_4w"] = (
        weekly["weekly_close"]
        / weekly["weekly_close"].shift(4)
        - 1.0
    )

    weekly["spy_momentum_12w"] = (
        weekly["weekly_close"]
        / weekly["weekly_close"].shift(12)
        - 1.0
    )

    return weekly[
        [
            "week_end",
            "entry_date",
            "weekly_close",
            "spy_momentum_4w",
            "spy_momentum_12w",
        ]
    ].rename(
        columns={
            "entry_date": "spy_entry_date",
            "weekly_close": "spy_weekly_close",
        }
    )


def percentile_rank_last(window: np.ndarray) -> float:
    """Return the percentile rank of the last value in a window."""

    last_value = window[-1]

    return float(np.mean(window <= last_value))


def build_vix_features(vix: pd.DataFrame) -> pd.DataFrame:
    """Create daily VIX change, trend and percentile features."""

    vix = vix.sort_values("date").copy()
    vix["date"] = pd.to_datetime(vix["date"])

    vix["vix_change_5d"] = (
        vix["close"] / vix["close"].shift(5) - 1.0
    )

    vix["vix_change_20d"] = (
        vix["close"] / vix["close"].shift(20) - 1.0
    )

    vix["vix_sma20"] = vix["close"].rolling(20).mean()

    vix["vix_vs_sma20"] = (
        vix["close"] / vix["vix_sma20"] - 1.0
    )

    vix["vix_percentile_252d"] = (
        vix["close"]
        .rolling(252)
        .apply(percentile_rank_last, raw=True)
    )

    vix["vix_above_sma20"] = (
        vix["close"] > vix["vix_sma20"]
    )

    return vix[
        [
            "date",
            "close",
            "vix_change_5d",
            "vix_change_20d",
            "vix_sma20",
            "vix_vs_sma20",
            "vix_percentile_252d",
            "vix_above_sma20",
        ]
    ].rename(
        columns={
            "date": "vix_feature_date",
            "close": "vix_feature_close",
        }
    )


def main() -> int:
    """Build and save the Friday feature table."""

    print("MNQ PCS Regime Research — Friday Feature Build")
    print("=" * 47)

    if not DATABASE_PATH.is_file():
        print(f"ERROR: Missing database: {DATABASE_PATH}")
        return 1

    connection = duckdb.connect(str(DATABASE_PATH))

    try:
        baseline = connection.execute(
            """
            SELECT *
            FROM pcs.friday_trade_baseline
            ORDER BY entry_date
            """
        ).fetchdf()

        ndx = connection.execute(
            """
            SELECT date, open, high, low, close
            FROM market.ndx_daily
            ORDER BY date
            """
        ).fetchdf()

        spy = connection.execute(
            """
            SELECT date, open, high, low, close
            FROM market.spy_daily
            ORDER BY date
            """
        ).fetchdf()

        vix = connection.execute(
            """
            SELECT date, close
            FROM market.vix_daily
            ORDER BY date
            """
        ).fetchdf()

        baseline["entry_date"] = pd.to_datetime(
            baseline["entry_date"]
        )

        ndx_weekly = add_ndx_features(
            build_weekly_bars(ndx)
        )

        spy_weekly = add_spy_features(
            build_weekly_bars(spy)
        )

        weekly_features = ndx_weekly.merge(
            spy_weekly,
            on="week_end",
            how="left",
            validate="one_to_one",
        )

        weekly_features["ndx_vs_spy_4w"] = (
            (
                1.0 + weekly_features["ndx_momentum_4w"]
            )
            / (
                1.0 + weekly_features["spy_momentum_4w"]
            )
            - 1.0
        )

        weekly_features["ndx_vs_spy_12w"] = (
            (
                1.0 + weekly_features["ndx_momentum_12w"]
            )
            / (
                1.0 + weekly_features["spy_momentum_12w"]
            )
            - 1.0
        )

        feature_columns = [
            "entry_date",
            "week_end",
            "weekly_open",
            "weekly_high",
            "weekly_low",
            "weekly_close",
            "weekly_range",
            "weekly_atr14_prior",
            "weekly_range_atr14_ratio",
            "sma20_week",
            "sma40_week",
            "ndx_vs_sma20",
            "ndx_vs_sma40",
            "sma20_slope_4w",
            "sma40_slope_4w",
            "ndx_momentum_4w",
            "ndx_momentum_12w",
            "prior_20w_high",
            "prior_52w_high",
            "drawdown_from_20w_high",
            "drawdown_from_52w_high",
            "rolling4w_open",
            "rolling4w_high",
            "rolling4w_low",
            "rolling4w_close",
            "rolling4w_range",
            "rolling4w_atr14_prior",
            "rolling4w_range_atr14_ratio",
            "rolling4w_close_location",
            "below_sma20",
            "below_sma40",
            "sma20_falling",
            "sma40_falling",
            "spy_weekly_close",
            "spy_momentum_4w",
            "spy_momentum_12w",
            "ndx_vs_spy_4w",
            "ndx_vs_spy_12w",
        ]

        features = baseline.merge(
            weekly_features[feature_columns],
            on="entry_date",
            how="left",
            validate="one_to_one",
        )

        vix_features = build_vix_features(vix)

        features = pd.merge_asof(
            features.sort_values("entry_date"),
            vix_features.sort_values("vix_feature_date"),
            left_on="entry_date",
            right_on="vix_feature_date",
            direction="backward",
            tolerance=pd.Timedelta(days=4),
        )

        if len(features) != len(baseline):
            raise ValueError(
                "Feature build changed the baseline row count."
            )

        if features["entry_date"].duplicated().any():
            raise ValueError(
                "Duplicate Friday entry dates were created."
            )

        invalid_vix_dates = (
            features["vix_feature_date"]
            > features["entry_date"]
        )

        if invalid_vix_dates.fillna(False).any():
            raise ValueError(
                "Look-ahead detected in VIX feature alignment."
            )

        core_features = [
            "ndx_vs_sma20",
            "ndx_vs_sma40",
            "sma20_slope_4w",
            "sma40_slope_4w",
            "ndx_momentum_4w",
            "ndx_momentum_12w",
            "drawdown_from_20w_high",
            "drawdown_from_52w_high",
            "weekly_range_atr14_ratio",
            "rolling4w_range_atr14_ratio",
            "rolling4w_close_location",
            "vix_change_5d",
            "vix_change_20d",
            "vix_percentile_252d",
            "ndx_vs_spy_4w",
            "ndx_vs_spy_12w",
        ]

        features["complete_core_features"] = (
            features[core_features].notna().all(axis=1)
        )

        OUTPUT_PATH.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        features.to_parquet(
            OUTPUT_PATH,
            index=False,
            engine="pyarrow",
            compression="zstd",
        )

        connection.register(
            "friday_features_dataframe",
            features,
        )

        connection.execute(
            """
            CREATE SCHEMA IF NOT EXISTS pcs
            """
        )

        connection.execute(
            """
            CREATE OR REPLACE TABLE
                pcs.friday_regime_features AS
            SELECT *
            FROM friday_features_dataframe
            ORDER BY entry_date
            """
        )

        total_rows = len(features)

        complete_rows = int(
            features["complete_core_features"].sum()
        )

        complete = features[
            features["complete_core_features"]
        ]

        print(
            f"Feature rows: {total_rows:,}"
        )

        print(
            f"Rows with complete core features: "
            f"{complete_rows:,} "
            f"({complete_rows / total_rows:.1%})"
        )

        if not complete.empty:
            print(
                f"Complete feature range: "
                f"{complete['entry_date'].min().date()} to "
                f"{complete['entry_date'].max().date()}"
            )

        latest = features.iloc[-1]

        print("\nLatest feature row")
        print("-" * 48)
        print(f"Entry date: {latest['entry_date'].date()}")
        print(
            f"NDX vs 20-week SMA: "
            f"{latest['ndx_vs_sma20']:.2%}"
        )
        print(
            f"NDX 4-week momentum: "
            f"{latest['ndx_momentum_4w']:.2%}"
        )
        print(
            f"Drawdown from prior 52-week high: "
            f"{latest['drawdown_from_52w_high']:.2%}"
        )
        print(
            f"Rolling 4-week range / prior ATR: "
            f"{latest['rolling4w_range_atr14_ratio']:.2f}x"
        )
        print(
            f"Rolling 4-week close location: "
            f"{latest['rolling4w_close_location']:.1%}"
        )
        print(
            f"VIX 20-day change: "
            f"{latest['vix_change_20d']:.2%}"
        )
        print(
            f"NDX vs SPY relative strength, 12 weeks: "
            f"{latest['ndx_vs_spy_12w']:.2%}"
        )

        print(f"\nParquet output: {OUTPUT_PATH}")
        print(
            "DuckDB table: pcs.friday_regime_features"
        )
        print("Friday feature build completed successfully.")

        return 0

    except Exception as exc:
        print(f"\nERROR: {exc}")
        return 1

    finally:
        connection.close()


if __name__ == "__main__":
    sys.exit(main())

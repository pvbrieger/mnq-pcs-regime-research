"""Build the baseline Friday PCS research dataset."""

from __future__ import annotations

import sys
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATABASE_PATH = PROJECT_ROOT / "database" / "research.duckdb"
CONFIG_PATH = PROJECT_ROOT / "config" / "strategy_rules.yaml"
OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "friday_trade_baseline.parquet"


def load_config() -> dict:
    """Load the official strategy research configuration."""

    with CONFIG_PATH.open("r", encoding="utf-8") as file_handle:
        return yaml.safe_load(file_handle)


def ladder_for_vix(
    vix_close: float,
    ladder_rules: list[dict],
) -> tuple[str, float]:
    """Return the applicable VIX tier and short-strike distance."""

    for rule in ladder_rules:
        minimum = rule["minimum_inclusive"]
        maximum = rule["maximum_exclusive"]

        minimum_ok = minimum is None or vix_close >= minimum
        maximum_ok = maximum is None or vix_close < maximum

        if minimum_ok and maximum_ok:
            return rule["label"], float(rule["short_strike_distance"])

    raise ValueError(f"No VIX ladder rule matched VIX={vix_close}")


def build_weekly_bars(ndx: pd.DataFrame) -> pd.DataFrame:
    """Aggregate daily NDX data into completed weeks ending Friday."""

    ndx = ndx.sort_values("date").copy()

    days_to_friday = (4 - ndx["date"].dt.weekday) % 7
    ndx["week_end"] = ndx["date"] + pd.to_timedelta(
        days_to_friday,
        unit="D",
    )

    weekly = (
        ndx.groupby("week_end", as_index=False)
        .agg(
            entry_date=("date", "max"),
            weekly_open=("open", "first"),
            weekly_high=("high", "max"),
            weekly_low=("low", "min"),
            entry_close=("close", "last"),
            trading_sessions=("date", "size"),
        )
        .sort_values("entry_date")
        .reset_index(drop=True)
    )

    # Retain Friday and holiday-shortened Thursday entries.
    # This also removes the incomplete final week ending July 10.
    weekly = weekly[
        weekly["entry_date"].dt.weekday >= 3
    ].reset_index(drop=True)

    return weekly


def main() -> int:
    """Build and validate the baseline Friday dataset."""

    print("MNQ PCS Regime Research — Friday Baseline Build")
    print("=" * 48)

    if not DATABASE_PATH.is_file():
        print(f"ERROR: Database does not exist: {DATABASE_PATH}")
        print("Run scripts/build_database.py first.")
        return 1

    config = load_config()
    holding_weeks = int(config["holding_period_weeks"])
    maximum_vix_staleness = int(
        config["maximum_vix_staleness_days"]
    )
    ladder_rules = config["vix_ladder"]

    connection = duckdb.connect(str(DATABASE_PATH))

    try:
        ndx = connection.execute(
            """
            SELECT date, open, high, low, close
            FROM market.ndx_daily
            ORDER BY date
            """
        ).fetchdf()

        vix = connection.execute(
            """
            SELECT
                date AS vix_date,
                close AS vix_close
            FROM market.vix_daily
            ORDER BY date
            """
        ).fetchdf()

        ndx["date"] = pd.to_datetime(ndx["date"])
        vix["vix_date"] = pd.to_datetime(vix["vix_date"])

        weekly = build_weekly_bars(ndx)

        weekly = pd.merge_asof(
            weekly.sort_values("entry_date"),
            vix.sort_values("vix_date"),
            left_on="entry_date",
            right_on="vix_date",
            direction="backward",
            tolerance=pd.Timedelta(
                days=maximum_vix_staleness
            ),
        )

        records: list[dict[str, object]] = []

        for index in range(len(weekly) - holding_weeks):
            entry = weekly.iloc[index]
            expiration = weekly.iloc[index + holding_weeks]

            if pd.isna(entry["vix_close"]):
                continue

            tier, ladder_distance = ladder_for_vix(
                float(entry["vix_close"]),
                ladder_rules,
            )

            entry_date = pd.Timestamp(entry["entry_date"])
            expiration_date = pd.Timestamp(
                expiration["entry_date"]
            )
            entry_close = float(entry["entry_close"])
            expiration_close = float(
                expiration["entry_close"]
            )

            holding_daily = ndx[
                (ndx["date"] > entry_date)
                & (ndx["date"] <= expiration_date)
            ]

            if holding_daily.empty:
                continue

            short_strike = entry_close * (
                1.0 - ladder_distance
            )
            holding_low = float(holding_daily["low"].min())
            holding_high = float(holding_daily["high"].max())
            minimum_close = float(
                holding_daily["close"].min()
            )

            expiration_failure = (
                expiration_close < short_strike
            )

            records.append(
                {
                    "entry_date": entry_date,
                    "expiration_proxy_date": expiration_date,
                    "holding_calendar_days": (
                        expiration_date - entry_date
                    ).days,
                    "holding_trading_sessions": len(
                        holding_daily
                    ),
                    "ndx_entry_close": entry_close,
                    "vix_date": pd.Timestamp(entry["vix_date"]),
                    "vix_close": float(entry["vix_close"]),
                    "vix_tier": tier,
                    "ladder_distance": ladder_distance,
                    "short_strike_proxy": short_strike,
                    "expiration_close": expiration_close,
                    "expiration_return": (
                        expiration_close / entry_close - 1.0
                    ),
                    "holding_low": holding_low,
                    "maximum_adverse_return": (
                        holding_low / entry_close - 1.0
                    ),
                    "holding_high": holding_high,
                    "maximum_favorable_return": (
                        holding_high / entry_close - 1.0
                    ),
                    "minimum_holding_close": minimum_close,
                    "short_strike_touched": (
                        holding_low <= short_strike
                    ),
                    "expiration_failure": expiration_failure,
                    "expiration_distance_from_strike": (
                        expiration_close / short_strike - 1.0
                    ),
                }
            )

        baseline = pd.DataFrame(records).sort_values(
            "entry_date"
        ).reset_index(drop=True)

        if baseline.empty:
            raise ValueError("No eligible Friday records were created.")

        if baseline["entry_date"].duplicated().any():
            raise ValueError(
                "Duplicate Friday entry dates were created."
            )

        failure_rows = baseline[
            baseline["expiration_failure"]
        ].copy()

        overlapping_failure_counts = []

        for row in baseline.itertuples(index=False):
            overlaps = failure_rows[
                (
                    failure_rows["entry_date"]
                    <= row.expiration_proxy_date
                )
                & (
                    failure_rows["expiration_proxy_date"]
                    >= row.entry_date
                )
            ]

            overlapping_failure_counts.append(len(overlaps))

        baseline["overlapping_failure_count"] = (
            overlapping_failure_counts
        )

        baseline["clustered_expiration_failure"] = (
            baseline["expiration_failure"]
            & (baseline["overlapping_failure_count"] > 1)
        )

        OUTPUT_PATH.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        baseline.to_parquet(
            OUTPUT_PATH,
            index=False,
            engine="pyarrow",
            compression="zstd",
        )

        connection.execute("CREATE SCHEMA IF NOT EXISTS pcs")

        connection.register(
            "friday_baseline_dataframe",
            baseline,
        )

        connection.execute(
            """
            CREATE OR REPLACE TABLE
                pcs.friday_trade_baseline AS
            SELECT *
            FROM friday_baseline_dataframe
            ORDER BY entry_date
            """
        )

        connection.execute(
            """
            CREATE OR REPLACE VIEW
                pcs.friday_baseline_summary AS
            SELECT
                vix_tier,
                COUNT(*) AS entries,
                SUM(
                    CASE
                        WHEN short_strike_touched THEN 1
                        ELSE 0
                    END
                ) AS touches,
                AVG(
                    CASE
                        WHEN short_strike_touched THEN 1.0
                        ELSE 0.0
                    END
                ) AS touch_rate,
                SUM(
                    CASE
                        WHEN expiration_failure THEN 1
                        ELSE 0
                    END
                ) AS failures,
                AVG(
                    CASE
                        WHEN expiration_failure THEN 1.0
                        ELSE 0.0
                    END
                ) AS failure_rate,
                AVG(expiration_return) AS average_return,
                MIN(expiration_return) AS worst_return
            FROM pcs.friday_trade_baseline
            GROUP BY vix_tier
            ORDER BY MIN(ladder_distance)
            """
        )

        summary = connection.execute(
            """
            SELECT *
            FROM pcs.friday_baseline_summary
            """
        ).fetchdf()

        total_entries = len(baseline)
        total_touches = int(
            baseline["short_strike_touched"].sum()
        )
        total_failures = int(
            baseline["expiration_failure"].sum()
        )
        clustered_failures = int(
            baseline["clustered_expiration_failure"].sum()
        )

        print(
            f"Entry range: "
            f"{baseline['entry_date'].min().date()} to "
            f"{baseline['entry_date'].max().date()}"
        )
        print(f"Eligible Friday entries: {total_entries:,}")
        print(
            f"Short-strike touches: {total_touches} "
            f"({total_touches / total_entries:.2%})"
        )
        print(
            f"Expiration failures: {total_failures} "
            f"({total_failures / total_entries:.2%})"
        )
        print(
            f"Failures occurring in overlapping clusters: "
            f"{clustered_failures}"
        )

        print("\nResults by VIX tier")
        print("-" * 92)

        for row in summary.itertuples(index=False):
            print(
                f"{row.vix_tier:<22} | "
                f"{row.entries:>4} entries | "
                f"{row.touches:>3} touches "
                f"({row.touch_rate:>6.2%}) | "
                f"{row.failures:>2} failures "
                f"({row.failure_rate:>6.2%})"
            )

        print(f"\nParquet output: {OUTPUT_PATH}")
        print(
            "DuckDB table: "
            "pcs.friday_trade_baseline"
        )
        print("Friday baseline build completed successfully.")
        return 0

    except Exception as exc:
        print(f"\nERROR: {exc}")
        return 1

    finally:
        connection.close()


if __name__ == "__main__":
    sys.exit(main())

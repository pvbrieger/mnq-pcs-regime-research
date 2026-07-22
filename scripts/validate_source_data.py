"""Validate shared market-data source files."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ENV_FILE = PROJECT_ROOT / ".env"

SOURCE_FILES = {
    "NDX": ("NDX", "ndx_daily.csv"),
    "QQQ": ("QQQ", "qqq_daily.csv"),
    "SPY": ("SPY", "spy_daily.csv"),
    "VIX": ("VIX", "vix_daily.csv"),
}

REQUIRED_COLUMNS = {"Date", "Open", "High", "Low", "Close"}


def validate_file(symbol: str, path: Path) -> tuple[bool, bool]:
    """Return (critical_passed, has_warnings)."""

    print(f"\n{symbol}")
    print("-" * len(symbol))
    print(f"File: {path}")

    if not path.is_file():
        print("ERROR: File does not exist.")
        return False, False

    try:
        df = pd.read_csv(path)
    except Exception as exc:
        print(f"ERROR: Could not read CSV: {exc}")
        return False, False

    missing_columns = REQUIRED_COLUMNS.difference(df.columns)
    if missing_columns:
        print(f"ERROR: Missing columns: {sorted(missing_columns)}")
        print(f"Columns found: {list(df.columns)}")
        return False, False

    dates = pd.to_datetime(df["Date"], errors="coerce")

    numeric_columns = ["Open", "High", "Low", "Close"]
    numeric = df[numeric_columns].apply(pd.to_numeric, errors="coerce")

    bad_dates = int(dates.isna().sum())
    duplicate_dates = int(dates.duplicated().sum())
    bad_numeric_rows = int(numeric.isna().any(axis=1).sum())

    invalid_ohlc = (
        numeric["High"]
        < numeric[["Open", "Low", "Close"]].max(axis=1)
    ) | (
        numeric["Low"]
        > numeric[["Open", "High", "Close"]].min(axis=1)
    )

    invalid_ohlc_rows = int(invalid_ohlc.sum())
    valid_dates = dates.dropna()

    print(f"Rows: {len(df):,}")
    print(f"Columns: {', '.join(df.columns)}")

    if not valid_dates.empty:
        print(f"First date: {valid_dates.min().date()}")
        print(f"Last date:  {valid_dates.max().date()}")

    if dates.is_monotonic_increasing:
        date_order = "ascending"
    elif dates.is_monotonic_decreasing:
        date_order = "descending"
    else:
        date_order = "unsorted"

    print(f"Date order: {date_order}")
    print(f"Invalid dates: {bad_dates}")
    print(f"Duplicate dates: {duplicate_dates}")
    print(f"Rows with missing numeric values: {bad_numeric_rows}")
    print(f"OHLC consistency warnings: {invalid_ohlc_rows}")

    if invalid_ohlc_rows:
        print("Warning dates:")
        warning_rows = df.loc[
            invalid_ohlc,
            ["Date", "Open", "High", "Low", "Close"],
        ]

        for row in warning_rows.itertuples(index=False):
            print(
                f" - {row.Date}: "
                f"O={row.Open}, H={row.High}, "
                f"L={row.Low}, C={row.Close}"
            )

    critical_passed = (
        len(df) > 0
        and bad_dates == 0
        and duplicate_dates == 0
        and bad_numeric_rows == 0
    )

    has_warnings = invalid_ohlc_rows > 0

    if not critical_passed:
        print("Result: FAIL")
    elif has_warnings:
        print("Result: PASS WITH WARNINGS")
    else:
        print("Result: PASS")

    return critical_passed, has_warnings


def main() -> int:
    load_dotenv(dotenv_path=ENV_FILE)

    shared_path_text = os.getenv("SHARED_MARKET_DATA")
    if not shared_path_text:
        print("ERROR: SHARED_MARKET_DATA is not defined in .env")
        return 1

    shared_path = Path(shared_path_text)

    print("MNQ PCS Regime Research — Source Data Validation")
    print("=" * 49)
    print(f"Shared data root: {shared_path}")

    critical_results = []
    warning_symbols = []

    for symbol, (folder, filename) in SOURCE_FILES.items():
        passed, has_warnings = validate_file(
            symbol=symbol,
            path=shared_path / folder / filename,
        )

        critical_results.append(passed)

        if has_warnings:
            warning_symbols.append(symbol)

    print("\nOverall result")
    print("--------------")

    if not all(critical_results):
        print("One or more source files failed critical validation.")
        return 1

    print("All source files passed critical validation.")

    if warning_symbols:
        print(
            "OHLC warnings will be handled during processed-data creation: "
            + ", ".join(warning_symbols)
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())

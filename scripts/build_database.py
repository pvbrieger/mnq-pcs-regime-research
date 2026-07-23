"""Build the local DuckDB research database from processed Parquet files."""

from __future__ import annotations

import sys
from pathlib import Path

import duckdb


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
DATABASE_DIR = PROJECT_ROOT / "database"
DATABASE_PATH = DATABASE_DIR / "research.duckdb"
MANIFEST_PATH = PROJECT_ROOT / "documentation" / "source_data_manifest.csv"

SYMBOLS = ("ndx", "qqq", "spy", "vix")


def sql_path(path: Path) -> str:
    """Return a safely quoted path for DuckDB SQL."""

    return str(path.resolve()).replace("'", "''")


def main() -> int:
    print("MNQ PCS Regime Research — DuckDB Build")
    print("=" * 42)

    missing_files = [
        PROCESSED_DIR / f"{symbol}_daily.parquet"
        for symbol in SYMBOLS
        if not (PROCESSED_DIR / f"{symbol}_daily.parquet").is_file()
    ]

    if missing_files:
        print("ERROR: Missing processed Parquet files:")

        for path in missing_files:
            print(f" - {path}")

        print("\nRun scripts/build_processed_market_data.py first.")
        return 1

    if not MANIFEST_PATH.is_file():
        print(f"ERROR: Missing source manifest: {MANIFEST_PATH}")
        return 1

    DATABASE_DIR.mkdir(parents=True, exist_ok=True)

    connection = duckdb.connect(str(DATABASE_PATH))

    try:
        connection.execute("CREATE SCHEMA IF NOT EXISTS market")
        connection.execute("CREATE SCHEMA IF NOT EXISTS metadata")
        connection.execute("CREATE SCHEMA IF NOT EXISTS research")

        for symbol in SYMBOLS:
            parquet_path = PROCESSED_DIR / f"{symbol}_daily.parquet"
            table_name = f"market.{symbol}_daily"

            connection.execute(
                f"""
                CREATE OR REPLACE TABLE {table_name} AS
                SELECT *
                FROM read_parquet('{sql_path(parquet_path)}')
                ORDER BY date
                """
            )

            row_count = connection.execute(
                f"SELECT COUNT(*) FROM {table_name}"
            ).fetchone()[0]

            print(f"{symbol.upper()}: {row_count:,} rows loaded")

        connection.execute(
            """
            CREATE OR REPLACE TABLE market.daily_bars AS
            SELECT * FROM market.ndx_daily
            UNION ALL BY NAME
            SELECT * FROM market.qqq_daily
            UNION ALL BY NAME
            SELECT * FROM market.spy_daily
            UNION ALL BY NAME
            SELECT * FROM market.vix_daily
            """
        )

        connection.execute(
            f"""
            CREATE OR REPLACE TABLE metadata.source_manifest AS
            SELECT *
            FROM read_csv_auto(
                '{sql_path(MANIFEST_PATH)}',
                header = true
            )
            """
        )

        connection.execute(
            """
            CREATE OR REPLACE VIEW metadata.data_inventory AS
            SELECT
                symbol,
                COUNT(*) AS row_count,
                MIN(date) AS first_date,
                MAX(date) AS last_date,
                SUM(
                    CASE
                        WHEN ohlc_adjusted THEN 1
                        ELSE 0
                    END
                ) AS adjusted_rows
            FROM market.daily_bars
            GROUP BY symbol
            ORDER BY symbol
            """
        )

        connection.execute(
            """
            CREATE OR REPLACE VIEW metadata.latest_market_dates AS
            SELECT
                symbol,
                MAX(date) AS latest_date
            FROM market.daily_bars
            GROUP BY symbol
            ORDER BY symbol
            """
        )

        inventory = connection.execute(
            """
            SELECT
                symbol,
                row_count,
                first_date,
                last_date,
                adjusted_rows
            FROM metadata.data_inventory
            """
        ).fetchall()

        combined_rows = connection.execute(
            "SELECT COUNT(*) FROM market.daily_bars"
        ).fetchone()[0]

        print(f"\nCombined daily bars: {combined_rows:,}")
        print(f"Database: {DATABASE_PATH}")
        print("\nData inventory")
        print("-" * 72)

        for symbol, rows, first_date, last_date, adjustments in inventory:
            print(
                f"{symbol:>3} | "
                f"{rows:>6,} rows | "
                f"{first_date} to {last_date} | "
                f"{adjustments} adjustment(s)"
            )

        print("\nDuckDB build completed successfully.")
        return 0

    except Exception as exc:
        print(f"\nERROR: {exc}")
        return 1

    finally:
        connection.close()


if __name__ == "__main__":
    sys.exit(main())

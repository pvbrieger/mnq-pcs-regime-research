"""Build standardized daily Parquet files from shared source CSV files."""

from __future__ import annotations

import hashlib
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ENV_FILE = PROJECT_ROOT / ".env"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MANIFEST_FILE = PROJECT_ROOT / "documentation" / "source_data_manifest.csv"

SOURCE_FILES = {
    "NDX": ("NDX", "ndx_daily.csv"),
    "QQQ": ("QQQ", "qqq_daily.csv"),
    "SPY": ("SPY", "spy_daily.csv"),
    "VIX": ("VIX", "vix_daily.csv"),
}

REQUIRED_COLUMNS = {"date", "open", "high", "low", "close"}


def file_sha256(path: Path) -> str:
    """Calculate a file's SHA-256 hash without loading it all into memory."""

    digest = hashlib.sha256()

    with path.open("rb") as file_handle:
        for block in iter(lambda: file_handle.read(1024 * 1024), b""):
            digest.update(block)

    return digest.hexdigest()


def normalize_column_name(name: str) -> str:
    """Convert source column names to consistent snake_case."""

    return (
        str(name)
        .strip()
        .lower()
        .replace(" ", "_")
        .replace("-", "_")
    )


def process_symbol(
    symbol: str,
    source_path: Path,
    output_path: Path,
) -> dict[str, object]:
    """Clean one source CSV and write its standardized Parquet file."""

    if not source_path.is_file():
        raise FileNotFoundError(f"Missing source file: {source_path}")

    source = pd.read_csv(source_path)
    source.columns = [normalize_column_name(column) for column in source.columns]

    missing_columns = REQUIRED_COLUMNS.difference(source.columns)

    if missing_columns:
        raise ValueError(
            f"{symbol}: Missing required columns: {sorted(missing_columns)}"
        )

    source.insert(
        0,
        "source_row_number",
        np.arange(2, len(source) + 2),
    )

    source["date"] = pd.to_datetime(
        source["date"],
        errors="raise",
    ).dt.normalize()

    numeric_columns = ["open", "high", "low", "close"]

    for column in numeric_columns:
        source[column] = pd.to_numeric(source[column], errors="raise")

    if source["date"].duplicated().any():
        duplicate_dates = source.loc[
            source["date"].duplicated(keep=False),
            "date",
        ].dt.strftime("%Y-%m-%d").tolist()

        raise ValueError(
            f"{symbol}: Duplicate dates found: {duplicate_dates[:10]}"
        )

    source = source.sort_values("date").reset_index(drop=True)

    # Preserve the original reported high and low for auditability.
    source["source_high"] = source["high"]
    source["source_low"] = source["low"]

    original_ohlc = source[["open", "high", "low", "close"]]

    # Repair only the processed copy. Raw Google Drive files remain unchanged.
    source["high"] = original_ohlc.max(axis=1)
    source["low"] = original_ohlc.min(axis=1)

    source["ohlc_adjusted"] = (
        (source["high"] != source["source_high"])
        | (source["low"] != source["source_low"])
    )

    clean = pd.DataFrame(
        {
            "symbol": symbol,
            "date": source["date"],
            "open": source["open"].astype("float64"),
            "high": source["high"].astype("float64"),
            "low": source["low"].astype("float64"),
            "close": source["close"].astype("float64"),
            "source_high": source["source_high"].astype("float64"),
            "source_low": source["source_low"].astype("float64"),
            "ohlc_adjusted": source["ohlc_adjusted"].astype("bool"),
            "source_row_number": source["source_row_number"].astype("int64"),
            "source_file": source_path.name,
        }
    )

    if "adj_close" in source.columns:
        clean["adj_close"] = pd.to_numeric(
            source["adj_close"],
            errors="coerce",
        ).astype("float64")

    if "volume" in source.columns:
        clean["volume"] = pd.to_numeric(
            source["volume"],
            errors="coerce",
        ).astype("float64")

    invalid_ohlc = (
        clean["high"]
        < clean[["open", "low", "close"]].max(axis=1)
    ) | (
        clean["low"]
        > clean[["open", "high", "close"]].min(axis=1)
    )

    if invalid_ohlc.any():
        raise ValueError(
            f"{symbol}: OHLC validation still fails after processing."
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)

    clean.to_parquet(
        output_path,
        index=False,
        engine="pyarrow",
        compression="zstd",
    )

    verification = pd.read_parquet(output_path)

    if len(verification) != len(clean):
        raise ValueError(
            f"{symbol}: Parquet verification row count does not match."
        )

    adjusted_rows = int(clean["ohlc_adjusted"].sum())

    return {
        "symbol": symbol,
        "source_relative_path": (
            f"{source_path.parent.name}/{source_path.name}"
        ),
        "source_rows": len(clean),
        "first_date": clean["date"].min().date().isoformat(),
        "last_date": clean["date"].max().date().isoformat(),
        "ohlc_adjusted_rows": adjusted_rows,
        "source_sha256": file_sha256(source_path),
        "processed_file": output_path.name,
        "processed_sha256": file_sha256(output_path),
    }


def main() -> int:
    """Build all standardized market-data files."""

    load_dotenv(dotenv_path=ENV_FILE)

    shared_path_text = os.getenv("SHARED_MARKET_DATA")

    if not shared_path_text:
        print("ERROR: SHARED_MARKET_DATA is not defined in .env")
        return 1

    shared_root = Path(shared_path_text)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    MANIFEST_FILE.parent.mkdir(parents=True, exist_ok=True)

    print("MNQ PCS Regime Research — Processed Data Build")
    print("=" * 48)
    print(f"Shared source: {shared_root}")
    print(f"Local output: {PROCESSED_DIR}")

    manifest_rows = []

    try:
        for symbol, (folder, filename) in SOURCE_FILES.items():
            source_path = shared_root / folder / filename
            output_path = PROCESSED_DIR / f"{symbol.lower()}_daily.parquet"

            result = process_symbol(
                symbol=symbol,
                source_path=source_path,
                output_path=output_path,
            )

            manifest_rows.append(result)

            print(
                f"{symbol}: "
                f"{result['source_rows']:,} rows | "
                f"{result['first_date']} to {result['last_date']} | "
                f"{result['ohlc_adjusted_rows']} OHLC adjustment(s)"
            )

    except Exception as exc:
        print(f"\nERROR: {exc}")
        return 1

    manifest = pd.DataFrame(manifest_rows)
    manifest.to_csv(MANIFEST_FILE, index=False)

    print(f"\nManifest written: {MANIFEST_FILE}")
    print("Processed market-data build completed successfully.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

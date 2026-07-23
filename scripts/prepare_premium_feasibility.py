"""Prepare and validate premium-feasibility input data.

The script is provider-agnostic. It initializes a raw-data workspace and
validates spread-candidate rows exported or assembled from a historical options
data source.

Commands
--------
Initialize the ignored raw-data template:
    python scripts/prepare_premium_feasibility.py --initialize

Validate populated input data:
    python scripts/prepare_premium_feasibility.py --validate

Validate a different CSV:
    python scripts/prepare_premium_feasibility.py --validate --input path/to/file.csv
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, time
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import duckdb
import numpy as np
import pandas as pd
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "config" / "premium_feasibility.yaml"
DATABASE_PATH = PROJECT_ROOT / "database" / "research.duckdb"

REQUIRED_COLUMNS = (
    "entry_date",
    "quote_timestamp_utc",
    "data_source",
    "underlying_symbol",
    "underlying_price",
    "expiration_date",
    "dte_calendar",
    "settlement_type",
    "candidate_type",
    "target_short_strike",
    "actual_short_strike",
    "actual_long_strike",
    "short_bid",
    "short_ask",
    "long_bid",
    "long_ask",
    "short_open_interest",
    "long_open_interest",
    "short_volume",
    "long_volume",
)

NUMERIC_COLUMNS = (
    "underlying_price",
    "dte_calendar",
    "target_short_strike",
    "actual_short_strike",
    "actual_long_strike",
    "short_bid",
    "short_ask",
    "long_bid",
    "long_ask",
    "short_open_interest",
    "long_open_interest",
    "short_volume",
    "long_volume",
)


def load_config() -> dict[str, Any]:
    """Load the premium-feasibility configuration."""

    if not CONFIG_PATH.is_file():
        raise FileNotFoundError(
            f"Missing configuration file: {CONFIG_PATH}"
        )

    with CONFIG_PATH.open("r", encoding="utf-8") as file_handle:
        return yaml.safe_load(file_handle)


def resolve_project_path(value: str) -> Path:
    """Resolve a project-relative or absolute path."""

    path = Path(value).expanduser()

    if path.is_absolute():
        return path

    return PROJECT_ROOT / path


def initialize_template(
    config: dict[str, Any],
    force: bool,
) -> int:
    """Create the ignored raw-data template."""

    input_path = resolve_project_path(
        str(config["input_csv"])
    )

    input_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if input_path.exists() and not force:
        print(
            f"Template already exists: {input_path}\n"
            "Use --force only when you intend to replace it."
        )
        return 0

    template = pd.DataFrame(
        columns=list(REQUIRED_COLUMNS)
    )

    template.to_csv(
        input_path,
        index=False,
    )

    readme_path = input_path.parent / "README.txt"

    readme_text = (
        "Premium feasibility raw-data workspace\n"
        "======================================\n\n"
        f"Populate: {input_path.name}\n\n"
        "Each row represents one candidate 150-point-wide MNQ put "
        "credit spread at the Friday entry snapshot.\n"
        "Create separate rows for the standard and defensive candidate.\n"
        "Do not commit this raw options data to Git.\n"
        "See documentation/premium_feasibility_data_contract.md.\n"
    )

    readme_path.write_text(
        readme_text,
        encoding="utf-8",
    )

    print("Premium-feasibility workspace initialized.")
    print(f"CSV template: {input_path}")
    print(f"Workspace note: {readme_path}")
    print(
        "The raw-data directory is ignored by Git and remains local."
    )

    return 0


def validation_issue(
    issues: list[dict[str, Any]],
    severity: str,
    check: str,
    count: int,
    detail: str,
) -> None:
    """Append one validation issue."""

    issues.append(
        {
            "severity": severity,
            "check": check,
            "affected_rows": int(count),
            "detail": detail,
        }
    )


def validate_input(
    config: dict[str, Any],
    override_input: str | None,
) -> int:
    """Validate, normalize, and store premium-feasibility data."""

    input_path = resolve_project_path(
        override_input
        if override_input is not None
        else str(config["input_csv"])
    )

    if not input_path.is_file():
        print(f"ERROR: Input CSV not found: {input_path}")
        print(
            "Run with --initialize first, then populate the template."
        )
        return 1

    dataframe = pd.read_csv(input_path)

    if dataframe.empty:
        print(
            f"ERROR: Input CSV contains no data rows: {input_path}"
        )
        return 1

    missing_columns = [
        column
        for column in REQUIRED_COLUMNS
        if column not in dataframe.columns
    ]

    if missing_columns:
        print(
            "ERROR: Missing required columns: "
            + ", ".join(missing_columns)
        )
        return 1

    dataframe = dataframe.loc[
        :,
        list(REQUIRED_COLUMNS),
    ].copy()

    issues: list[dict[str, Any]] = []

    dataframe["entry_date"] = pd.to_datetime(
        dataframe["entry_date"],
        errors="coerce",
    ).dt.normalize()

    dataframe["expiration_date"] = pd.to_datetime(
        dataframe["expiration_date"],
        errors="coerce",
    ).dt.normalize()

    dataframe["quote_timestamp_utc"] = pd.to_datetime(
        dataframe["quote_timestamp_utc"],
        errors="coerce",
        utc=True,
    )

    for column in NUMERIC_COLUMNS:
        dataframe[column] = pd.to_numeric(
            dataframe[column],
            errors="coerce",
        )

    invalid_entry_dates = int(
        dataframe["entry_date"].isna().sum()
    )

    invalid_expiration_dates = int(
        dataframe["expiration_date"].isna().sum()
    )

    invalid_quote_timestamps = int(
        dataframe["quote_timestamp_utc"].isna().sum()
    )

    if invalid_entry_dates:
        validation_issue(
            issues,
            "ERROR",
            "entry_date_parse",
            invalid_entry_dates,
            "Entry dates must be valid ISO dates.",
        )

    if invalid_expiration_dates:
        validation_issue(
            issues,
            "ERROR",
            "expiration_date_parse",
            invalid_expiration_dates,
            "Expiration dates must be valid ISO dates.",
        )

    if invalid_quote_timestamps:
        validation_issue(
            issues,
            "ERROR",
            "quote_timestamp_parse",
            invalid_quote_timestamps,
            "Quote timestamps must be valid UTC timestamps.",
        )

    numeric_missing = dataframe[
        list(NUMERIC_COLUMNS)
    ].isna().any(axis=1)

    if numeric_missing.any():
        validation_issue(
            issues,
            "ERROR",
            "required_numeric_values",
            int(numeric_missing.sum()),
            "Required numeric fields cannot be blank or nonnumeric.",
        )

    candidate_types = {
        str(value).strip().lower()
        for value in config["allowed_candidate_types"]
    }

    dataframe["candidate_type"] = (
        dataframe["candidate_type"]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    invalid_candidate_type = ~dataframe[
        "candidate_type"
    ].isin(candidate_types)

    if invalid_candidate_type.any():
        validation_issue(
            issues,
            "ERROR",
            "candidate_type",
            int(invalid_candidate_type.sum()),
            "Candidate type must be one of: "
            + ", ".join(sorted(candidate_types)),
        )

    settlement_types = {
        str(value).strip().upper()
        for value in config["allowed_settlement_types"]
    }

    dataframe["settlement_type"] = (
        dataframe["settlement_type"]
        .astype(str)
        .str.strip()
        .str.upper()
    )

    invalid_settlement = ~dataframe[
        "settlement_type"
    ].isin(settlement_types)

    if invalid_settlement.any():
        validation_issue(
            issues,
            "ERROR",
            "settlement_type",
            int(invalid_settlement.sum()),
            "Settlement type must be one of: "
            + ", ".join(sorted(settlement_types)),
        )

    dataframe["calculated_dte_calendar"] = (
        dataframe["expiration_date"]
        - dataframe["entry_date"]
    ).dt.days

    dte_mismatch = (
        dataframe["calculated_dte_calendar"]
        != dataframe["dte_calendar"]
    )

    if dte_mismatch.fillna(False).any():
        validation_issue(
            issues,
            "ERROR",
            "dte_consistency",
            int(dte_mismatch.fillna(False).sum()),
            "dte_calendar must equal expiration_date minus entry_date.",
        )

    dte_min = int(config["dte_calendar_min"])
    dte_max = int(config["dte_calendar_max"])

    invalid_dte_range = ~dataframe[
        "dte_calendar"
    ].between(
        dte_min,
        dte_max,
        inclusive="both",
    )

    if invalid_dte_range.fillna(True).any():
        validation_issue(
            issues,
            "ERROR",
            "dte_range",
            int(invalid_dte_range.fillna(True).sum()),
            f"DTE must be between {dte_min} and {dte_max} calendar days.",
        )

    width_target = float(
        config["spread_width_points"]
    )

    width_tolerance = float(
        config["spread_width_tolerance_points"]
    )

    dataframe["spread_width_points"] = (
        dataframe["actual_short_strike"]
        - dataframe["actual_long_strike"]
    )

    invalid_width = (
        dataframe["spread_width_points"]
        - width_target
    ).abs() > width_tolerance

    if invalid_width.fillna(True).any():
        validation_issue(
            issues,
            "ERROR",
            "spread_width",
            int(invalid_width.fillna(True).sum()),
            f"Spread width must equal {width_target:.2f} points "
            f"within {width_tolerance:.4f}.",
        )

    invalid_positive_values = (
        (dataframe["underlying_price"] <= 0)
        | (dataframe["target_short_strike"] <= 0)
        | (dataframe["actual_short_strike"] <= 0)
        | (dataframe["actual_long_strike"] <= 0)
    )

    if invalid_positive_values.fillna(True).any():
        validation_issue(
            issues,
            "ERROR",
            "positive_market_values",
            int(invalid_positive_values.fillna(True).sum()),
            "Underlying and strike values must be positive.",
        )

    quote_columns = (
        "short_bid",
        "short_ask",
        "long_bid",
        "long_ask",
    )

    negative_quotes = (
        dataframe[list(quote_columns)] < 0
    ).any(axis=1)

    if negative_quotes.any():
        validation_issue(
            issues,
            "ERROR",
            "negative_quotes",
            int(negative_quotes.sum()),
            "Bid and ask values cannot be negative.",
        )

    crossed_quotes = (
        (dataframe["short_bid"] > dataframe["short_ask"])
        | (dataframe["long_bid"] > dataframe["long_ask"])
    )

    if crossed_quotes.fillna(True).any():
        validation_issue(
            issues,
            "ERROR",
            "crossed_quotes",
            int(crossed_quotes.fillna(True).sum()),
            "Each leg bid must be less than or equal to its ask.",
        )

    invalid_liquidity_values = (
        dataframe[
            [
                "short_open_interest",
                "long_open_interest",
                "short_volume",
                "long_volume",
            ]
        ] < 0
    ).any(axis=1)

    if invalid_liquidity_values.any():
        validation_issue(
            issues,
            "ERROR",
            "negative_liquidity_values",
            int(invalid_liquidity_values.sum()),
            "Open interest and volume cannot be negative.",
        )

    if bool(
        config.get(
            "require_actual_short_at_or_below_target",
            True,
        )
    ):
        above_target = (
            dataframe["actual_short_strike"]
            > dataframe["target_short_strike"]
            + width_tolerance
        )

        if above_target.fillna(True).any():
            validation_issue(
                issues,
                "ERROR",
                "strike_selection",
                int(above_target.fillna(True).sum()),
                "Actual short strike must be at or below its target.",
            )

    timezone_name = str(config["entry_timezone"])
    target_time_text = str(config["target_entry_time"])
    max_quote_age_minutes = float(
        config["maximum_quote_age_minutes"]
    )

    target_hour, target_minute = (
        int(part)
        for part in target_time_text.split(":")
    )

    local_timezone = ZoneInfo(timezone_name)

    dataframe["quote_timestamp_local"] = (
        dataframe["quote_timestamp_utc"]
        .dt.tz_convert(local_timezone)
    )

    dataframe["quote_local_date"] = (
        dataframe["quote_timestamp_local"]
        .dt.tz_localize(None)
        .dt.normalize()
    )

    wrong_local_date = (
        dataframe["quote_local_date"]
        != dataframe["entry_date"]
    )

    if wrong_local_date.fillna(True).any():
        validation_issue(
            issues,
            "ERROR",
            "quote_local_date",
            int(wrong_local_date.fillna(True).sum()),
            "The local quote date must match entry_date.",
        )

    target_local_timestamps = []

    for entry_date in dataframe["entry_date"]:
        if pd.isna(entry_date):
            target_local_timestamps.append(pd.NaT)
            continue

        target_naive = datetime.combine(
            entry_date.date(),
            time(
                hour=target_hour,
                minute=target_minute,
            ),
        )

        target_local_timestamps.append(
            pd.Timestamp(
                target_naive,
                tz=local_timezone,
            )
        )

    dataframe["target_entry_timestamp_local"] = pd.Series(
        target_local_timestamps,
        index=dataframe.index,
    )

    dataframe["quote_age_minutes"] = (
        dataframe["quote_timestamp_local"]
        - dataframe["target_entry_timestamp_local"]
    ).dt.total_seconds().abs() / 60.0

    stale_quotes = (
        dataframe["quote_age_minutes"]
        > max_quote_age_minutes
    )

    if stale_quotes.fillna(True).any():
        validation_issue(
            issues,
            "ERROR",
            "quote_age",
            int(stale_quotes.fillna(True).sum()),
            f"Quote must be within {max_quote_age_minutes:.1f} "
            f"minutes of {target_time_text} {timezone_name}.",
        )

    duplicate_key = [
        "entry_date",
        "expiration_date",
        "candidate_type",
        "actual_short_strike",
        "actual_long_strike",
    ]

    duplicates = dataframe.duplicated(
        subset=duplicate_key,
        keep=False,
    )

    if duplicates.any():
        validation_issue(
            issues,
            "ERROR",
            "duplicate_candidates",
            int(duplicates.sum()),
            "Duplicate candidate-spread rows were found.",
        )

    dataframe["short_mid"] = (
        dataframe["short_bid"]
        + dataframe["short_ask"]
    ) / 2.0

    dataframe["long_mid"] = (
        dataframe["long_bid"]
        + dataframe["long_ask"]
    ) / 2.0

    dataframe["natural_credit_points"] = (
        dataframe["short_bid"]
        - dataframe["long_ask"]
    )

    dataframe["mid_credit_points"] = (
        dataframe["short_mid"]
        - dataframe["long_mid"]
    )

    dataframe["optimistic_credit_points"] = (
        dataframe["short_ask"]
        - dataframe["long_bid"]
    )

    dataframe["short_leg_spread"] = (
        dataframe["short_ask"]
        - dataframe["short_bid"]
    )

    dataframe["long_leg_spread"] = (
        dataframe["long_ask"]
        - dataframe["long_bid"]
    )

    dataframe["actual_short_distance_fraction"] = (
        dataframe["underlying_price"]
        - dataframe["actual_short_strike"]
    ) / dataframe["underlying_price"]

    dataframe["target_short_distance_fraction"] = (
        dataframe["underlying_price"]
        - dataframe["target_short_strike"]
    ) / dataframe["underlying_price"]

    credit_floor = float(
        config["minimum_credit_points"]
    )

    dataframe["natural_credit_floor_pass"] = (
        dataframe["natural_credit_points"]
        >= credit_floor
    )

    dataframe["mid_credit_floor_pass"] = (
        dataframe["mid_credit_points"]
        >= credit_floor
    )

    negative_natural_credit = (
        dataframe["natural_credit_points"] < 0
    )

    if negative_natural_credit.any():
        validation_issue(
            issues,
            "WARNING",
            "negative_natural_credit",
            int(negative_natural_credit.sum()),
            "Natural fill credit is negative for these candidates.",
        )

    issue_frame = pd.DataFrame(
        issues,
        columns=[
            "severity",
            "check",
            "affected_rows",
            "detail",
        ],
    )

    results_dir = resolve_project_path(
        str(config["results_directory"])
    )

    results_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    report_path = (
        results_dir
        / "premium_input_validation_report.csv"
    )

    issue_frame.to_csv(
        report_path,
        index=False,
    )

    error_count = int(
        (issue_frame["severity"] == "ERROR").sum()
    ) if not issue_frame.empty else 0

    if error_count:
        print("Premium input validation failed.")
        print(
            issue_frame.to_string(
                index=False,
            )
        )
        print(f"\nValidation report: {report_path}")
        return 1

    processed_path = resolve_project_path(
        str(config["processed_parquet"])
    )

    processed_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    dataframe.to_parquet(
        processed_path,
        index=False,
        engine="pyarrow",
        compression="zstd",
    )

    connection = duckdb.connect(
        str(DATABASE_PATH)
    )

    try:
        connection.execute(
            "CREATE SCHEMA IF NOT EXISTS pcs"
        )

        connection.register(
            "premium_feasibility_dataframe",
            dataframe,
        )

        connection.execute(
            """
            CREATE OR REPLACE TABLE
                pcs.premium_feasibility_quotes AS
            SELECT *
            FROM premium_feasibility_dataframe
            ORDER BY
                entry_date,
                expiration_date,
                candidate_type
            """
        )
    finally:
        connection.close()

    summary = {
        "input_csv": str(input_path),
        "rows": len(dataframe),
        "entry_date_min": str(
            dataframe["entry_date"].min().date()
        ),
        "entry_date_max": str(
            dataframe["entry_date"].max().date()
        ),
        "standard_rows": int(
            (
                dataframe["candidate_type"]
                == "standard"
            ).sum()
        ),
        "defensive_rows": int(
            (
                dataframe["candidate_type"]
                == "defensive"
            ).sum()
        ),
        "natural_credit_floor_pass_rows": int(
            dataframe[
                "natural_credit_floor_pass"
            ].sum()
        ),
        "mid_credit_floor_pass_rows": int(
            dataframe[
                "mid_credit_floor_pass"
            ].sum()
        ),
        "median_natural_credit_points": float(
            dataframe[
                "natural_credit_points"
            ].median()
        ),
        "median_mid_credit_points": float(
            dataframe[
                "mid_credit_points"
            ].median()
        ),
    }

    summary_path = (
        results_dir
        / "premium_input_validation_summary.json"
    )

    summary_path.write_text(
        json.dumps(
            summary,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    print("Premium input validation passed.")
    print(
        f"Rows: {len(dataframe):,} | "
        f"Date range: "
        f"{dataframe['entry_date'].min().date()} to "
        f"{dataframe['entry_date'].max().date()}"
    )

    print(
        f"Standard candidates: "
        f"{summary['standard_rows']:,} | "
        f"Defensive candidates: "
        f"{summary['defensive_rows']:,}"
    )

    print(
        f"Natural credit floor pass: "
        f"{summary['natural_credit_floor_pass_rows']:,} | "
        f"Mid credit floor pass: "
        f"{summary['mid_credit_floor_pass_rows']:,}"
    )

    print(
        f"Median natural credit: "
        f"{summary['median_natural_credit_points']:.2f} | "
        f"Median mid credit: "
        f"{summary['median_mid_credit_points']:.2f}"
    )

    print(f"Processed output: {processed_path}")
    print(f"Validation report: {report_path}")
    print(f"Validation summary: {summary_path}")
    print(
        "DuckDB table: pcs.premium_feasibility_quotes"
    )

    return 0


def build_parser() -> argparse.ArgumentParser:
    """Build command-line argument parsing."""

    parser = argparse.ArgumentParser(
        description=(
            "Initialize or validate premium-feasibility "
            "spread-candidate data."
        )
    )

    action_group = parser.add_mutually_exclusive_group(
        required=True
    )

    action_group.add_argument(
        "--initialize",
        action="store_true",
        help="Create the ignored raw-data CSV template.",
    )

    action_group.add_argument(
        "--validate",
        action="store_true",
        help="Validate and normalize the populated CSV.",
    )

    parser.add_argument(
        "--input",
        help=(
            "Optional CSV path for --validate. "
            "Defaults to the configured input_csv."
        ),
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help=(
            "Replace the existing blank template during "
            "--initialize."
        ),
    )

    return parser


def main() -> int:
    """Run the requested workspace action."""

    parser = build_parser()
    arguments = parser.parse_args()

    try:
        config = load_config()

        if arguments.initialize:
            return initialize_template(
                config,
                arguments.force,
            )

        return validate_input(
            config,
            arguments.input,
        )

    except Exception as exc:
        print(f"ERROR: {exc}")
        return 1


if __name__ == "__main__":
    sys.exit(main())

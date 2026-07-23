"""Inspect Databento access, symbology, and pilot-request costs.

This script does not download historical time-series data. It authenticates,
checks dataset availability, selects recent elevated-regime Fridays from the
local DuckDB database, resolves likely MNQ option parent symbols using
Databento's free symbology endpoint, and estimates the cost of small definition
and quote requests.

Run:
    python scripts/inspect_databento_pilot.py
"""

from __future__ import annotations

import json
import os
import sys
from datetime import timedelta
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import duckdb
import pandas as pd
import yaml
from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "config" / "databento_pilot.yaml"
DATABASE_PATH = PROJECT_ROOT / "database" / "research.duckdb"
RESULTS_DIR = PROJECT_ROOT / "results" / "databento_pilot"
INSPECTION_OUTPUT = RESULTS_DIR / "databento_pilot_inspection.json"


def load_config() -> dict[str, Any]:
    """Load the pilot configuration."""

    if not CONFIG_PATH.is_file():
        raise FileNotFoundError(
            f"Missing configuration file: {CONFIG_PATH}"
        )

    with CONFIG_PATH.open("r", encoding="utf-8") as file_handle:
        return yaml.safe_load(file_handle)


def load_api_key() -> str:
    """Load the Databento API key without displaying it."""

    load_dotenv(PROJECT_ROOT / ".env")

    api_key = os.getenv("DATABENTO_API_KEY", "").strip()

    if not api_key:
        raise RuntimeError(
            "DATABENTO_API_KEY is missing from the project .env file."
        )

    if not api_key.startswith("db-"):
        raise RuntimeError(
            "DATABENTO_API_KEY does not have the expected db- prefix."
        )

    return api_key


def import_databento() -> Any:
    """Import Databento with a useful installation error."""

    try:
        import databento as db
    except ImportError as exc:
        raise RuntimeError(
            "The databento package is not installed. Run "
            "'python -m pip install -U databento' in the project "
            "virtual environment."
        ) from exc

    return db


def select_pilot_dates(
    config: dict[str, Any],
) -> pd.DataFrame:
    """Select recent elevated-regime Fridays from DuckDB."""

    if not DATABASE_PATH.is_file():
        raise FileNotFoundError(
            f"Missing local research database: {DATABASE_PATH}"
        )

    minimum_date = pd.Timestamp(
        config["minimum_pilot_entry_date"]
    )

    maximum_dates = int(
        config["pilot_date_count"]
    )

    connection = duckdb.connect(
        str(DATABASE_PATH),
        read_only=True,
    )

    try:
        pilot_dates = connection.execute(
            """
            SELECT
                entry_date,
                weekly_close,
                short_strike_proxy,
                vix_close,
                rolling4w_range_atr14_ratio,
                weekly_range_atr14_ratio,
                below_sma40,
                (
                    CAST(
                        rolling4w_range_atr14_ratio >= 1.20
                        AS INTEGER
                    )
                    + CAST(
                        weekly_range_atr14_ratio >= 1.20
                        AS INTEGER
                    )
                    + CAST(
                        below_sma40
                        AS INTEGER
                    )
                ) AS candidate_score
            FROM pcs.friday_regime_features
            WHERE complete_core_features = TRUE
              AND entry_date >= ?
              AND (
                    CAST(
                        rolling4w_range_atr14_ratio >= 1.20
                        AS INTEGER
                    )
                    + CAST(
                        weekly_range_atr14_ratio >= 1.20
                        AS INTEGER
                    )
                    + CAST(
                        below_sma40
                        AS INTEGER
                    )
                  ) >= 2
            ORDER BY entry_date DESC
            LIMIT ?
            """,
            [
                minimum_date.to_pydatetime(),
                maximum_dates,
            ],
        ).fetchdf()
    finally:
        connection.close()

    if pilot_dates.empty:
        raise RuntimeError(
            "No elevated-regime pilot dates were found."
        )

    pilot_dates["entry_date"] = pd.to_datetime(
        pilot_dates["entry_date"]
    )

    defensive_distance = float(
        config["defensive_additional_distance_fraction"]
    )

    pilot_dates["defensive_target_short_strike"] = (
        pilot_dates["short_strike_proxy"]
        - (
            pilot_dates["weekly_close"]
            * defensive_distance
        )
    )

    return pilot_dates.sort_values(
        "entry_date"
    ).reset_index(drop=True)


def extract_resolution_count(
    response: Any,
    symbol: str,
) -> int:
    """Count resolved mappings from a Databento response."""

    if not isinstance(response, dict):
        return 0

    result = response.get("result", {})

    if not isinstance(result, dict):
        return 0

    mappings = result.get(symbol, [])

    if not isinstance(mappings, list):
        return 0

    count = 0

    for interval in mappings:
        if not isinstance(interval, dict):
            continue

        resolved = interval.get("s")

        if resolved is None:
            continue

        if isinstance(resolved, list):
            count += len(resolved)
        else:
            count += 1

    return count


def inspect_parent_symbol(
    client: Any,
    dataset: str,
    symbol: str,
    pilot_date: pd.Timestamp,
    quote_start_utc: pd.Timestamp,
    quote_end_utc: pd.Timestamp,
    schemas: list[str],
) -> dict[str, Any]:
    """Resolve a parent symbol and estimate request costs."""

    output: dict[str, Any] = {
        "symbol": symbol,
        "resolved": False,
        "resolution_count": 0,
        "resolution_error": None,
        "cost_estimates_usd": {},
    }

    try:
        resolution = client.symbology.resolve(
            dataset=dataset,
            symbols=[symbol],
            stype_in="parent",
            stype_out="instrument_id",
            start_date=pilot_date.date().isoformat(),
        )

        resolution_count = extract_resolution_count(
            resolution,
            symbol,
        )

        output["resolution_count"] = resolution_count
        output["resolved"] = resolution_count > 0
    except Exception as exc:
        output["resolution_error"] = str(exc)
        return output

    if not output["resolved"]:
        return output

    definition_start = pilot_date.normalize()
    definition_end = (
        definition_start
        + pd.Timedelta(days=1)
    )

    requests = [
        (
            "definition_one_day",
            "definition",
            definition_start,
            definition_end,
        )
    ]

    for schema in schemas:
        requests.append(
            (
                f"{schema}_ten_minutes",
                schema,
                quote_start_utc,
                quote_end_utc,
            )
        )

    for label, schema, start, end in requests:
        try:
            cost = client.metadata.get_cost(
                dataset=dataset,
                symbols=[symbol],
                stype_in="parent",
                schema=schema,
                start=start.isoformat(),
                end=end.isoformat(),
            )

            output["cost_estimates_usd"][label] = float(
                cost
            )
        except Exception as exc:
            output["cost_estimates_usd"][label] = {
                "error": str(exc)
            }

    return output


def main() -> int:
    """Run the no-download Databento inspection."""

    print(
        "MNQ PCS Regime Research — Databento Pilot Inspection"
    )
    print("=" * 58)

    try:
        config = load_config()
        api_key = load_api_key()
        db = import_databento()

        dataset = str(config["dataset"])
        client = db.Historical(api_key)

        datasets = client.metadata.list_datasets()

        if dataset not in datasets:
            raise RuntimeError(
                f"Dataset {dataset} is not available to this account."
            )

        dataset_range = client.metadata.get_dataset_range(
            dataset=dataset
        )

        pilot_dates = select_pilot_dates(config)

        print(
            f"Databento package: "
            f"{getattr(db, '__version__', 'version unavailable')}"
        )

        print(
            f"Authentication: successful | Dataset: {dataset}"
        )

        print(
            f"Dataset range: {dataset_range.get('start')} to "
            f"{dataset_range.get('end')}"
        )

        schema_ranges = dataset_range.get("schema", {})

        print("\nRelevant schema availability")
        print("-" * 76)

        for schema in (
            "definition",
            *config["quote_schemas_to_estimate"],
        ):
            availability = schema_ranges.get(
                schema,
                "not listed",
            )

            print(
                f"{schema:<12} | {availability}"
            )

        print("\nSelected elevated-regime pilot Fridays")
        print("-" * 112)

        for row in pilot_dates.itertuples(index=False):
            print(
                f"{row.entry_date.date()} | "
                f"Score={int(row.candidate_score)} | "
                f"NDX={row.weekly_close:,.2f} | "
                f"Standard target={row.short_strike_proxy:,.2f} | "
                f"3% defensive target="
                f"{row.defensive_target_short_strike:,.2f}"
            )

        latest_date = pilot_dates[
            "entry_date"
        ].max()

        local_timezone = ZoneInfo(
            str(config["entry_timezone"])
        )

        entry_time = str(
            config["target_entry_time"]
        )

        entry_hour, entry_minute = (
            int(part)
            for part in entry_time.split(":")
        )

        target_local = pd.Timestamp(
            year=latest_date.year,
            month=latest_date.month,
            day=latest_date.day,
            hour=entry_hour,
            minute=entry_minute,
            tz=local_timezone,
        )

        window_minutes = int(
            config["cost_estimate_window_minutes"]
        )

        half_window = pd.Timedelta(
            minutes=window_minutes / 2
        )

        quote_start_utc = (
            target_local - half_window
        ).tz_convert("UTC")

        quote_end_utc = (
            target_local + half_window
        ).tz_convert("UTC")

        print(
            f"\nSymbology and cost inspection date: "
            f"{latest_date.date()}"
        )

        print(
            f"Quote window: {quote_start_utc.isoformat()} to "
            f"{quote_end_utc.isoformat()}"
        )

        futures_symbol = str(
            config["continuous_futures_symbol"]
        )

        futures_resolution: dict[str, Any] = {
            "symbol": futures_symbol,
            "resolved": False,
            "resolution_count": 0,
            "error": None,
        }

        try:
            response = client.symbology.resolve(
                dataset=dataset,
                symbols=[futures_symbol],
                stype_in="continuous",
                stype_out="instrument_id",
                start_date=latest_date.date().isoformat(),
            )

            count = extract_resolution_count(
                response,
                futures_symbol,
            )

            futures_resolution[
                "resolution_count"
            ] = count

            futures_resolution[
                "resolved"
            ] = count > 0
        except Exception as exc:
            futures_resolution["error"] = str(exc)

        print("\nUnderlying futures resolution")
        print("-" * 76)

        print(
            f"{futures_symbol}: "
            f"{'resolved' if futures_resolution['resolved'] else 'not resolved'} "
            f"({futures_resolution['resolution_count']} mapping(s))"
        )

        parent_results: list[dict[str, Any]] = []

        print("\nOption parent-symbol inspection")
        print("-" * 118)

        for parent_symbol in config[
            "option_parent_candidates"
        ]:
            result = inspect_parent_symbol(
                client=client,
                dataset=dataset,
                symbol=str(parent_symbol),
                pilot_date=latest_date,
                quote_start_utc=quote_start_utc,
                quote_end_utc=quote_end_utc,
                schemas=[
                    str(schema)
                    for schema in config[
                        "quote_schemas_to_estimate"
                    ]
                ],
            )

            parent_results.append(result)

            status = (
                "resolved"
                if result["resolved"]
                else "not resolved"
            )

            print(
                f"{parent_symbol:<12} | "
                f"{status:<12} | "
                f"Mappings={result['resolution_count']:>4}"
            )

            if result["resolution_error"]:
                print(
                    f"  Resolution error: "
                    f"{result['resolution_error']}"
                )

            for label, value in result[
                "cost_estimates_usd"
            ].items():
                if isinstance(value, dict):
                    print(
                        f"  {label:<28} | "
                        f"estimate unavailable: "
                        f"{value['error']}"
                    )
                else:
                    print(
                        f"  {label:<28} | "
                        f"${value:,.6f}"
                    )

        resolved_parents = [
            result
            for result in parent_results
            if result["resolved"]
        ]

        RESULTS_DIR.mkdir(
            parents=True,
            exist_ok=True,
        )

        inspection = {
            "inspection_timestamp_utc": (
                pd.Timestamp.now(tz="UTC").isoformat()
            ),
            "dataset": dataset,
            "dataset_range": dataset_range,
            "pilot_dates": (
                pilot_dates.assign(
                    entry_date=pilot_dates[
                        "entry_date"
                    ].dt.strftime("%Y-%m-%d")
                ).to_dict(orient="records")
            ),
            "quote_window_start_utc": (
                quote_start_utc.isoformat()
            ),
            "quote_window_end_utc": (
                quote_end_utc.isoformat()
            ),
            "futures_resolution": futures_resolution,
            "option_parent_results": parent_results,
            "resolved_option_parents": [
                result["symbol"]
                for result in resolved_parents
            ],
            "historical_data_downloaded": False,
        }

        INSPECTION_OUTPUT.write_text(
            json.dumps(
                inspection,
                indent=2,
                default=str,
            )
            + "\n",
            encoding="utf-8",
        )

        print("\nInspection result")
        print("-" * 76)

        if resolved_parents:
            print(
                "Resolved option parent candidate(s): "
                + ", ".join(
                    result["symbol"]
                    for result in resolved_parents
                )
            )
        else:
            print(
                "No configured option parent candidate resolved. "
                "We will use a definition-discovery fallback next."
            )

        print(
            "No historical time-series data was downloaded."
        )

        print(
            f"Inspection output: {INSPECTION_OUTPUT}"
        )

        return 0

    except Exception as exc:
        print(f"\nERROR: {exc}")
        return 1


if __name__ == "__main__":
    sys.exit(main())

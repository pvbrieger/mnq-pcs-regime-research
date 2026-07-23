"""Estimate Databento definition costs for two additional pilot Fridays.

This script makes metadata calls only. It does not download historical data.

Run:
    python scripts/estimate_databento_multidate_definitions.py
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import yaml
from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = (
    PROJECT_ROOT
    / "config"
    / "databento_multidate_pilot.yaml"
)
RESULTS_DIR = (
    PROJECT_ROOT
    / "results"
    / "databento_multidate_pilot"
)
OUTPUT_PATH = (
    RESULTS_DIR
    / "definition_cost_estimates.json"
)


def load_config() -> dict[str, Any]:
    """Load pilot configuration."""

    if not CONFIG_PATH.is_file():
        raise FileNotFoundError(
            f"Missing configuration file: {CONFIG_PATH}"
        )

    with CONFIG_PATH.open(
        "r",
        encoding="utf-8",
    ) as file_handle:
        return yaml.safe_load(file_handle)


def load_api_key() -> str:
    """Load the Databento API key without displaying it."""

    load_dotenv(
        PROJECT_ROOT / ".env",
        override=True,
    )

    key = os.getenv(
        "DATABENTO_API_KEY",
        "",
    ).strip()

    if not key:
        raise RuntimeError(
            "DATABENTO_API_KEY is missing from .env."
        )

    if not key.startswith("db-"):
        raise RuntimeError(
            "DATABENTO_API_KEY does not have the expected db- prefix."
        )

    return key


def import_databento() -> Any:
    """Import Databento with a useful error."""

    try:
        import databento as db
    except ImportError as exc:
        raise RuntimeError(
            "The databento package is not installed."
        ) from exc

    return db


def estimate_one_date(
    client: Any,
    dataset: str,
    pilot: dict[str, Any],
) -> dict[str, Any]:
    """Estimate one UTC calendar day of ALL_SYMBOLS definitions."""

    entry_date = pd.Timestamp(
        pilot["entry_date"]
    ).normalize()

    start_utc = entry_date.tz_localize(
        "UTC"
    )

    end_utc = (
        start_utc
        + pd.Timedelta(days=1)
    )

    result: dict[str, Any] = {
        "entry_date": entry_date.date().isoformat(),
        "score": int(pilot["score"]),
        "ndx_close": float(pilot["ndx_close"]),
        "standard_target_short_strike": float(
            pilot["standard_target_short_strike"]
        ),
        "defensive_target_short_strike": float(
            pilot["defensive_target_short_strike"]
        ),
        "request_start_utc": start_utc.isoformat(),
        "request_end_utc": end_utc.isoformat(),
        "estimated_cost_usd": None,
        "estimated_record_count": None,
        "cost_error": None,
        "record_count_error": None,
    }

    try:
        result["estimated_cost_usd"] = float(
            client.metadata.get_cost(
                dataset=dataset,
                symbols="ALL_SYMBOLS",
                stype_in="raw_symbol",
                schema="definition",
                start=start_utc.isoformat(),
                end=end_utc.isoformat(),
            )
        )
    except Exception as exc:
        result["cost_error"] = str(exc)

    try:
        result["estimated_record_count"] = int(
            client.metadata.get_record_count(
                dataset=dataset,
                symbols="ALL_SYMBOLS",
                stype_in="raw_symbol",
                schema="definition",
                start=start_utc.isoformat(),
                end=end_utc.isoformat(),
            )
        )
    except Exception as exc:
        result["record_count_error"] = str(exc)

    return result


def main() -> int:
    """Estimate both pilot dates without downloading data."""

    print(
        "MNQ PCS Regime Research — Multi-Date Definition Estimate"
    )
    print("=" * 64)

    try:
        config = load_config()
        db = import_databento()
        client = db.Historical(
            load_api_key()
        )

        dataset = str(
            config["dataset"]
        )

        pilots = list(
            config["pilots"]
        )

        if not pilots:
            raise ValueError(
                "No pilot dates are configured."
            )

        estimates: list[dict[str, Any]] = []

        print(
            f"Dataset: {dataset}"
        )

        print(
            f"Pilot dates: {len(pilots)}"
        )

        print("\nPer-date definition estimates")
        print("-" * 132)

        for pilot in pilots:
            estimate = estimate_one_date(
                client=client,
                dataset=dataset,
                pilot=pilot,
            )

            estimates.append(
                estimate
            )

            cost = estimate[
                "estimated_cost_usd"
            ]

            records = estimate[
                "estimated_record_count"
            ]

            cost_text = (
                f"${cost:,.6f}"
                if cost is not None
                else "unavailable"
            )

            records_text = (
                f"{records:,}"
                if records is not None
                else "unavailable"
            )

            print(
                f"{estimate['entry_date']} | "
                f"Score={estimate['score']} | "
                f"NDX={estimate['ndx_close']:>10,.2f} | "
                f"Std={estimate['standard_target_short_strike']:>10,.2f} | "
                f"Def={estimate['defensive_target_short_strike']:>10,.2f} | "
                f"Cost={cost_text:<14} | "
                f"Records={records_text}"
            )

            if estimate["cost_error"]:
                print(
                    f"  Cost error: "
                    f"{estimate['cost_error']}"
                )

            if estimate["record_count_error"]:
                print(
                    f"  Record-count error: "
                    f"{estimate['record_count_error']}"
                )

        successful_costs = [
            float(row["estimated_cost_usd"])
            for row in estimates
            if row["estimated_cost_usd"]
            is not None
        ]

        total_cost = (
            sum(successful_costs)
            if len(successful_costs)
            == len(estimates)
            else None
        )

        successful_records = [
            int(row["estimated_record_count"])
            for row in estimates
            if row["estimated_record_count"]
            is not None
        ]

        total_records = (
            sum(successful_records)
            if len(successful_records)
            == len(estimates)
            else None
        )

        print("\nCombined estimate")
        print("-" * 72)

        print(
            "Total cost: "
            + (
                f"${total_cost:,.6f}"
                if total_cost is not None
                else "unavailable"
            )
        )

        print(
            "Total records: "
            + (
                f"{total_records:,}"
                if total_records is not None
                else "unavailable"
            )
        )

        RESULTS_DIR.mkdir(
            parents=True,
            exist_ok=True,
        )

        payload = {
            "dataset": dataset,
            "schema": "definition",
            "symbols": "ALL_SYMBOLS",
            "stype_in": "raw_symbol",
            "pilots": estimates,
            "combined_estimated_cost_usd": (
                total_cost
            ),
            "combined_estimated_record_count": (
                total_records
            ),
            "historical_data_downloaded": False,
        }

        OUTPUT_PATH.write_text(
            json.dumps(
                payload,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )

        print("\nNo historical data was downloaded.")
        print(
            f"Estimate output: {OUTPUT_PATH}"
        )

        return 0

    except Exception as exc:
        print(
            f"\nERROR: {type(exc).__name__}: {exc}"
        )
        return 1


if __name__ == "__main__":
    sys.exit(main())

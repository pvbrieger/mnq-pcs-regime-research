"""Estimate exact-symbol BBO costs for the April pilot dates.

This script reads the quote manifest created by the multi-date definition
download. It checks Databento's dataset condition for each date and estimates
the cost and record count for a ten-minute BBO-1s request containing only:

- the four option legs for every jointly executable expiration; and
- the corresponding MNQ futures contract.

No historical quote data is downloaded.

Run:
    python scripts/estimate_databento_multidate_quotes.py
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd
import yaml
from dotenv import load_dotenv


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "config" / "databento_multidate_pilot.yaml"
MANIFEST_PATH = (
    ROOT
    / "results"
    / "databento_multidate_pilot"
    / "quote_request_manifest.csv"
)
RESULTS_DIR = (
    ROOT
    / "results"
    / "databento_multidate_pilot"
)
OUTPUT_PATH = RESULTS_DIR / "quote_cost_estimates.json"
SUMMARY_PATH = RESULTS_DIR / "quote_cost_estimates.txt"


def load_config() -> dict[str, Any]:
    """Load pilot configuration."""

    if not CONFIG_PATH.is_file():
        raise FileNotFoundError(
            f"Missing configuration: {CONFIG_PATH}"
        )

    with CONFIG_PATH.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def load_manifest() -> pd.DataFrame:
    """Load and validate the exact-symbol quote manifest."""

    if not MANIFEST_PATH.is_file():
        raise FileNotFoundError(
            f"Missing quote manifest: {MANIFEST_PATH}"
        )

    frame = pd.read_csv(MANIFEST_PATH)

    required = {
        "entry_date",
        "expiration",
        "dte_calendar",
        "underlying",
        "candidate_type",
        "leg_name",
        "selected_short_strike",
        "selected_long_strike",
        "raw_symbol",
    }

    missing = sorted(required.difference(frame.columns))

    if missing:
        raise ValueError(
            "Quote manifest is missing fields: "
            + ", ".join(missing)
        )

    frame["entry_date"] = (
        frame["entry_date"]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    frame["expiration"] = pd.to_datetime(
        frame["expiration"],
        errors="coerce",
        utc=True,
    )

    for column in (
        "underlying",
        "candidate_type",
        "leg_name",
        "raw_symbol",
    ):
        frame[column] = (
            frame[column]
            .fillna("")
            .astype(str)
            .str.strip()
        )

    if frame.empty:
        raise ValueError(
            "The quote manifest contains no executable spreads."
        )

    return frame


def api_key() -> str:
    """Load the Databento API key without displaying it."""

    load_dotenv(ROOT / ".env", override=True)

    value = os.getenv(
        "DATABENTO_API_KEY",
        "",
    ).strip()

    if not value.startswith("db-"):
        raise RuntimeError(
            "A valid DATABENTO_API_KEY is required in .env."
        )

    return value


def quote_window(
    entry_date: str,
    config: dict[str, Any],
) -> tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp]:
    """Return local target plus UTC ten-minute quote window."""

    date = pd.Timestamp(entry_date)

    hour, minute = (
        int(part)
        for part in str(
            config["target_entry_time"]
        ).split(":")
    )

    timezone = ZoneInfo(
        str(config["entry_timezone"])
    )

    target_local = pd.Timestamp(
        year=date.year,
        month=date.month,
        day=date.day,
        hour=hour,
        minute=minute,
        tz=timezone,
    )

    half_window = pd.Timedelta(
        minutes=float(
            config.get(
                "quote_window_minutes",
                10,
            )
        )
        / 2.0
    )

    return (
        target_local,
        (
            target_local - half_window
        ).tz_convert("UTC"),
        (
            target_local + half_window
        ).tz_convert("UTC"),
    )


def condition_for_date(
    client: Any,
    dataset: str,
    entry_date: str,
) -> dict[str, Any]:
    """Return Databento's day-level dataset condition."""

    rows = client.metadata.get_dataset_condition(
        dataset=dataset,
        start_date=entry_date,
        end_date=entry_date,
    )

    if not rows:
        return {
            "date": entry_date,
            "condition": "unknown",
            "last_modified_date": None,
        }

    row = rows[0]

    if isinstance(row, dict):
        return {
            "date": str(
                row.get("date", entry_date)
            ),
            "condition": str(
                row.get("condition", "unknown")
            ),
            "last_modified_date": (
                row.get("last_modified_date")
            ),
        }

    return {
        "date": str(
            getattr(row, "date", entry_date)
        ),
        "condition": str(
            getattr(
                row,
                "condition",
                "unknown",
            )
        ),
        "last_modified_date": getattr(
            row,
            "last_modified_date",
            None,
        ),
    }


def estimate_request(
    client: Any,
    dataset: str,
    symbols: list[str],
    start_utc: pd.Timestamp,
    end_utc: pd.Timestamp,
) -> dict[str, Any]:
    """Estimate one exact-symbol BBO-1s request."""

    kwargs = dict(
        dataset=dataset,
        symbols=symbols,
        stype_in="raw_symbol",
        schema="bbo-1s",
        start=start_utc.isoformat(),
        end=end_utc.isoformat(),
    )

    result: dict[str, Any] = {
        "estimated_cost_usd": None,
        "estimated_record_count": None,
        "cost_error": None,
        "record_count_error": None,
    }

    try:
        result["estimated_cost_usd"] = float(
            client.metadata.get_cost(**kwargs)
        )
    except Exception as exc:
        result["cost_error"] = str(exc)

    try:
        result["estimated_record_count"] = int(
            client.metadata.get_record_count(
                **kwargs
            )
        )
    except Exception as exc:
        result["record_count_error"] = str(exc)

    return result


def main() -> int:
    """Check conditions and estimate exact-symbol quote requests."""

    import databento as db

    print(
        "MNQ PCS Regime Research — Multi-Date Quote Estimate"
    )
    print("=" * 60)

    config = load_config()
    manifest = load_manifest()
    client = db.Historical(api_key())
    dataset = str(config["dataset"])

    results: list[dict[str, Any]] = []

    print("\nDate, condition, and selected instruments")
    print("-" * 152)

    for entry_date, date_rows in manifest.groupby(
        "entry_date",
        sort=True,
    ):
        condition = condition_for_date(
            client,
            dataset,
            entry_date,
        )

        target_local, start_utc, end_utc = quote_window(
            entry_date,
            config,
        )

        option_symbols = sorted(
            date_rows["raw_symbol"]
            .drop_duplicates()
            .tolist()
        )

        underlying_symbols = sorted(
            date_rows["underlying"]
            .drop_duplicates()
            .tolist()
        )

        all_symbols = sorted(
            set(
                option_symbols
                + underlying_symbols
            )
        )

        expirations = (
            date_rows[
                [
                    "expiration",
                    "dte_calendar",
                ]
            ]
            .drop_duplicates()
            .sort_values("expiration")
        )

        estimate = estimate_request(
            client,
            dataset,
            all_symbols,
            start_utc,
            end_utc,
        )

        cost = estimate["estimated_cost_usd"]
        records = estimate[
            "estimated_record_count"
        ]

        print(
            f"{entry_date} | "
            f"Condition={condition['condition']:<9} | "
            f"Expirations={len(expirations):>2} | "
            f"Options={len(option_symbols):>2} | "
            f"Underlying={', '.join(underlying_symbols):<8} | "
            f"Cost="
            f"{f'${cost:,.6f}' if cost is not None else 'unavailable':<14} | "
            f"Records="
            f"{f'{records:,}' if records is not None else 'unavailable'}"
        )

        for expiration in expirations.itertuples(
            index=False
        ):
            exp_rows = date_rows.loc[
                date_rows["expiration"]
                == expiration.expiration
            ]

            standard = (
                exp_rows.loc[
                    exp_rows["candidate_type"]
                    == "standard",
                    [
                        "selected_short_strike",
                        "selected_long_strike",
                    ],
                ]
                .drop_duplicates()
                .iloc[0]
            )

            defensive = (
                exp_rows.loc[
                    exp_rows["candidate_type"]
                    == "defensive",
                    [
                        "selected_short_strike",
                        "selected_long_strike",
                    ],
                ]
                .drop_duplicates()
                .iloc[0]
            )

            print(
                f"  Exp={expiration.expiration} | "
                f"DTE={int(expiration.dte_calendar):>2} | "
                f"Standard="
                f"{standard['selected_short_strike']:,.0f}/"
                f"{standard['selected_long_strike']:,.0f} | "
                f"Defensive="
                f"{defensive['selected_short_strike']:,.0f}/"
                f"{defensive['selected_long_strike']:,.0f}"
            )

        print(
            "  Symbols: "
            + ", ".join(all_symbols)
        )

        results.append(
            {
                "entry_date": entry_date,
                "dataset_condition": condition,
                "target_entry_timestamp_local": (
                    target_local.isoformat()
                ),
                "quote_window_start_utc": (
                    start_utc.isoformat()
                ),
                "quote_window_end_utc": (
                    end_utc.isoformat()
                ),
                "option_symbols": option_symbols,
                "underlying_symbols": (
                    underlying_symbols
                ),
                "all_symbols": all_symbols,
                **estimate,
            }
        )

    successful_costs = [
        float(row["estimated_cost_usd"])
        for row in results
        if row["estimated_cost_usd"]
        is not None
    ]

    total_cost = (
        sum(successful_costs)
        if len(successful_costs)
        == len(results)
        else None
    )

    print("\nCombined quote estimate")
    print("-" * 76)

    print(
        "Total estimated cost: "
        + (
            f"${total_cost:,.6f}"
            if total_cost is not None
            else "unavailable"
        )
    )

    print(
        "\nCondition policy: degraded dates may be used for "
        "exploratory evidence, but not as clean validation dates."
    )

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    payload = {
        "dataset": dataset,
        "schema": "bbo-1s",
        "dates": results,
        "combined_estimated_cost_usd": total_cost,
        "historical_quotes_downloaded": False,
    }

    OUTPUT_PATH.write_text(
        json.dumps(
            payload,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    summary_lines = [
        "MNQ PCS REGIME RESEARCH — MULTI-DATE QUOTE ESTIMATE",
        "=" * 62,
        "",
    ]

    for row in results:
        summary_lines.append(
            f"{row['entry_date']} | "
            f"Condition="
            f"{row['dataset_condition']['condition']} | "
            f"Symbols={len(row['all_symbols'])} | "
            f"Cost={row['estimated_cost_usd']} | "
            f"Records={row['estimated_record_count']}"
        )

    summary_lines.extend(
        [
            "",
            (
                "Combined estimated cost: "
                f"{total_cost}"
            ),
            (
                "Degraded dates are exploratory only."
            ),
        ]
    )

    SUMMARY_PATH.write_text(
        "\n".join(summary_lines) + "\n",
        encoding="utf-8",
    )

    print("\nNo historical quote data was downloaded.")
    print(f"Estimate output: {OUTPUT_PATH}")
    print(f"Summary: {SUMMARY_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

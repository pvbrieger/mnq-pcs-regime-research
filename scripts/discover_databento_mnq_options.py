"""Discover MNQ option definitions from Databento.

The configured parent symbols did not resolve, so this script uses Databento's
documented fallback: request the point-in-time definition snapshot for
ALL_SYMBOLS and filter locally for options on MNQ futures.

Commands
--------
Estimate cost only; no time-series download:
    python scripts/discover_databento_mnq_options.py --estimate

Download only after reviewing the estimate:
    python scripts/discover_databento_mnq_options.py --download --max-cost-usd 1.00
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import yaml
from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = (
    PROJECT_ROOT
    / "config"
    / "databento_definition_discovery.yaml"
)
RESULTS_DIR = (
    PROJECT_ROOT
    / "results"
    / "databento_definition_discovery"
)
RAW_DIR = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "databento"
    / "definitions"
)

ESTIMATE_OUTPUT = RESULTS_DIR / "definition_cost_estimate.json"
DISCOVERY_OUTPUT = RESULTS_DIR / "mnq_option_definition_candidates.csv"
SUMMARY_OUTPUT = RESULTS_DIR / "mnq_definition_discovery_summary.json"


def load_config() -> dict[str, Any]:
    """Load configuration."""

    if not CONFIG_PATH.is_file():
        raise FileNotFoundError(
            f"Missing configuration file: {CONFIG_PATH}"
        )

    with CONFIG_PATH.open("r", encoding="utf-8") as file_handle:
        return yaml.safe_load(file_handle)


def load_api_key() -> str:
    """Load the Databento API key without displaying it."""

    load_dotenv(PROJECT_ROOT / ".env", override=True)

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
    """Import Databento."""

    try:
        import databento as db
    except ImportError as exc:
        raise RuntimeError(
            "The databento package is not installed."
        ) from exc

    return db


def request_window(
    config: dict[str, Any],
) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Return one UTC-midnight definition snapshot window."""

    entry_date = pd.Timestamp(
        config["pilot_entry_date"]
    ).normalize()

    start = entry_date.tz_localize("UTC")
    end = start + pd.Timedelta(days=1)

    return start, end


def get_cost(
    client: Any,
    config: dict[str, Any],
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> float:
    """Estimate the ALL_SYMBOLS definition request cost."""

    cost = client.metadata.get_cost(
        dataset=str(config["dataset"]),
        symbols="ALL_SYMBOLS",
        schema="definition",
        start=start.isoformat(),
        end=end.isoformat(),
    )

    return float(cost)


def save_estimate(
    config: dict[str, Any],
    start: pd.Timestamp,
    end: pd.Timestamp,
    cost: float,
) -> None:
    """Write the cost estimate."""

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    payload = {
        "dataset": str(config["dataset"]),
        "schema": "definition",
        "symbols": "ALL_SYMBOLS",
        "start_utc": start.isoformat(),
        "end_utc": end.isoformat(),
        "estimated_cost_usd": cost,
        "historical_data_downloaded": False,
    }

    ESTIMATE_OUTPUT.write_text(
        json.dumps(payload, indent=2) + "\n",
        encoding="utf-8",
    )


def normalize_text(
    series: pd.Series,
) -> pd.Series:
    """Normalize text safely."""

    return (
        series
        .fillna("")
        .astype(str)
        .str.strip()
    )


def candidate_filter(
    definitions: pd.DataFrame,
    config: dict[str, Any],
) -> pd.DataFrame:
    """Filter point-in-time definitions to likely MNQ put options."""

    required_columns = {
        "raw_symbol",
        "security_type",
        "instrument_class",
        "strike_price",
        "expiration",
    }

    missing = sorted(
        required_columns.difference(definitions.columns)
    )

    if missing:
        raise ValueError(
            "Definition data is missing required fields: "
            + ", ".join(missing)
        )

    frame = definitions.copy()

    if "underlying" not in frame.columns:
        frame["underlying"] = ""

    if "asset" not in frame.columns:
        frame["asset"] = ""

    frame["raw_symbol"] = normalize_text(
        frame["raw_symbol"]
    )

    frame["security_type"] = normalize_text(
        frame["security_type"]
    ).str.upper()

    frame["instrument_class"] = normalize_text(
        frame["instrument_class"]
    ).str.upper()

    frame["underlying"] = normalize_text(
        frame["underlying"]
    )

    frame["asset"] = normalize_text(
        frame["asset"]
    )

    frame["strike_price"] = pd.to_numeric(
        frame["strike_price"],
        errors="coerce",
    )

    frame["expiration"] = pd.to_datetime(
        frame["expiration"],
        errors="coerce",
        utc=True,
    )

    entry_date = pd.Timestamp(
        config["pilot_entry_date"]
    ).normalize()

    frame["expiration_date"] = (
        frame["expiration"]
        .dt.tz_convert(
            ZoneInfo(
                str(config["entry_timezone"])
            )
        )
        .dt.tz_localize(None)
        .dt.normalize()
    )

    frame["dte_calendar"] = (
        frame["expiration_date"]
        - entry_date
    ).dt.days

    minimum_dte = int(
        config["dte_calendar_min"]
    )

    maximum_dte = int(
        config["dte_calendar_max"]
    )

    roots = tuple(
        str(root).upper()
        for root in config["underlying_root_candidates"]
    )

    underlying_upper = frame[
        "underlying"
    ].str.upper()

    asset_upper = frame[
        "asset"
    ].str.upper()

    raw_upper = frame[
        "raw_symbol"
    ].str.upper()

    root_match = pd.Series(
        False,
        index=frame.index,
    )

    for root in roots:
        root_match |= underlying_upper.str.startswith(
            root
        )

        root_match |= asset_upper.str.startswith(
            root
        )

        root_match |= raw_upper.str.startswith(
            root
        )

    filtered = frame.loc[
        (
            frame["security_type"]
            == "OOF"
        )
        & (
            frame["instrument_class"]
            == "P"
        )
        & frame["dte_calendar"].between(
            minimum_dte,
            maximum_dte,
            inclusive="both",
        )
        & root_match
    ].copy()

    standard_target = float(
        config["standard_target_short_strike"]
    )

    defensive_target = float(
        config["defensive_target_short_strike"]
    )

    strike_band = float(
        config["strike_search_band_points"]
    )

    minimum_relevant_strike = (
        min(
            standard_target,
            defensive_target,
        )
        - strike_band
    )

    maximum_relevant_strike = (
        max(
            standard_target,
            defensive_target,
        )
        + strike_band
    )

    filtered = filtered.loc[
        filtered["strike_price"].between(
            minimum_relevant_strike,
            maximum_relevant_strike,
            inclusive="both",
        )
    ].copy()

    filtered["standard_target_distance_points"] = (
        filtered["strike_price"]
        - standard_target
    ).abs()

    filtered["defensive_target_distance_points"] = (
        filtered["strike_price"]
        - defensive_target
    ).abs()

    filtered["expiration_local"] = (
        filtered["expiration"]
        .dt.tz_convert(
            ZoneInfo(
                str(config["entry_timezone"])
            )
        )
    )

    preferred_columns = [
        "raw_symbol",
        "instrument_id",
        "underlying",
        "asset",
        "security_type",
        "instrument_class",
        "strike_price",
        "expiration",
        "expiration_local",
        "expiration_date",
        "dte_calendar",
        "activation",
        "standard_target_distance_points",
        "defensive_target_distance_points",
    ]

    available_columns = [
        column
        for column in preferred_columns
        if column in filtered.columns
    ]

    return filtered.loc[
        :,
        available_columns,
    ].sort_values(
        [
            "expiration",
            "underlying",
            "strike_price",
            "raw_symbol",
        ]
    ).reset_index(drop=True)


def choose_nearest_below(
    candidates: pd.DataFrame,
    target: float,
) -> pd.DataFrame:
    """Choose the nearest listed strike at or below a target by expiration."""

    eligible = candidates.loc[
        candidates["strike_price"] <= target
    ].copy()

    if eligible.empty:
        return eligible

    eligible["distance_below_target"] = (
        target - eligible["strike_price"]
    )

    selected = (
        eligible.sort_values(
            [
                "expiration",
                "distance_below_target",
                "raw_symbol",
            ]
        )
        .groupby(
            "expiration",
            as_index=False,
            group_keys=False,
        )
        .head(1)
        .copy()
    )

    return selected


def run_download(
    client: Any,
    config: dict[str, Any],
    start: pd.Timestamp,
    end: pd.Timestamp,
    cost: float,
    maximum_cost: float,
) -> int:
    """Download definitions only after a cost guard passes."""

    if cost > maximum_cost:
        print(
            f"DOWNLOAD BLOCKED: estimated cost ${cost:,.6f} "
            f"exceeds --max-cost-usd ${maximum_cost:,.6f}."
        )
        return 1

    dataset = str(config["dataset"])
    date_text = pd.Timestamp(
        config["pilot_entry_date"]
    ).strftime("%Y-%m-%d")

    RAW_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    raw_path = (
        RAW_DIR
        / f"{dataset.replace('.', '_')}_{date_text}_definitions.dbn.zst"
    )

    print(
        f"Downloading ALL_SYMBOLS definition snapshot; "
        f"estimated cost ${cost:,.6f}."
    )

    data = client.timeseries.get_range(
        dataset=dataset,
        schema="definition",
        symbols="ALL_SYMBOLS",
        start=start.isoformat(),
        end=end.isoformat(),
    )

    data.to_file(raw_path)

    definitions = data.to_df()

    candidates = candidate_filter(
        definitions,
        config,
    )

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    candidates.to_csv(
        DISCOVERY_OUTPUT,
        index=False,
    )

    standard_target = float(
        config["standard_target_short_strike"]
    )

    defensive_target = float(
        config["defensive_target_short_strike"]
    )

    standard_choices = choose_nearest_below(
        candidates,
        standard_target,
    )

    defensive_choices = choose_nearest_below(
        candidates,
        defensive_target,
    )

    expirations = (
        candidates["expiration"]
        .dropna()
        .sort_values()
        .drop_duplicates()
    )

    summary = {
        "dataset": dataset,
        "pilot_entry_date": date_text,
        "request_start_utc": start.isoformat(),
        "request_end_utc": end.isoformat(),
        "estimated_cost_usd": cost,
        "raw_definition_rows": int(
            len(definitions)
        ),
        "mnq_put_candidate_rows": int(
            len(candidates)
        ),
        "candidate_expirations": [
            value.isoformat()
            for value in expirations
        ],
        "underlyings": sorted(
            candidates["underlying"]
            .dropna()
            .astype(str)
            .unique()
            .tolist()
        ),
        "assets": sorted(
            candidates["asset"]
            .dropna()
            .astype(str)
            .unique()
            .tolist()
        ),
        "standard_target_short_strike": standard_target,
        "defensive_target_short_strike": defensive_target,
        "standard_choices": (
            standard_choices.to_dict(
                orient="records"
            )
        ),
        "defensive_choices": (
            defensive_choices.to_dict(
                orient="records"
            )
        ),
        "historical_definition_data_downloaded": True,
        "raw_output": str(raw_path),
        "filtered_output": str(DISCOVERY_OUTPUT),
    }

    SUMMARY_OUTPUT.write_text(
        json.dumps(
            summary,
            indent=2,
            default=str,
        )
        + "\n",
        encoding="utf-8",
    )

    print("\nDefinition discovery result")
    print("-" * 104)

    print(
        f"Raw definition rows: {len(definitions):,}"
    )

    print(
        f"Likely MNQ put candidates in target band: "
        f"{len(candidates):,}"
    )

    if candidates.empty:
        print(
            "No candidates matched. The filtered CSV and summary "
            "were still written for diagnosis."
        )
    else:
        print(
            "Underlyings: "
            + ", ".join(
                sorted(
                    candidates["underlying"]
                    .dropna()
                    .astype(str)
                    .unique()
                )
            )
        )

        print(
            "Assets: "
            + ", ".join(
                sorted(
                    candidates["asset"]
                    .dropna()
                    .astype(str)
                    .unique()
                )
            )
        )

        print("\nNearest listed strikes at or below targets")
        print("-" * 104)

        for label, target, selected in (
            (
                "Standard",
                standard_target,
                standard_choices,
            ),
            (
                "Defensive",
                defensive_target,
                defensive_choices,
            ),
        ):
            print(f"\n{label} target: {target:,.2f}")

            if selected.empty:
                print("  No eligible candidate.")
                continue

            for row in selected.itertuples(index=False):
                print(
                    f"  Exp={row.expiration} | "
                    f"DTE={int(row.dte_calendar):>2} | "
                    f"Underlying={row.underlying:<8} | "
                    f"Asset={row.asset:<8} | "
                    f"Strike={row.strike_price:>9,.2f} | "
                    f"Raw={row.raw_symbol}"
                )

    print(f"\nIgnored raw DBN output: {raw_path}")
    print(f"Filtered candidates: {DISCOVERY_OUTPUT}")
    print(f"Discovery summary: {SUMMARY_OUTPUT}")

    return 0


def build_parser() -> argparse.ArgumentParser:
    """Build command-line parsing."""

    parser = argparse.ArgumentParser(
        description=(
            "Estimate or download a one-day Databento "
            "ALL_SYMBOLS definition snapshot."
        )
    )

    actions = parser.add_mutually_exclusive_group(
        required=True
    )

    actions.add_argument(
        "--estimate",
        action="store_true",
        help="Estimate cost without downloading data.",
    )

    actions.add_argument(
        "--download",
        action="store_true",
        help="Download after applying the cost guard.",
    )

    parser.add_argument(
        "--max-cost-usd",
        type=float,
        default=0.0,
        help=(
            "Maximum accepted cost for --download. "
            "A value must be explicitly supplied."
        ),
    )

    return parser


def main() -> int:
    """Run the selected action."""

    parser = build_parser()
    arguments = parser.parse_args()

    try:
        config = load_config()
        api_key = load_api_key()
        db = import_databento()
        client = db.Historical(api_key)

        start, end = request_window(config)

        print(
            "MNQ PCS Regime Research — Databento "
            "Definition Discovery"
        )
        print("=" * 64)

        print(
            f"Dataset: {config['dataset']} | "
            f"Schema: definition | Symbols: ALL_SYMBOLS"
        )

        print(
            f"Request window: {start.isoformat()} to "
            f"{end.isoformat()}"
        )

        cost = get_cost(
            client,
            config,
            start,
            end,
        )

        save_estimate(
            config,
            start,
            end,
            cost,
        )

        print(
            f"Estimated request cost: ${cost:,.6f}"
        )

        if arguments.estimate:
            print(
                "No historical definition data was downloaded."
            )
            print(
                f"Estimate output: {ESTIMATE_OUTPUT}"
            )
            return 0

        if arguments.max_cost_usd <= 0:
            print(
                "ERROR: --download requires an explicit positive "
                "--max-cost-usd guard."
            )
            return 1

        return run_download(
            client=client,
            config=config,
            start=start,
            end=end,
            cost=cost,
            maximum_cost=arguments.max_cost_usd,
        )

    except Exception as exc:
        print(f"\nERROR: {exc}")
        return 1


if __name__ == "__main__":
    sys.exit(main())

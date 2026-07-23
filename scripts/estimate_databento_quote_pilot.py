"""Estimate Databento quote costs using only jointly executable spreads.

For each eligible expiration, the script tests whether both the standard and
3%-defensive candidates can form an exact 150-point-wide listed spread.

An expiration is included in the quote pilot only when BOTH candidates are
executable. Infeasible expirations are reported rather than causing the entire
pilot to fail.

No historical quote data is downloaded.

Run:
    python scripts/estimate_databento_quote_pilot.py
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd
import yaml
from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = (
    PROJECT_ROOT / "config" / "databento_quote_pilot.yaml"
)
DEFINITION_CANDIDATES_PATH = (
    PROJECT_ROOT
    / "results"
    / "databento_definition_discovery"
    / "mnq_option_definition_candidates.csv"
)
RESULTS_DIR = PROJECT_ROOT / "results" / "databento_quote_pilot"

MANIFEST_OUTPUT = RESULTS_DIR / "quote_request_manifest.csv"
SELECTION_OUTPUT = RESULTS_DIR / "spread_selection_audit.csv"
ESTIMATE_OUTPUT = RESULTS_DIR / "quote_request_cost_estimates.json"
SUMMARY_OUTPUT = RESULTS_DIR / "quote_request_summary.txt"


def load_config() -> dict[str, Any]:
    """Load configuration."""

    if not CONFIG_PATH.is_file():
        raise FileNotFoundError(
            f"Missing configuration file: {CONFIG_PATH}"
        )

    with CONFIG_PATH.open("r", encoding="utf-8") as file_handle:
        return yaml.safe_load(file_handle)


def load_api_key() -> str:
    """Load the Databento key without displaying it."""

    load_dotenv(PROJECT_ROOT / ".env", override=True)

    api_key = os.getenv("DATABENTO_API_KEY", "").strip()

    if not api_key:
        raise RuntimeError(
            "DATABENTO_API_KEY is missing from .env."
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


def load_definitions() -> pd.DataFrame:
    """Load filtered option definitions."""

    if not DEFINITION_CANDIDATES_PATH.is_file():
        raise FileNotFoundError(
            "Missing definition candidates: "
            f"{DEFINITION_CANDIDATES_PATH}"
        )

    frame = pd.read_csv(
        DEFINITION_CANDIDATES_PATH
    )

    required = {
        "raw_symbol",
        "instrument_id",
        "underlying",
        "asset",
        "strike_price",
        "expiration",
        "dte_calendar",
    }

    missing = sorted(
        required.difference(frame.columns)
    )

    if missing:
        raise ValueError(
            "Definition candidates are missing: "
            + ", ".join(missing)
        )

    frame["strike_price"] = pd.to_numeric(
        frame["strike_price"],
        errors="coerce",
    )

    frame["dte_calendar"] = pd.to_numeric(
        frame["dte_calendar"],
        errors="coerce",
    )

    frame["expiration"] = pd.to_datetime(
        frame["expiration"],
        errors="coerce",
        utc=True,
    )

    frame["instrument_id"] = pd.to_numeric(
        frame["instrument_id"],
        errors="coerce",
    )

    for column in (
        "raw_symbol",
        "underlying",
        "asset",
    ):
        frame[column] = (
            frame[column]
            .fillna("")
            .astype(str)
            .str.strip()
        )

    invalid = frame[
        [
            "strike_price",
            "dte_calendar",
            "expiration",
            "instrument_id",
        ]
    ].isna().any(axis=1)

    if invalid.any():
        raise ValueError(
            "Definition candidates contain invalid values on "
            f"{int(invalid.sum())} rows."
        )

    return frame


def exact_strike_row(
    frame: pd.DataFrame,
    strike: float,
    tolerance: float,
) -> pd.Series | None:
    """Return one exact listed strike row, when present."""

    matches = frame.loc[
        (
            frame["strike_price"] - strike
        ).abs() <= tolerance
    ].copy()

    if matches.empty:
        return None

    return matches.sort_values(
        [
            "strike_price",
            "raw_symbol",
        ]
    ).iloc[0]


def evaluate_candidate(
    frame: pd.DataFrame,
    target_short: float,
    spread_width: float,
    tolerance: float,
) -> dict[str, Any]:
    """Evaluate one candidate on one expiration."""

    available_strikes = (
        frame["strike_price"]
        .drop_duplicates()
        .sort_values()
        .tolist()
    )

    eligible_shorts = frame.loc[
        frame["strike_price"]
        <= target_short + tolerance
    ].copy()

    if eligible_shorts.empty:
        return {
            "feasible": False,
            "reason": "No listed short strike at or below target",
            "nearest_unconstrained_short": None,
            "selected_short": None,
            "selected_long": None,
            "short_row": None,
            "long_row": None,
            "available_strikes": available_strikes,
        }

    eligible_shorts["distance_below_target"] = (
        target_short
        - eligible_shorts["strike_price"]
    )

    eligible_shorts = eligible_shorts.sort_values(
        [
            "distance_below_target",
            "raw_symbol",
        ]
    )

    nearest_unconstrained = float(
        eligible_shorts.iloc[0]["strike_price"]
    )

    for _, short_row in eligible_shorts.iterrows():
        short_strike = float(
            short_row["strike_price"]
        )

        required_long = (
            short_strike - spread_width
        )

        long_row = exact_strike_row(
            frame,
            required_long,
            tolerance,
        )

        if long_row is not None:
            return {
                "feasible": True,
                "reason": "Executable",
                "nearest_unconstrained_short": (
                    nearest_unconstrained
                ),
                "selected_short": short_strike,
                "selected_long": float(
                    long_row["strike_price"]
                ),
                "short_row": short_row,
                "long_row": long_row,
                "available_strikes": available_strikes,
            }

    return {
        "feasible": False,
        "reason": (
            f"No exact {spread_width:.0f}-point-wide pair "
            "at or below target"
        ),
        "nearest_unconstrained_short": (
            nearest_unconstrained
        ),
        "selected_short": None,
        "selected_long": None,
        "short_row": None,
        "long_row": None,
        "available_strikes": available_strikes,
    }


def build_selection_and_manifest(
    definitions: pd.DataFrame,
    config: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Audit all candidates and retain jointly feasible expirations."""

    dte_min = int(config["dte_calendar_min"])
    dte_max = int(config["dte_calendar_max"])

    eligible = definitions.loc[
        definitions["dte_calendar"].between(
            dte_min,
            dte_max,
            inclusive="both",
        )
    ].copy()

    if eligible.empty:
        raise ValueError(
            "No definitions remain after applying DTE limits."
        )

    spread_width = float(
        config["spread_width_points"]
    )

    tolerance = float(
        config["strike_match_tolerance_points"]
    )

    candidates = (
        (
            "standard",
            float(
                config["standard_target_short_strike"]
            ),
        ),
        (
            "defensive",
            float(
                config["defensive_target_short_strike"]
            ),
        ),
    )

    audit_rows: list[dict[str, Any]] = []
    selection_objects: dict[
        tuple[pd.Timestamp, str, str, int, str],
        dict[str, Any],
    ] = {}

    group_columns = [
        "expiration",
        "underlying",
        "asset",
        "dte_calendar",
    ]

    for group_key, group in eligible.groupby(
        group_columns,
        dropna=False,
        sort=True,
    ):
        (
            expiration,
            underlying,
            asset,
            dte_calendar,
        ) = group_key

        for candidate_type, target in candidates:
            selection = evaluate_candidate(
                group,
                target,
                spread_width,
                tolerance,
            )

            selection_key = (
                expiration,
                str(underlying),
                str(asset),
                int(dte_calendar),
                candidate_type,
            )

            selection_objects[
                selection_key
            ] = selection

            audit_rows.append(
                {
                    "expiration": expiration,
                    "dte_calendar": int(dte_calendar),
                    "underlying": str(underlying),
                    "asset": str(asset),
                    "candidate_type": candidate_type,
                    "target_short_strike": target,
                    "feasible": bool(
                        selection["feasible"]
                    ),
                    "reason": selection["reason"],
                    "nearest_unconstrained_short_strike": (
                        selection[
                            "nearest_unconstrained_short"
                        ]
                    ),
                    "selected_short_strike": (
                        selection["selected_short"]
                    ),
                    "selected_long_strike": (
                        selection["selected_long"]
                    ),
                    "short_distance_below_target_points": (
                        (
                            target
                            - selection["selected_short"]
                        )
                        if selection["selected_short"]
                        is not None
                        else None
                    ),
                    "available_strikes": ", ".join(
                        f"{value:g}"
                        for value in selection[
                            "available_strikes"
                        ]
                    ),
                }
            )

    audit = pd.DataFrame(audit_rows)

    manifest_rows: list[dict[str, Any]] = []

    base_group_columns = [
        "expiration",
        "underlying",
        "asset",
        "dte_calendar",
    ]

    for base_key, group in audit.groupby(
        base_group_columns,
        dropna=False,
        sort=True,
    ):
        feasibility = dict(
            zip(
                group["candidate_type"],
                group["feasible"],
            )
        )

        jointly_feasible = (
            bool(feasibility.get("standard", False))
            and bool(
                feasibility.get("defensive", False)
            )
        )

        audit.loc[
            group.index,
            "expiration_included_in_quote_pilot",
        ] = jointly_feasible

        if not jointly_feasible:
            continue

        (
            expiration,
            underlying,
            asset,
            dte_calendar,
        ) = base_key

        for candidate_type, target in candidates:
            selection_key = (
                expiration,
                str(underlying),
                str(asset),
                int(dte_calendar),
                candidate_type,
            )

            selection = selection_objects[
                selection_key
            ]

            short_row = selection["short_row"]
            long_row = selection["long_row"]

            assert short_row is not None
            assert long_row is not None

            for leg_name, side, row in (
                (
                    "short_put",
                    "sell",
                    short_row,
                ),
                (
                    "long_put",
                    "buy",
                    long_row,
                ),
            ):
                manifest_rows.append(
                    {
                        "pilot_entry_date": str(
                            config["pilot_entry_date"]
                        ),
                        "expiration": expiration,
                        "dte_calendar": int(dte_calendar),
                        "underlying": str(underlying),
                        "asset": str(asset),
                        "candidate_type": candidate_type,
                        "target_short_strike": target,
                        "nearest_unconstrained_short_strike": (
                            selection[
                                "nearest_unconstrained_short"
                            ]
                        ),
                        "selected_short_strike": (
                            selection["selected_short"]
                        ),
                        "selected_long_strike": (
                            selection["selected_long"]
                        ),
                        "short_distance_below_target_points": (
                            target
                            - float(
                                selection["selected_short"]
                            )
                        ),
                        "spread_width_points": (
                            float(
                                selection["selected_short"]
                            )
                            - float(
                                selection["selected_long"]
                            )
                        ),
                        "leg_name": leg_name,
                        "side": side,
                        "strike_price": float(
                            row["strike_price"]
                        ),
                        "raw_symbol": str(
                            row["raw_symbol"]
                        ),
                        "instrument_id": int(
                            row["instrument_id"]
                        ),
                    }
                )

    manifest = pd.DataFrame(manifest_rows)

    if manifest.empty:
        raise ValueError(
            "No expiration supports both the standard and defensive "
            "150-point-wide spreads."
        )

    audit[
        "expiration_included_in_quote_pilot"
    ] = audit[
        "expiration_included_in_quote_pilot"
    ].fillna(False).astype(bool)

    manifest = manifest.sort_values(
        [
            "expiration",
            "candidate_type",
            "leg_name",
        ]
    ).reset_index(drop=True)

    audit = audit.sort_values(
        [
            "expiration",
            "candidate_type",
        ]
    ).reset_index(drop=True)

    return audit, manifest


def quote_window(
    config: dict[str, Any],
) -> tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp]:
    """Build the target local time and UTC request window."""

    date = pd.Timestamp(
        config["pilot_entry_date"]
    )

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
            config["quote_window_minutes"]
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


def estimate_request(
    client: Any,
    dataset: str,
    schema: str,
    symbols: list[str],
    start_utc: pd.Timestamp,
    end_utc: pd.Timestamp,
) -> dict[str, Any]:
    """Estimate request cost and records."""

    result: dict[str, Any] = {
        "schema": schema,
        "symbols": symbols,
        "symbol_count": len(symbols),
        "estimated_cost_usd": None,
        "estimated_record_count": None,
        "cost_error": None,
        "record_count_error": None,
    }

    try:
        result["estimated_cost_usd"] = float(
            client.metadata.get_cost(
                dataset=dataset,
                symbols=symbols,
                stype_in="raw_symbol",
                schema=schema,
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
                symbols=symbols,
                stype_in="raw_symbol",
                schema=schema,
                start=start_utc.isoformat(),
                end=end_utc.isoformat(),
            )
        )
    except Exception as exc:
        result["record_count_error"] = str(exc)

    return result


def main() -> int:
    """Run the jointly executable quote estimate."""

    print(
        "MNQ PCS Regime Research — Databento Quote Pilot Estimate"
    )
    print("=" * 62)

    try:
        config = load_config()
        db = import_databento()
        client = db.Historical(
            load_api_key()
        )

        definitions = load_definitions()

        audit, manifest = (
            build_selection_and_manifest(
                definitions,
                config,
            )
        )

        (
            target_local,
            start_utc,
            end_utc,
        ) = quote_window(config)

        print(
            f"Entry target: {target_local.isoformat()}"
        )

        print(
            f"Quote window: {start_utc.isoformat()} to "
            f"{end_utc.isoformat()}"
        )

        print("\nExpiration and spread feasibility")
        print("-" * 174)

        for row in audit.itertuples(index=False):
            selected_text = (
                (
                    f"{row.selected_short_strike:,.0f}/"
                    f"{row.selected_long_strike:,.0f}"
                )
                if row.feasible
                else "not executable"
            )

            pilot_text = (
                "included"
                if row.expiration_included_in_quote_pilot
                else "excluded"
            )

            print(
                f"Exp={row.expiration} | "
                f"DTE={row.dte_calendar:>2} | "
                f"{row.candidate_type:<9} | "
                f"Target={row.target_short_strike:>9,.2f} | "
                f"Spread={selected_text:<20} | "
                f"Status={row.reason:<47} | "
                f"Pilot={pilot_text}"
            )

        included_expirations = (
            manifest["expiration"]
            .drop_duplicates()
            .sort_values()
            .tolist()
        )

        option_symbols = sorted(
            manifest["raw_symbol"]
            .drop_duplicates()
            .tolist()
        )

        underlying_symbols = sorted(
            manifest["underlying"]
            .drop_duplicates()
            .tolist()
        )

        all_symbols = sorted(
            set(
                option_symbols
                + underlying_symbols
            )
        )

        print(
            f"\nIncluded expirations: "
            f"{len(included_expirations)} | "
            f"Option symbols: {len(option_symbols)} | "
            f"Underlying symbols: {len(underlying_symbols)}"
        )

        print("\nSelected raw symbols")
        print("-" * 142)

        for row in manifest.itertuples(index=False):
            print(
                f"Exp={row.expiration} | "
                f"{row.candidate_type:<9} | "
                f"{row.leg_name:<9} | "
                f"Strike={row.strike_price:>9,.2f} | "
                f"Raw={row.raw_symbol}"
            )

        estimates: list[dict[str, Any]] = []

        print("\nSelected-symbol quote estimates")
        print("-" * 112)

        for schema_value in config[
            "schemas_to_estimate"
        ]:
            estimate = estimate_request(
                client=client,
                dataset=str(
                    config["dataset"]
                ),
                schema=str(schema_value),
                symbols=all_symbols,
                start_utc=start_utc,
                end_utc=end_utc,
            )

            estimates.append(estimate)

            cost = estimate[
                "estimated_cost_usd"
            ]

            records = estimate[
                "estimated_record_count"
            ]

            print(
                f"{estimate['schema']:<10} | "
                f"Cost="
                f"{f'${cost:,.6f}' if cost is not None else 'unavailable':<14} | "
                f"Records="
                f"{f'{records:,}' if records is not None else 'unavailable'}"
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

        RESULTS_DIR.mkdir(
            parents=True,
            exist_ok=True,
        )

        audit.to_csv(
            SELECTION_OUTPUT,
            index=False,
        )

        manifest.to_csv(
            MANIFEST_OUTPUT,
            index=False,
        )

        payload = {
            "dataset": str(
                config["dataset"]
            ),
            "pilot_entry_date": str(
                config["pilot_entry_date"]
            ),
            "target_entry_timestamp_local": (
                target_local.isoformat()
            ),
            "quote_window_start_utc": (
                start_utc.isoformat()
            ),
            "quote_window_end_utc": (
                end_utc.isoformat()
            ),
            "included_expirations": [
                str(value)
                for value in included_expirations
            ],
            "option_symbols": option_symbols,
            "underlying_symbols": (
                underlying_symbols
            ),
            "all_symbols": all_symbols,
            "estimates": estimates,
            "historical_quote_data_downloaded": False,
        }

        ESTIMATE_OUTPUT.write_text(
            json.dumps(
                payload,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )

        summary_lines = [
            "MNQ PCS REGIME RESEARCH — DATABENTO QUOTE PILOT",
            "=" * 57,
            "",
            f"Entry target: {target_local.isoformat()}",
            (
                f"Quote window: {start_utc.isoformat()} "
                f"to {end_utc.isoformat()}"
            ),
            "",
            "Feasibility",
            "-" * 57,
        ]

        for row in audit.itertuples(index=False):
            summary_lines.append(
                f"{row.expiration} | "
                f"{row.candidate_type} | "
                f"Feasible={row.feasible} | "
                f"Included="
                f"{row.expiration_included_in_quote_pilot} | "
                f"Reason={row.reason}"
            )

        summary_lines.extend(
            [
                "",
                "Cost estimates",
                "-" * 57,
            ]
        )

        for estimate in estimates:
            summary_lines.append(
                f"{estimate['schema']} | "
                f"Cost={estimate['estimated_cost_usd']} | "
                f"Records="
                f"{estimate['estimated_record_count']}"
            )

        SUMMARY_OUTPUT.write_text(
            "\n".join(summary_lines) + "\n",
            encoding="utf-8",
        )

        print("\nNo historical quote data was downloaded.")
        print(f"Selection audit: {SELECTION_OUTPUT}")
        print(f"Request manifest: {MANIFEST_OUTPUT}")
        print(f"Cost estimates: {ESTIMATE_OUTPUT}")
        print(f"Summary: {SUMMARY_OUTPUT}")

        return 0

    except Exception as exc:
        print(f"\nERROR: {exc}")
        return 1


if __name__ == "__main__":
    sys.exit(main())

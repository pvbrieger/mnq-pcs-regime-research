"""Download and analyze the Databento BBO-1s quote pilot.

The script uses the exact raw-symbol manifest produced by
estimate_databento_quote_pilot.py. It applies a hard cost ceiling, downloads
only the selected option legs plus the underlying future, stores the raw DBN
file in the Git-ignored data directory, and calculates spread credits from the
latest valid quote at or before 3:45 p.m. New York time.

Run:
    python scripts/download_databento_quote_pilot.py \
        --download \
        --max-cost-usd 0.01

A previously downloaded raw DBN file is reused automatically.
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
    PROJECT_ROOT / "config" / "databento_quote_pilot.yaml"
)
MANIFEST_PATH = (
    PROJECT_ROOT
    / "results"
    / "databento_quote_pilot"
    / "quote_request_manifest.csv"
)
RESULTS_DIR = PROJECT_ROOT / "results" / "databento_quote_pilot"
RAW_DIR = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "databento"
    / "quotes"
)

SNAPSHOT_OUTPUT = RESULTS_DIR / "pilot_quote_snapshot.csv"
SPREAD_OUTPUT = RESULTS_DIR / "pilot_spread_credits.csv"
DOWNLOAD_OUTPUT = RESULTS_DIR / "pilot_quote_download.json"
ANALYSIS_OUTPUT = RESULTS_DIR / "pilot_quote_analysis.txt"


def load_config() -> dict[str, Any]:
    """Load quote-pilot configuration."""

    if not CONFIG_PATH.is_file():
        raise FileNotFoundError(
            f"Missing configuration file: {CONFIG_PATH}"
        )

    with CONFIG_PATH.open("r", encoding="utf-8") as file_handle:
        return yaml.safe_load(file_handle)


def load_api_key() -> str:
    """Load Databento key without displaying it."""

    load_dotenv(PROJECT_ROOT / ".env", override=True)

    key = os.getenv("DATABENTO_API_KEY", "").strip()

    if not key:
        raise RuntimeError(
            "DATABENTO_API_KEY is missing from the project .env file."
        )

    if not key.startswith("db-"):
        raise RuntimeError(
            "DATABENTO_API_KEY does not have the expected db- prefix."
        )

    return key


def import_databento() -> Any:
    """Import Databento."""

    try:
        import databento as db
    except ImportError as exc:
        raise RuntimeError(
            "The databento package is not installed."
        ) from exc

    return db


def load_manifest() -> pd.DataFrame:
    """Load and validate the exact quote-request manifest."""

    if not MANIFEST_PATH.is_file():
        raise FileNotFoundError(
            f"Missing quote request manifest: {MANIFEST_PATH}"
        )

    manifest = pd.read_csv(MANIFEST_PATH)

    required = {
        "pilot_entry_date",
        "expiration",
        "dte_calendar",
        "underlying",
        "asset",
        "candidate_type",
        "target_short_strike",
        "selected_short_strike",
        "selected_long_strike",
        "leg_name",
        "side",
        "strike_price",
        "raw_symbol",
        "instrument_id",
    }

    missing = sorted(required.difference(manifest.columns))

    if missing:
        raise ValueError(
            "Quote manifest is missing required fields: "
            + ", ".join(missing)
        )

    manifest["expiration"] = pd.to_datetime(
        manifest["expiration"],
        errors="coerce",
        utc=True,
    )

    manifest["strike_price"] = pd.to_numeric(
        manifest["strike_price"],
        errors="coerce",
    )

    manifest["raw_symbol"] = (
        manifest["raw_symbol"]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    manifest["underlying"] = (
        manifest["underlying"]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    invalid = manifest[
        [
            "expiration",
            "strike_price",
        ]
    ].isna().any(axis=1)

    if invalid.any():
        raise ValueError(
            "Quote manifest contains invalid values on "
            f"{int(invalid.sum())} rows."
        )

    if manifest.empty:
        raise ValueError(
            "Quote request manifest contains no spread legs."
        )

    return manifest


def quote_window(
    config: dict[str, Any],
) -> tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp]:
    """Build local target and UTC request window."""

    entry_date = pd.Timestamp(
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
        year=entry_date.year,
        month=entry_date.month,
        day=entry_date.day,
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

    start_utc = (
        target_local - half_window
    ).tz_convert("UTC")

    end_utc = (
        target_local + half_window
    ).tz_convert("UTC")

    return target_local, start_utc, end_utc


def raw_output_path(
    config: dict[str, Any],
) -> Path:
    """Return the ignored raw DBN path."""

    date_text = pd.Timestamp(
        config["pilot_entry_date"]
    ).strftime("%Y-%m-%d")

    schema = str(
        config.get(
            "download_schema",
            "bbo-1s",
        )
    )

    dataset_text = str(
        config["dataset"]
    ).replace(".", "_")

    return (
        RAW_DIR
        / f"{dataset_text}_{date_text}_{schema}.dbn.zst"
    )


def estimate_cost(
    client: Any,
    config: dict[str, Any],
    symbols: list[str],
    start_utc: pd.Timestamp,
    end_utc: pd.Timestamp,
) -> float:
    """Estimate selected-symbol quote cost."""

    return float(
        client.metadata.get_cost(
            dataset=str(config["dataset"]),
            symbols=symbols,
            stype_in="raw_symbol",
            schema=str(
                config.get(
                    "download_schema",
                    "bbo-1s",
                )
            ),
            start=start_utc.isoformat(),
            end=end_utc.isoformat(),
        )
    )


def get_or_download_store(
    db: Any,
    client: Any,
    config: dict[str, Any],
    symbols: list[str],
    start_utc: pd.Timestamp,
    end_utc: pd.Timestamp,
    raw_path: Path,
    estimated_cost: float,
    maximum_cost: float,
    force_redownload: bool,
) -> tuple[Any, bool]:
    """Reuse a raw DBN file or download it under a cost guard."""

    if raw_path.is_file() and not force_redownload:
        print(
            f"Reusing existing raw DBN file: {raw_path}"
        )

        return (
            db.DBNStore.from_file(raw_path),
            False,
        )

    if estimated_cost > maximum_cost:
        raise RuntimeError(
            f"Download blocked: current estimate "
            f"${estimated_cost:,.6f} exceeds the "
            f"${maximum_cost:,.6f} cost ceiling."
        )

    RAW_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        f"Downloading selected BBO-1s data; estimated cost "
        f"${estimated_cost:,.6f}."
    )

    store = client.timeseries.get_range(
        dataset=str(config["dataset"]),
        symbols=symbols,
        stype_in="raw_symbol",
        schema=str(
            config.get(
                "download_schema",
                "bbo-1s",
            )
        ),
        start=start_utc.isoformat(),
        end=end_utc.isoformat(),
    )

    store.to_file(raw_path)

    return store, True


def normalized_quote_frame(
    store: Any,
) -> pd.DataFrame:
    """Convert the DBN store into a normalized quote DataFrame."""

    frame = store.to_df(
        price_type="float",
        pretty_ts=True,
    ).reset_index()

    timestamp_column = None

    for candidate in (
        "ts_recv",
        "ts_event",
        "index",
    ):
        if candidate in frame.columns:
            timestamp_column = candidate
            break

    if timestamp_column is None:
        raise ValueError(
            "Unable to identify a Databento quote timestamp column."
        )

    symbol_column = None

    for candidate in (
        "symbol",
        "raw_symbol",
    ):
        if candidate in frame.columns:
            symbol_column = candidate
            break

    if symbol_column is None:
        raise ValueError(
            "The quote data does not contain raw-symbol output. "
            "Available columns: "
            + ", ".join(frame.columns)
        )

    required_quote_columns = {
        "bid_px_00",
        "ask_px_00",
    }

    missing_quote_columns = sorted(
        required_quote_columns.difference(
            frame.columns
        )
    )

    if missing_quote_columns:
        raise ValueError(
            "BBO data is missing: "
            + ", ".join(missing_quote_columns)
        )

    frame["quote_timestamp_utc"] = pd.to_datetime(
        frame[timestamp_column],
        errors="coerce",
        utc=True,
    )

    frame["raw_symbol"] = (
        frame[symbol_column]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    frame["bid"] = pd.to_numeric(
        frame["bid_px_00"],
        errors="coerce",
    )

    frame["ask"] = pd.to_numeric(
        frame["ask_px_00"],
        errors="coerce",
    )

    for source, destination in (
        ("bid_sz_00", "bid_size"),
        ("ask_sz_00", "ask_size"),
        ("bid_ct_00", "bid_order_count"),
        ("ask_ct_00", "ask_order_count"),
    ):
        if source in frame.columns:
            frame[destination] = pd.to_numeric(
                frame[source],
                errors="coerce",
            )
        else:
            frame[destination] = np.nan

    invalid_timestamp = frame[
        "quote_timestamp_utc"
    ].isna()

    if invalid_timestamp.any():
        frame = frame.loc[
            ~invalid_timestamp
        ].copy()

    return frame[
        [
            "quote_timestamp_utc",
            "raw_symbol",
            "instrument_id",
            "bid",
            "ask",
            "bid_size",
            "ask_size",
            "bid_order_count",
            "ask_order_count",
        ]
    ].sort_values(
        [
            "raw_symbol",
            "quote_timestamp_utc",
        ]
    ).reset_index(drop=True)


def select_snapshot(
    quotes: pd.DataFrame,
    symbols: list[str],
    target_utc: pd.Timestamp,
    maximum_age_minutes: float,
) -> pd.DataFrame:
    """Select the latest valid quote at or before the target."""

    rows: list[dict[str, Any]] = []

    maximum_age = pd.Timedelta(
        minutes=maximum_age_minutes
    )

    for symbol in symbols:
        symbol_quotes = quotes.loc[
            quotes["raw_symbol"] == symbol
        ].copy()

        records_received = len(symbol_quotes)

        valid = symbol_quotes.loc[
            (
                symbol_quotes[
                    "quote_timestamp_utc"
                ]
                <= target_utc
            )
            & (
                symbol_quotes[
                    "quote_timestamp_utc"
                ]
                >= target_utc
                - maximum_age
            )
            & symbol_quotes["bid"].notna()
            & symbol_quotes["ask"].notna()
            & (symbol_quotes["bid"] >= 0)
            & (
                symbol_quotes["ask"]
                >= symbol_quotes["bid"]
            )
        ].copy()

        if valid.empty:
            rows.append(
                {
                    "raw_symbol": symbol,
                    "quote_available": False,
                    "records_received": records_received,
                    "quote_timestamp_utc": pd.NaT,
                    "quote_age_seconds": np.nan,
                    "bid": np.nan,
                    "ask": np.nan,
                    "mid": np.nan,
                    "bid_size": np.nan,
                    "ask_size": np.nan,
                    "bid_order_count": np.nan,
                    "ask_order_count": np.nan,
                }
            )

            continue

        selected = valid.sort_values(
            "quote_timestamp_utc"
        ).iloc[-1]

        quote_age_seconds = (
            target_utc
            - selected["quote_timestamp_utc"]
        ).total_seconds()

        rows.append(
            {
                "raw_symbol": symbol,
                "quote_available": True,
                "records_received": records_received,
                "quote_timestamp_utc": selected[
                    "quote_timestamp_utc"
                ],
                "quote_age_seconds": (
                    quote_age_seconds
                ),
                "bid": float(selected["bid"]),
                "ask": float(selected["ask"]),
                "mid": float(
                    (
                        selected["bid"]
                        + selected["ask"]
                    )
                    / 2.0
                ),
                "bid_size": selected[
                    "bid_size"
                ],
                "ask_size": selected[
                    "ask_size"
                ],
                "bid_order_count": selected[
                    "bid_order_count"
                ],
                "ask_order_count": selected[
                    "ask_order_count"
                ],
            }
        )

    return pd.DataFrame(rows)


def build_snapshot_output(
    snapshot: pd.DataFrame,
    manifest: pd.DataFrame,
    underlying_symbols: list[str],
) -> pd.DataFrame:
    """Attach spread-leg metadata to selected quotes."""

    option_metadata = manifest[
        [
            "expiration",
            "dte_calendar",
            "underlying",
            "asset",
            "candidate_type",
            "target_short_strike",
            "selected_short_strike",
            "selected_long_strike",
            "leg_name",
            "side",
            "strike_price",
            "raw_symbol",
        ]
    ].copy()

    option_snapshot = option_metadata.merge(
        snapshot,
        on="raw_symbol",
        how="left",
        validate="many_to_one",
    )

    option_snapshot["instrument_role"] = (
        "option_leg"
    )

    underlying_snapshot = snapshot.loc[
        snapshot["raw_symbol"].isin(
            underlying_symbols
        )
    ].copy()

    underlying_snapshot[
        "instrument_role"
    ] = "underlying_future"

    for column in option_metadata.columns:
        if column not in underlying_snapshot.columns:
            underlying_snapshot[column] = np.nan

    output_columns = [
        "instrument_role",
        "expiration",
        "dte_calendar",
        "underlying",
        "asset",
        "candidate_type",
        "target_short_strike",
        "selected_short_strike",
        "selected_long_strike",
        "leg_name",
        "side",
        "strike_price",
        "raw_symbol",
        "quote_available",
        "records_received",
        "quote_timestamp_utc",
        "quote_age_seconds",
        "bid",
        "ask",
        "mid",
        "bid_size",
        "ask_size",
        "bid_order_count",
        "ask_order_count",
    ]

    return pd.concat(
        [
            option_snapshot[output_columns],
            underlying_snapshot[output_columns],
        ],
        ignore_index=True,
    )


def build_spread_credits(
    snapshot_output: pd.DataFrame,
    config: dict[str, Any],
) -> pd.DataFrame:
    """Calculate natural, midpoint, and optimistic spread credits."""

    option_rows = snapshot_output.loc[
        snapshot_output[
            "instrument_role"
        ] == "option_leg"
    ].copy()

    rows: list[dict[str, Any]] = []

    group_columns = [
        "expiration",
        "dte_calendar",
        "underlying",
        "asset",
        "candidate_type",
        "target_short_strike",
        "selected_short_strike",
        "selected_long_strike",
    ]

    for group_key, group in option_rows.groupby(
        group_columns,
        dropna=False,
        sort=True,
    ):
        (
            expiration,
            dte_calendar,
            underlying,
            asset,
            candidate_type,
            target_short,
            selected_short,
            selected_long,
        ) = group_key

        short_rows = group.loc[
            group["leg_name"] == "short_put"
        ]

        long_rows = group.loc[
            group["leg_name"] == "long_put"
        ]

        if len(short_rows) != 1 or len(long_rows) != 1:
            raise ValueError(
                "Each spread must contain exactly one short and one long leg."
            )

        short = short_rows.iloc[0]
        long = long_rows.iloc[0]

        complete_quotes = bool(
            short["quote_available"]
            and long["quote_available"]
        )

        if complete_quotes:
            natural_credit = float(
                short["bid"] - long["ask"]
            )

            midpoint_credit = float(
                short["mid"] - long["mid"]
            )

            optimistic_credit = float(
                short["ask"] - long["bid"]
            )
        else:
            natural_credit = np.nan
            midpoint_credit = np.nan
            optimistic_credit = np.nan

        rows.append(
            {
                "expiration": expiration,
                "dte_calendar": dte_calendar,
                "underlying": underlying,
                "asset": asset,
                "candidate_type": candidate_type,
                "target_short_strike": target_short,
                "selected_short_strike": selected_short,
                "selected_long_strike": selected_long,
                "spread_width_points": (
                    selected_short - selected_long
                ),
                "complete_leg_quotes": complete_quotes,
                "short_symbol": short["raw_symbol"],
                "long_symbol": long["raw_symbol"],
                "short_quote_timestamp_utc": (
                    short["quote_timestamp_utc"]
                ),
                "long_quote_timestamp_utc": (
                    long["quote_timestamp_utc"]
                ),
                "short_quote_age_seconds": (
                    short["quote_age_seconds"]
                ),
                "long_quote_age_seconds": (
                    long["quote_age_seconds"]
                ),
                "short_bid": short["bid"],
                "short_ask": short["ask"],
                "long_bid": long["bid"],
                "long_ask": long["ask"],
                "natural_credit_points": (
                    natural_credit
                ),
                "mid_credit_points": (
                    midpoint_credit
                ),
                "optimistic_credit_points": (
                    optimistic_credit
                ),
            }
        )

    credits = pd.DataFrame(rows)

    credit_floor = float(
        config.get(
            "minimum_credit_points",
            8.25,
        )
    )

    credits["natural_credit_floor_pass"] = (
        credits["natural_credit_points"]
        >= credit_floor
    )

    credits["mid_credit_floor_pass"] = (
        credits["mid_credit_points"]
        >= credit_floor
    )

    comparisons: list[dict[str, Any]] = []

    for expiration, group in credits.groupby(
        "expiration",
        sort=True,
    ):
        standard_rows = group.loc[
            group["candidate_type"]
            == "standard"
        ]

        defensive_rows = group.loc[
            group["candidate_type"]
            == "defensive"
        ]

        if (
            len(standard_rows) != 1
            or len(defensive_rows) != 1
        ):
            continue

        standard = standard_rows.iloc[0]
        defensive = defensive_rows.iloc[0]

        comparisons.append(
            {
                "expiration": expiration,
                "standard_natural_credit": (
                    standard[
                        "natural_credit_points"
                    ]
                ),
                "defensive_natural_credit": (
                    defensive[
                        "natural_credit_points"
                    ]
                ),
                "natural_credit_haircut_points": (
                    standard[
                        "natural_credit_points"
                    ]
                    - defensive[
                        "natural_credit_points"
                    ]
                ),
                "standard_mid_credit": (
                    standard[
                        "mid_credit_points"
                    ]
                ),
                "defensive_mid_credit": (
                    defensive[
                        "mid_credit_points"
                    ]
                ),
                "mid_credit_haircut_points": (
                    standard[
                        "mid_credit_points"
                    ]
                    - defensive[
                        "mid_credit_points"
                    ]
                ),
            }
        )

    comparison_frame = pd.DataFrame(
        comparisons
    )

    if not comparison_frame.empty:
        credits = credits.merge(
            comparison_frame,
            on="expiration",
            how="left",
            validate="many_to_one",
        )

    tolerable_haircut = float(
        config.get(
            "proxy_tolerable_credit_haircut_points",
            2.91,
        )
    )

    if "natural_credit_haircut_points" in credits.columns:
        credits[
            "natural_haircut_within_proxy_tolerance"
        ] = (
            credits[
                "natural_credit_haircut_points"
            ]
            <= tolerable_haircut
        )

        credits[
            "mid_haircut_within_proxy_tolerance"
        ] = (
            credits[
                "mid_credit_haircut_points"
            ]
            <= tolerable_haircut
        )

    return credits.sort_values(
        [
            "expiration",
            "candidate_type",
        ]
    ).reset_index(drop=True)


def build_parser() -> argparse.ArgumentParser:
    """Build command-line parsing."""

    parser = argparse.ArgumentParser(
        description=(
            "Download and analyze the selected Databento "
            "BBO-1s quote pilot."
        )
    )

    parser.add_argument(
        "--download",
        action="store_true",
        help=(
            "Authorize a download when no reusable raw file exists."
        ),
    )

    parser.add_argument(
        "--max-cost-usd",
        type=float,
        default=0.0,
        help="Maximum accepted download cost.",
    )

    parser.add_argument(
        "--force-redownload",
        action="store_true",
        help=(
            "Redownload even when the raw DBN file already exists."
        ),
    )

    return parser


def main() -> int:
    """Download or reuse quotes and analyze the pilot."""

    parser = build_parser()
    args = parser.parse_args()

    print(
        "MNQ PCS Regime Research — Databento BBO-1s Pilot"
    )
    print("=" * 56)

    try:
        config = load_config()
        manifest = load_manifest()
        db = import_databento()
        client = db.Historical(
            load_api_key()
        )

        (
            target_local,
            start_utc,
            end_utc,
        ) = quote_window(config)

        target_utc = target_local.tz_convert(
            "UTC"
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

        estimated_cost = estimate_cost(
            client,
            config,
            all_symbols,
            start_utc,
            end_utc,
        )

        raw_path = raw_output_path(
            config
        )

        print(
            f"Entry target: {target_local.isoformat()}"
        )

        print(
            f"Quote window: {start_utc.isoformat()} to "
            f"{end_utc.isoformat()}"
        )

        print(
            f"Symbols: {len(all_symbols)} | "
            f"Current estimated cost: "
            f"${estimated_cost:,.6f}"
        )

        if (
            not raw_path.is_file()
            and not args.download
        ):
            print(
                "No reusable raw DBN file exists. Rerun with "
                "--download and an explicit --max-cost-usd guard."
            )
            return 1

        if (
            not raw_path.is_file()
            and args.max_cost_usd <= 0
        ):
            print(
                "ERROR: A positive --max-cost-usd is required "
                "for a new download."
            )
            return 1

        store, downloaded = get_or_download_store(
            db=db,
            client=client,
            config=config,
            symbols=all_symbols,
            start_utc=start_utc,
            end_utc=end_utc,
            raw_path=raw_path,
            estimated_cost=estimated_cost,
            maximum_cost=args.max_cost_usd,
            force_redownload=args.force_redownload,
        )

        quotes = normalized_quote_frame(
            store
        )

        snapshot = select_snapshot(
            quotes=quotes,
            symbols=all_symbols,
            target_utc=target_utc,
            maximum_age_minutes=float(
                config.get(
                    "maximum_quote_age_minutes",
                    5.0,
                )
            ),
        )

        snapshot_output = build_snapshot_output(
            snapshot=snapshot,
            manifest=manifest,
            underlying_symbols=underlying_symbols,
        )

        spread_credits = build_spread_credits(
            snapshot_output,
            config,
        )

        RESULTS_DIR.mkdir(
            parents=True,
            exist_ok=True,
        )

        snapshot_output.to_csv(
            SNAPSHOT_OUTPUT,
            index=False,
        )

        spread_credits.to_csv(
            SPREAD_OUTPUT,
            index=False,
        )

        available_symbols = int(
            snapshot[
                "quote_available"
            ].sum()
        )

        missing_symbols = snapshot.loc[
            ~snapshot["quote_available"],
            "raw_symbol",
        ].tolist()

        print("\nQuote availability")
        print("-" * 104)

        print(
            f"Available at or before target: "
            f"{available_symbols}/{len(snapshot)}"
        )

        for row in snapshot.itertuples(index=False):
            if row.quote_available:
                print(
                    f"{row.raw_symbol:<18} | "
                    f"{row.quote_timestamp_utc} | "
                    f"Age={row.quote_age_seconds:>6.1f}s | "
                    f"Bid={row.bid:>8.2f} | "
                    f"Ask={row.ask:>8.2f} | "
                    f"Size={row.bid_size:g}x"
                    f"{row.ask_size:g}"
                )
            else:
                print(
                    f"{row.raw_symbol:<18} | "
                    f"NO VALID QUOTE | "
                    f"Records received={row.records_received}"
                )

        print("\nSpread credits")
        print("-" * 162)

        for row in spread_credits.itertuples(
            index=False
        ):
            if not row.complete_leg_quotes:
                print(
                    f"Exp={row.expiration} | "
                    f"{row.candidate_type:<9} | "
                    "Incomplete leg quotes"
                )
                continue

            natural_pass = (
                "PASS"
                if row.natural_credit_floor_pass
                else "FAIL"
            )

            mid_pass = (
                "PASS"
                if row.mid_credit_floor_pass
                else "FAIL"
            )

            print(
                f"Exp={row.expiration} | "
                f"{row.candidate_type:<9} | "
                f"Spread={row.selected_short_strike:,.0f}/"
                f"{row.selected_long_strike:,.0f} | "
                f"Natural={row.natural_credit_points:>6.2f} "
                f"({natural_pass}) | "
                f"Mid={row.mid_credit_points:>6.2f} "
                f"({mid_pass}) | "
                f"Optimistic="
                f"{row.optimistic_credit_points:>6.2f}"
            )

        comparison_columns = {
            "natural_credit_haircut_points",
            "mid_credit_haircut_points",
        }

        if comparison_columns.issubset(
            spread_credits.columns
        ):
            comparisons = spread_credits[
                [
                    "expiration",
                    "natural_credit_haircut_points",
                    "mid_credit_haircut_points",
                    "natural_haircut_within_proxy_tolerance",
                    "mid_haircut_within_proxy_tolerance",
                ]
            ].drop_duplicates()

            print("\nDefensive credit haircut")
            print("-" * 112)

            for row in comparisons.itertuples(
                index=False
            ):
                print(
                    f"Exp={row.expiration} | "
                    f"Natural haircut="
                    f"{row.natural_credit_haircut_points:>6.2f} | "
                    f"Mid haircut="
                    f"{row.mid_credit_haircut_points:>6.2f} | "
                    f"Proxy tolerance="
                    f"{float(config.get('proxy_tolerable_credit_haircut_points', 2.91)):.2f}"
                )

        download_payload = {
            "dataset": str(
                config["dataset"]
            ),
            "schema": str(
                config.get(
                    "download_schema",
                    "bbo-1s",
                )
            ),
            "symbols": all_symbols,
            "target_entry_timestamp_local": (
                target_local.isoformat()
            ),
            "quote_window_start_utc": (
                start_utc.isoformat()
            ),
            "quote_window_end_utc": (
                end_utc.isoformat()
            ),
            "estimated_cost_usd": (
                estimated_cost
            ),
            "downloaded_during_this_run": (
                downloaded
            ),
            "raw_dbn_path": str(
                raw_path
            ),
            "raw_records": len(
                quotes
            ),
            "symbols_with_valid_quotes": (
                available_symbols
            ),
            "symbols_without_valid_quotes": (
                missing_symbols
            ),
        }

        DOWNLOAD_OUTPUT.write_text(
            json.dumps(
                download_payload,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )

        analysis_lines = [
            "MNQ PCS REGIME RESEARCH — DATABENTO BBO-1S PILOT",
            "=" * 60,
            "",
            f"Entry target: {target_local.isoformat()}",
            (
                f"Estimated cost: "
                f"${estimated_cost:,.6f}"
            ),
            (
                f"Raw records: {len(quotes):,}"
            ),
            (
                f"Symbols with valid quotes: "
                f"{available_symbols}/{len(snapshot)}"
            ),
            "",
            "Spread credits",
            "-" * 60,
        ]

        for row in spread_credits.itertuples(
            index=False
        ):
            analysis_lines.append(
                f"{row.expiration} | "
                f"{row.candidate_type} | "
                f"Complete={row.complete_leg_quotes} | "
                f"Natural={row.natural_credit_points} | "
                f"Mid={row.mid_credit_points} | "
                f"Optimistic={row.optimistic_credit_points}"
            )

        ANALYSIS_OUTPUT.write_text(
            "\n".join(analysis_lines) + "\n",
            encoding="utf-8",
        )

        print("\nMethodological note")
        print("-" * 104)

        print(
            "Each symbol uses its latest valid quote at or before "
            "3:45 p.m.; no future quote is used."
        )

        print(f"\nIgnored raw DBN: {raw_path}")
        print(f"Quote snapshot: {SNAPSHOT_OUTPUT}")
        print(f"Spread credits: {SPREAD_OUTPUT}")
        print(f"Download audit: {DOWNLOAD_OUTPUT}")
        print(f"Analysis summary: {ANALYSIS_OUTPUT}")

        return 0

    except Exception as exc:
        print(f"\nERROR: {exc}")
        return 1


if __name__ == "__main__":
    sys.exit(main())

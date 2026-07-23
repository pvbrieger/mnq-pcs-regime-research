"""Download and analyze closer-strike MNQ option quotes.

Run from the project root:

    python scripts/download_vix_closer_strike_quotes.py \
        --download \
        --max-cost-usd 0.20

The script:

- reads the exact-leg manifest from the closer-strike estimate;
- re-estimates only missing date-level quote files;
- skips dates with zero estimated BBO records;
- enforces a combined hard cost ceiling;
- downloads each date's unique option symbols once;
- selects the latest valid quote at or before 3:45 p.m. New York time;
- calculates natural, midpoint, and optimistic credit;
- compares midpoint credit with historical intrinsic-loss estimates; and
- measures the incremental economic effect of moving 1%, 2%, or 3% closer.

Raw DBN files remain under the Git-ignored raw-data directory.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from dotenv import load_dotenv


ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results" / "vix_closer_strike_quote_test"
RAW_DIR = ROOT / "data" / "raw" / "databento" / "quotes"

MANIFEST_PATH = RESULTS_DIR / "option_quote_manifest.csv"
SELECTION_PATH = RESULTS_DIR / "selected_distance_spreads.csv"

SNAPSHOT_CSV = RESULTS_DIR / "option_quote_snapshot.csv"
ECONOMICS_CSV = RESULTS_DIR / "distance_economics.csv"
INCREMENTAL_CSV = RESULTS_DIR / "incremental_vs_current.csv"
BUCKET_SUMMARY_CSV = RESULTS_DIR / "bucket_distance_summary.csv"
AUDIT_JSON = RESULTS_DIR / "quote_download_audit.json"
SUMMARY_TXT = RESULTS_DIR / "closer_strike_quote_analysis.txt"


MNQ_MULTIPLIER = 2.0
MAXIMUM_QUOTE_AGE_MINUTES = 5.0


def api_key() -> str:
    """Load the Databento key without displaying it."""

    load_dotenv(ROOT / ".env", override=True)

    value = os.getenv("DATABENTO_API_KEY", "").strip()

    if not value.startswith("db-"):
        raise RuntimeError(
            "A valid DATABENTO_API_KEY is required in the project .env file."
        )

    return value


def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load and validate the exact-leg manifest and spread selections."""

    for path in (MANIFEST_PATH, SELECTION_PATH):
        if not path.is_file():
            raise FileNotFoundError(f"Missing required file: {path}")

    manifest = pd.read_csv(MANIFEST_PATH)
    selections = pd.read_csv(SELECTION_PATH)

    manifest_required = {
        "entry_date",
        "bucket",
        "vix_close",
        "underlying",
        "underlying_mid",
        "expiration",
        "dte_calendar",
        "distance_label",
        "closer_offset_pct",
        "target_distance_pct",
        "target_short_strike",
        "selected_short_strike",
        "selected_long_strike",
        "actual_short_distance_pct",
        "leg_name",
        "strike_price",
        "raw_symbol",
        "target_entry_timestamp_utc",
    }

    selection_required = {
        "entry_date",
        "bucket",
        "distance_label",
        "closer_offset_pct",
        "target_distance_pct",
        "selected_short_strike",
        "selected_long_strike",
        "historical_sample_count",
        "historical_breach_rate",
        "historical_max_loss_rate",
        "historical_mean_intrinsic_loss_points",
        "historical_loss_ci_low",
        "historical_loss_ci_high",
        "selection_status",
    }

    for name, frame, required in (
        ("option manifest", manifest, manifest_required),
        ("spread selections", selections, selection_required),
    ):
        missing = sorted(required.difference(frame.columns))

        if missing:
            raise ValueError(
                f"{name} is missing fields: {', '.join(missing)}"
            )

    for frame in (manifest, selections):
        frame["entry_date"] = (
            frame["entry_date"].fillna("").astype(str).str.strip()
        )

    manifest["raw_symbol"] = (
        manifest["raw_symbol"].fillna("").astype(str).str.strip()
    )

    manifest["leg_name"] = (
        manifest["leg_name"].fillna("").astype(str).str.strip()
    )

    manifest["expiration"] = pd.to_datetime(
        manifest["expiration"],
        errors="coerce",
        utc=True,
    )

    manifest["target_entry_timestamp_utc"] = pd.to_datetime(
        manifest["target_entry_timestamp_utc"],
        errors="coerce",
        utc=True,
    )

    numeric_manifest_columns = (
        "vix_close",
        "underlying_mid",
        "dte_calendar",
        "closer_offset_pct",
        "target_distance_pct",
        "target_short_strike",
        "selected_short_strike",
        "selected_long_strike",
        "actual_short_distance_pct",
        "strike_price",
    )

    for column in numeric_manifest_columns:
        manifest[column] = pd.to_numeric(
            manifest[column],
            errors="coerce",
        )

    numeric_selection_columns = (
        "closer_offset_pct",
        "target_distance_pct",
        "selected_short_strike",
        "selected_long_strike",
        "historical_sample_count",
        "historical_breach_rate",
        "historical_max_loss_rate",
        "historical_mean_intrinsic_loss_points",
        "historical_loss_ci_low",
        "historical_loss_ci_high",
    )

    for column in numeric_selection_columns:
        selections[column] = pd.to_numeric(
            selections[column],
            errors="coerce",
        )

    manifest = manifest.loc[
        manifest["raw_symbol"].ne("")
        & manifest["target_entry_timestamp_utc"].notna()
    ].copy()

    selected = selections.loc[
        selections["selection_status"] == "Selected"
    ].copy()

    if manifest.empty or selected.empty:
        raise RuntimeError("No executable closer-strike comparisons exist.")

    return (
        manifest.sort_values(
            ["entry_date", "closer_offset_pct", "leg_name"]
        ).reset_index(drop=True),
        selected.sort_values(
            ["entry_date", "closer_offset_pct"]
        ).reset_index(drop=True),
    )


def raw_quote_path(entry_date: str) -> Path:
    """Return the ignored date-level multi-strike quote file."""

    return RAW_DIR / (
        f"GLBX_MDP3_{entry_date}_MNQ_closer_strikes_bbo-1s.dbn.zst"
    )


def estimate_request(
    client: Any,
    symbols: list[str],
    start_utc: pd.Timestamp,
    end_utc: pd.Timestamp,
) -> tuple[float, int]:
    """Estimate one exact-symbol BBO-1s request."""

    kwargs = dict(
        dataset="GLBX.MDP3",
        symbols=symbols,
        stype_in="raw_symbol",
        schema="bbo-1s",
        start=start_utc.isoformat(),
        end=end_utc.isoformat(),
    )

    return (
        float(client.metadata.get_cost(**kwargs)),
        int(client.metadata.get_record_count(**kwargs)),
    )


def download_request(
    client: Any,
    symbols: list[str],
    start_utc: pd.Timestamp,
    end_utc: pd.Timestamp,
    output_path: Path,
) -> None:
    """Download one date's unique exact option symbols."""

    output_path.parent.mkdir(parents=True, exist_ok=True)

    store = client.timeseries.get_range(
        dataset="GLBX.MDP3",
        symbols=symbols,
        stype_in="raw_symbol",
        schema="bbo-1s",
        start=start_utc.isoformat(),
        end=end_utc.isoformat(),
    )

    store.to_file(output_path)


def normalize_quotes(db: Any, file_path: Path) -> pd.DataFrame:
    """Load a BBO DBN file and normalize quote columns."""

    frame = db.DBNStore.from_file(file_path).to_df(
        price_type="float",
        pretty_ts=True,
    ).reset_index()

    timestamp_column = next(
        (
            column
            for column in ("ts_recv", "ts_event", "index")
            if column in frame.columns
        ),
        None,
    )

    symbol_column = next(
        (
            column
            for column in ("symbol", "raw_symbol")
            if column in frame.columns
        ),
        None,
    )

    if timestamp_column is None or symbol_column is None:
        raise ValueError(
            f"Unable to identify timestamp or symbol in {file_path.name}."
        )

    required = {"bid_px_00", "ask_px_00"}
    missing = sorted(required.difference(frame.columns))

    if missing:
        raise ValueError(
            f"{file_path.name} is missing fields: {', '.join(missing)}"
        )

    output = pd.DataFrame(
        {
            "quote_timestamp_utc": pd.to_datetime(
                frame[timestamp_column],
                errors="coerce",
                utc=True,
            ),
            "raw_symbol": (
                frame[symbol_column].fillna("").astype(str).str.strip()
            ),
            "bid": pd.to_numeric(
                frame["bid_px_00"],
                errors="coerce",
            ),
            "ask": pd.to_numeric(
                frame["ask_px_00"],
                errors="coerce",
            ),
            "bid_size": pd.to_numeric(
                frame.get("bid_sz_00", np.nan),
                errors="coerce",
            ),
            "ask_size": pd.to_numeric(
                frame.get("ask_sz_00", np.nan),
                errors="coerce",
            ),
        }
    )

    return output.dropna(
        subset=["quote_timestamp_utc"]
    ).sort_values(
        ["raw_symbol", "quote_timestamp_utc"]
    )


def select_snapshot(
    quotes: pd.DataFrame,
    symbols: list[str],
    target_utc: pd.Timestamp,
) -> pd.DataFrame:
    """Select the latest valid quote per symbol at or before target."""

    oldest = target_utc - pd.Timedelta(
        minutes=MAXIMUM_QUOTE_AGE_MINUTES
    )

    rows: list[dict[str, Any]] = []

    for symbol in symbols:
        symbol_rows = quotes.loc[
            quotes["raw_symbol"] == symbol
        ].copy()

        valid = symbol_rows.loc[
            (symbol_rows["quote_timestamp_utc"] <= target_utc)
            & (symbol_rows["quote_timestamp_utc"] >= oldest)
            & symbol_rows["bid"].notna()
            & symbol_rows["ask"].notna()
            & (symbol_rows["bid"] >= 0)
            & (symbol_rows["ask"] >= symbol_rows["bid"])
        ]

        if valid.empty:
            rows.append(
                {
                    "raw_symbol": symbol,
                    "quote_available": False,
                    "records_received": len(symbol_rows),
                    "quote_timestamp_utc": pd.NaT,
                    "quote_age_seconds": np.nan,
                    "bid": np.nan,
                    "ask": np.nan,
                    "mid": np.nan,
                    "bid_size": np.nan,
                    "ask_size": np.nan,
                }
            )
            continue

        selected = valid.iloc[-1]

        rows.append(
            {
                "raw_symbol": symbol,
                "quote_available": True,
                "records_received": len(symbol_rows),
                "quote_timestamp_utc": selected["quote_timestamp_utc"],
                "quote_age_seconds": (
                    target_utc - selected["quote_timestamp_utc"]
                ).total_seconds(),
                "bid": float(selected["bid"]),
                "ask": float(selected["ask"]),
                "mid": float(
                    (selected["bid"] + selected["ask"]) / 2.0
                ),
                "bid_size": selected["bid_size"],
                "ask_size": selected["ask_size"],
            }
        )

    return pd.DataFrame(rows)


def unavailable_snapshot(symbols: list[str]) -> pd.DataFrame:
    """Return unavailable rows for a zero-record date."""

    return pd.DataFrame(
        [
            {
                "raw_symbol": symbol,
                "quote_available": False,
                "records_received": 0,
                "quote_timestamp_utc": pd.NaT,
                "quote_age_seconds": np.nan,
                "bid": np.nan,
                "ask": np.nan,
                "mid": np.nan,
                "bid_size": np.nan,
                "ask_size": np.nan,
            }
            for symbol in symbols
        ]
    )


def build_incremental(economics: pd.DataFrame) -> pd.DataFrame:
    """Compare each closer distance with the same date's current spread."""

    rows: list[dict[str, Any]] = []

    for entry_date, group in economics.groupby(
        "entry_date",
        sort=True,
    ):
        current_rows = group.loc[
            group["closer_offset_pct"] == 0.0
        ]

        if len(current_rows) != 1:
            continue

        current = current_rows.iloc[0]

        for row in group.itertuples(index=False):
            credit_gain = (
                row.mid_credit_points
                - current["mid_credit_points"]
                if pd.notna(row.mid_credit_points)
                and pd.notna(current["mid_credit_points"])
                else np.nan
            )

            loss_increase = (
                row.historical_mean_intrinsic_loss_points
                - current[
                    "historical_mean_intrinsic_loss_points"
                ]
            )

            conservative_loss_increase = (
                row.historical_loss_ci_high
                - current["historical_loss_ci_high"]
            )

            rows.append(
                {
                    "entry_date": entry_date,
                    "bucket": row.bucket,
                    "vix_close": row.vix_close,
                    "distance_label": row.distance_label,
                    "closer_offset_pct": row.closer_offset_pct,
                    "target_distance_pct": row.target_distance_pct,
                    "actual_short_distance_pct": (
                        row.actual_short_distance_pct
                    ),
                    "complete_quotes": row.complete_quotes,
                    "current_mid_credit_points": current[
                        "mid_credit_points"
                    ],
                    "candidate_mid_credit_points": row.mid_credit_points,
                    "incremental_mid_credit_points": credit_gain,
                    "incremental_historical_mean_loss_points": (
                        loss_increase
                    ),
                    "incremental_historical_upper_loss_points": (
                        conservative_loss_increase
                    ),
                    "incremental_net_edge_points": (
                        credit_gain - loss_increase
                        if pd.notna(credit_gain)
                        else np.nan
                    ),
                    "incremental_conservative_edge_points": (
                        credit_gain - conservative_loss_increase
                        if pd.notna(credit_gain)
                        else np.nan
                    ),
                    "current_gross_edge_points": current[
                        "gross_edge_points"
                    ],
                    "candidate_gross_edge_points": row.gross_edge_points,
                    "current_conservative_edge_points": current[
                        "conservative_edge_points"
                    ],
                    "candidate_conservative_edge_points": (
                        row.conservative_edge_points
                    ),
                }
            )

    return pd.DataFrame(rows).sort_values(
        ["entry_date", "closer_offset_pct"]
    ).reset_index(drop=True)


def build_bucket_summary(
    economics: pd.DataFrame,
    incremental: pd.DataFrame,
) -> pd.DataFrame:
    """Summarize complete observations by VIX bucket and distance offset."""

    usable_economics = economics.loc[
        economics["complete_quotes"]
    ].copy()

    usable_incremental = incremental.loc[
        incremental["complete_quotes"]
    ].copy()

    rows: list[dict[str, Any]] = []

    group_columns = [
        "bucket",
        "distance_label",
        "closer_offset_pct",
        "target_distance_pct",
    ]

    for group_key, group in usable_economics.groupby(
        group_columns,
        sort=True,
    ):
        (
            bucket,
            distance_label,
            closer_offset,
            target_distance,
        ) = group_key

        inc = usable_incremental.loc[
            (usable_incremental["bucket"] == bucket)
            & (
                usable_incremental["closer_offset_pct"]
                == closer_offset
            )
        ]

        rows.append(
            {
                "bucket": bucket,
                "distance_label": distance_label,
                "closer_offset_pct": float(closer_offset),
                "target_distance_pct": float(target_distance),
                "complete_sample_dates": len(group),
                "median_mid_credit_points": float(
                    group["mid_credit_points"].median()
                ),
                "median_mid_credit_dollars": float(
                    group["mid_credit_dollars"].median()
                ),
                "median_historical_mean_loss_points": float(
                    group[
                        "historical_mean_intrinsic_loss_points"
                    ].median()
                ),
                "median_gross_edge_points": float(
                    group["gross_edge_points"].median()
                ),
                "minimum_gross_edge_points": float(
                    group["gross_edge_points"].min()
                ),
                "median_conservative_edge_points": float(
                    group["conservative_edge_points"].median()
                ),
                "positive_gross_edge_count": int(
                    (group["gross_edge_points"] > 0).sum()
                ),
                "positive_conservative_edge_count": int(
                    (
                        group["conservative_edge_points"] > 0
                    ).sum()
                ),
                "median_incremental_credit_points": (
                    float(
                        inc["incremental_mid_credit_points"].median()
                    )
                    if not inc.empty
                    else np.nan
                ),
                "median_incremental_mean_loss_points": (
                    float(
                        inc[
                            "incremental_historical_mean_loss_points"
                        ].median()
                    )
                    if not inc.empty
                    else np.nan
                ),
                "median_incremental_net_edge_points": (
                    float(
                        inc["incremental_net_edge_points"].median()
                    )
                    if not inc.empty
                    else np.nan
                ),
                "minimum_incremental_net_edge_points": (
                    float(
                        inc["incremental_net_edge_points"].min()
                    )
                    if not inc.empty
                    else np.nan
                ),
            }
        )

    desired_order = {
        "below_15": 0,
        "15_to_20": 1,
        "20_to_25": 2,
        "25_to_30": 3,
        "30_to_35": 4,
        "35_to_40": 5,
        "over_40": 6,
    }

    output = pd.DataFrame(rows)
    output["_bucket_order"] = output[
        "bucket"
    ].map(desired_order).fillna(999)

    return (
        output.sort_values(
            ["_bucket_order", "closer_offset_pct"]
        )
        .drop(columns="_bucket_order")
        .reset_index(drop=True)
    )


def build_parser() -> argparse.ArgumentParser:
    """Build command-line parser."""

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--download",
        action="store_true",
        help="Authorize missing historical option-quote downloads.",
    )

    parser.add_argument(
        "--max-cost-usd",
        type=float,
        default=0.0,
        help="Maximum combined estimated cost for missing files.",
    )

    parser.add_argument(
        "--force-redownload",
        action="store_true",
        help="Redownload even when a raw DBN file exists.",
    )

    return parser


def main() -> int:
    """Download exact quotes and compare closer strikes economically."""

    import databento as db

    args = build_parser().parse_args()

    print(
        "MNQ PCS Regime Research — Closer-Strike Quote Download"
    )
    print("=" * 66)

    manifest, selections = load_inputs()
    client = db.Historical(api_key())

    requests: list[dict[str, Any]] = []

    print("\nDate-level quote files and current estimates")
    print("-" * 136)

    for entry_date, group in manifest.groupby(
        "entry_date",
        sort=True,
    ):
        symbols = sorted(
            group["raw_symbol"].drop_duplicates().tolist()
        )

        targets = group[
            "target_entry_timestamp_utc"
        ].drop_duplicates()

        if len(targets) != 1:
            raise ValueError(
                f"{entry_date} has inconsistent target timestamps."
            )

        target_utc = pd.Timestamp(targets.iloc[0])
        start_utc = target_utc - pd.Timedelta(minutes=5)
        end_utc = target_utc + pd.Timedelta(minutes=5)
        output_path = raw_quote_path(entry_date)

        reuse = (
            output_path.is_file()
            and not args.force_redownload
        )

        if reuse:
            cost, records = 0.0, -1
            action = "reuse local file"
        else:
            cost, records = estimate_request(
                client,
                symbols,
                start_utc,
                end_utc,
            )
            action = (
                "no records available"
                if records == 0
                else "download required"
            )

        requests.append(
            {
                "entry_date": entry_date,
                "symbols": symbols,
                "target_utc": target_utc,
                "start_utc": start_utc,
                "end_utc": end_utc,
                "output_path": output_path,
                "reuse": reuse,
                "estimated_cost_usd": cost,
                "estimated_record_count": records,
                "action": action,
                "downloaded_during_this_run": False,
            }
        )

        records_text = "local" if records == -1 else f"{records:,}"

        print(
            f"{entry_date} | "
            f"{action:<20} | "
            f"Unique symbols={len(symbols):>2} | "
            f"Cost=${cost:,.6f} | "
            f"Records={records_text}"
        )

    downloadable = [
        request
        for request in requests
        if not request["reuse"]
        and request["estimated_record_count"] > 0
    ]

    zero_record_dates = [
        request["entry_date"]
        for request in requests
        if not request["reuse"]
        and request["estimated_record_count"] == 0
    ]

    total_cost = sum(
        request["estimated_cost_usd"]
        for request in downloadable
    )

    print(
        f"\nTotal estimated cost for downloadable files: "
        f"${total_cost:,.6f}"
    )

    if zero_record_dates:
        print(
            "Dates with zero estimated BBO records: "
            + ", ".join(zero_record_dates)
        )

    if downloadable and not args.download:
        raise RuntimeError(
            "Missing files require --download and --max-cost-usd."
        )

    if downloadable and args.max_cost_usd <= 0:
        raise RuntimeError(
            "A positive --max-cost-usd is required."
        )

    if total_cost > args.max_cost_usd:
        raise RuntimeError(
            f"Blocked: ${total_cost:,.6f} exceeds "
            f"${args.max_cost_usd:,.6f}."
        )

    for request in downloadable:
        print(
            f"\nDownloading {request['entry_date']}; "
            f"estimated cost "
            f"${request['estimated_cost_usd']:,.6f}."
        )

        download_request(
            client,
            request["symbols"],
            request["start_utc"],
            request["end_utc"],
            request["output_path"],
        )

        request["downloaded_during_this_run"] = True

    snapshot_frames: list[pd.DataFrame] = []
    economics_rows: list[dict[str, Any]] = []

    print("\nQuote availability")
    print("-" * 154)

    request_lookup = {
        request["entry_date"]: request
        for request in requests
    }

    for entry_date, date_manifest in manifest.groupby(
        "entry_date",
        sort=True,
    ):
        request = request_lookup[entry_date]

        if request["output_path"].is_file():
            quotes = normalize_quotes(
                db,
                request["output_path"],
            )
            snapshot = select_snapshot(
                quotes,
                request["symbols"],
                request["target_utc"],
            )
        else:
            snapshot = unavailable_snapshot(
                request["symbols"]
            )

        snapshot_output = date_manifest.merge(
            snapshot,
            on="raw_symbol",
            how="left",
            validate="many_to_one",
        )

        snapshot_frames.append(snapshot_output)

        available = int(
            snapshot["quote_available"].sum()
        )

        print(
            f"{entry_date} | "
            f"Valid symbols={available}/{len(snapshot)}"
        )

        for row in snapshot.itertuples(index=False):
            if bool(row.quote_available):
                print(
                    f"  {row.raw_symbol:<18} | "
                    f"{row.quote_timestamp_utc} | "
                    f"Age={row.quote_age_seconds:>6.1f}s | "
                    f"Bid={row.bid:>8.2f} | "
                    f"Ask={row.ask:>8.2f}"
                )
            else:
                print(
                    f"  {row.raw_symbol:<18} | "
                    f"NO VALID QUOTE | "
                    f"Records={int(row.records_received)}"
                )

        date_selections = selections.loc[
            selections["entry_date"] == entry_date
        ].copy()

        for selection in date_selections.itertuples(index=False):
            legs = snapshot_output.loc[
                snapshot_output["distance_label"]
                == selection.distance_label
            ]

            short = legs.loc[
                legs["leg_name"] == "short_put"
            ].iloc[0]

            long = legs.loc[
                legs["leg_name"] == "long_put"
            ].iloc[0]

            complete = bool(
                short["quote_available"]
                and long["quote_available"]
            )

            if complete:
                natural_credit = float(
                    short["bid"] - long["ask"]
                )
                mid_credit = float(
                    short["mid"] - long["mid"]
                )
                optimistic_credit = float(
                    short["ask"] - long["bid"]
                )
                quote_width = float(
                    optimistic_credit - natural_credit
                )
                timestamp_gap = abs(
                    (
                        pd.Timestamp(
                            short["quote_timestamp_utc"]
                        )
                        - pd.Timestamp(
                            long["quote_timestamp_utc"]
                        )
                    ).total_seconds()
                )
            else:
                natural_credit = np.nan
                mid_credit = np.nan
                optimistic_credit = np.nan
                quote_width = np.nan
                timestamp_gap = np.nan

            mean_loss = float(
                selection.historical_mean_intrinsic_loss_points
            )
            upper_loss = float(
                selection.historical_loss_ci_high
            )

            economics_rows.append(
                {
                    "entry_date": entry_date,
                    "bucket": selection.bucket,
                    "vix_close": float(
                        date_manifest["vix_close"].iloc[0]
                    ),
                    "expiration": (
                        date_manifest["expiration"].iloc[0]
                    ),
                    "dte_calendar": int(
                        date_manifest["dte_calendar"].iloc[0]
                    ),
                    "distance_label": selection.distance_label,
                    "closer_offset_pct": float(
                        selection.closer_offset_pct
                    ),
                    "target_distance_pct": float(
                        selection.target_distance_pct
                    ),
                    "actual_short_distance_pct": float(
                        date_manifest.loc[
                            date_manifest["distance_label"]
                            == selection.distance_label,
                            "actual_short_distance_pct",
                        ].iloc[0]
                    ),
                    "selected_short_strike": float(
                        selection.selected_short_strike
                    ),
                    "selected_long_strike": float(
                        selection.selected_long_strike
                    ),
                    "short_symbol": str(short["raw_symbol"]),
                    "long_symbol": str(long["raw_symbol"]),
                    "complete_quotes": complete,
                    "short_quote_age_seconds": (
                        short["quote_age_seconds"]
                    ),
                    "long_quote_age_seconds": (
                        long["quote_age_seconds"]
                    ),
                    "leg_timestamp_gap_seconds": timestamp_gap,
                    "short_bid": short["bid"],
                    "short_ask": short["ask"],
                    "long_bid": long["bid"],
                    "long_ask": long["ask"],
                    "natural_credit_points": natural_credit,
                    "mid_credit_points": mid_credit,
                    "optimistic_credit_points": optimistic_credit,
                    "mid_credit_dollars": (
                        mid_credit * MNQ_MULTIPLIER
                        if pd.notna(mid_credit)
                        else np.nan
                    ),
                    "spread_quote_width_points": quote_width,
                    "historical_sample_count": int(
                        selection.historical_sample_count
                    ),
                    "historical_breach_rate": float(
                        selection.historical_breach_rate
                    ),
                    "historical_max_loss_rate": float(
                        selection.historical_max_loss_rate
                    ),
                    "historical_mean_intrinsic_loss_points": (
                        mean_loss
                    ),
                    "historical_loss_ci_low": float(
                        selection.historical_loss_ci_low
                    ),
                    "historical_loss_ci_high": upper_loss,
                    "gross_edge_points": (
                        mid_credit - mean_loss
                        if pd.notna(mid_credit)
                        else np.nan
                    ),
                    "gross_edge_dollars": (
                        (mid_credit - mean_loss)
                        * MNQ_MULTIPLIER
                        if pd.notna(mid_credit)
                        else np.nan
                    ),
                    "conservative_edge_points": (
                        mid_credit - upper_loss
                        if pd.notna(mid_credit)
                        else np.nan
                    ),
                    "data_status": (
                        "complete"
                        if complete
                        else (
                            "metadata_zero_records"
                            if entry_date in zero_record_dates
                            else "incomplete_quotes"
                        )
                    ),
                }
            )

    snapshot_all = pd.concat(
        snapshot_frames,
        ignore_index=True,
    )

    economics = pd.DataFrame(
        economics_rows
    ).sort_values(
        ["entry_date", "closer_offset_pct"]
    ).reset_index(drop=True)

    incremental = build_incremental(
        economics
    )

    bucket_summary = build_bucket_summary(
        economics,
        incremental,
    )

    print("\nDistance economics")
    print("-" * 206)

    for row in economics.itertuples(index=False):
        if row.complete_quotes:
            print(
                f"{row.entry_date} | "
                f"Bucket={row.bucket:<10} | "
                f"{row.distance_label:<12} | "
                f"Target={row.target_distance_pct:>4.1f}% | "
                f"Actual={row.actual_short_distance_pct:>5.2f}% | "
                f"Mid={row.mid_credit_points:>7.2f} | "
                f"Hist loss={row.historical_mean_intrinsic_loss_points:>6.2f} | "
                f"Gross edge={row.gross_edge_points:>7.2f} | "
                f"Conservative={row.conservative_edge_points:>7.2f}"
            )
        else:
            print(
                f"{row.entry_date} | "
                f"Bucket={row.bucket:<10} | "
                f"{row.distance_label:<12} | "
                f"NO COMPLETE QUOTE | "
                f"Status={row.data_status}"
            )

    print("\nIncremental versus current distance")
    print("-" * 194)

    for row in incremental.itertuples(index=False):
        if row.complete_quotes and pd.notna(
            row.incremental_mid_credit_points
        ):
            print(
                f"{row.entry_date} | "
                f"Bucket={row.bucket:<10} | "
                f"{row.distance_label:<12} | "
                f"Credit gain={row.incremental_mid_credit_points:>7.2f} | "
                f"Mean-loss increase="
                f"{row.incremental_historical_mean_loss_points:>7.2f} | "
                f"Net incremental="
                f"{row.incremental_net_edge_points:>7.2f} | "
                f"Conservative incremental="
                f"{row.incremental_conservative_edge_points:>7.2f}"
            )
        else:
            print(
                f"{row.entry_date} | "
                f"Bucket={row.bucket:<10} | "
                f"{row.distance_label:<12} | "
                "NO COMPARABLE COMPLETE QUOTE"
            )

    print("\nBucket-distance summary")
    print("-" * 188)

    if bucket_summary.empty:
        print("No complete comparisons were available.")
    else:
        for row in bucket_summary.itertuples(index=False):
            print(
                f"{row.bucket:<10} | "
                f"{row.distance_label:<12} | "
                f"N={row.complete_sample_dates:>2} | "
                f"Median mid={row.median_mid_credit_points:>6.2f} | "
                f"Median gross edge={row.median_gross_edge_points:>7.2f} | "
                f"Median incremental="
                f"{row.median_incremental_net_edge_points:>7.2f} | "
                f"Min incremental="
                f"{row.minimum_incremental_net_edge_points:>7.2f}"
            )

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    snapshot_all.to_csv(
        SNAPSHOT_CSV,
        index=False,
    )

    economics.to_csv(
        ECONOMICS_CSV,
        index=False,
    )

    incremental.to_csv(
        INCREMENTAL_CSV,
        index=False,
    )

    bucket_summary.to_csv(
        BUCKET_SUMMARY_CSV,
        index=False,
    )

    audit_payload = {
        "dataset": "GLBX.MDP3",
        "schema": "bbo-1s",
        "mnq_multiplier": MNQ_MULTIPLIER,
        "estimated_downloadable_cost_usd": total_cost,
        "zero_record_dates": zero_record_dates,
        "complete_distance_quotes": int(
            economics["complete_quotes"].sum()
        ),
        "files": [
            {
                "entry_date": request["entry_date"],
                "raw_path": str(request["output_path"]),
                "file_reused": request["reuse"],
                "action": request["action"],
                "estimated_cost_usd": request[
                    "estimated_cost_usd"
                ],
                "estimated_record_count": request[
                    "estimated_record_count"
                ],
                "downloaded_during_this_run": request[
                    "downloaded_during_this_run"
                ],
            }
            for request in requests
        ],
    }

    AUDIT_JSON.write_text(
        json.dumps(audit_payload, indent=2) + "\n",
        encoding="utf-8",
    )

    summary_lines = [
        "MNQ PCS REGIME RESEARCH — CLOSER-STRIKE ECONOMICS",
        "=" * 66,
        "",
        f"Date-level requests: {len(requests)}",
        (
            "Complete distance quotes: "
            f"{int(economics['complete_quotes'].sum())}"
        ),
        (
            "Dates with zero BBO records: "
            + (
                ", ".join(zero_record_dates)
                if zero_record_dates
                else "None"
            )
        ),
        "",
        (
            "Gross edge equals midpoint credit minus historical mean "
            "expiration intrinsic loss."
        ),
        (
            "Conservative edge equals midpoint credit minus the upper "
            "year-block-bootstrap loss estimate."
        ),
        (
            "Midpoint is an indicative fair-value estimate, not a "
            "guaranteed fill."
        ),
    ]

    SUMMARY_TXT.write_text(
        "\n".join(summary_lines) + "\n",
        encoding="utf-8",
    )

    print("\nOutputs")
    print("-" * 108)
    print(f"Option quote snapshot: {SNAPSHOT_CSV}")
    print(f"Distance economics: {ECONOMICS_CSV}")
    print(f"Incremental comparison: {INCREMENTAL_CSV}")
    print(f"Bucket-distance summary: {BUCKET_SUMMARY_CSV}")
    print(f"Download audit: {AUDIT_JSON}")
    print(f"Analysis summary: {SUMMARY_TXT}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

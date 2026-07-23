"""Download and analyze exact MNQ option quotes for the VIX-bucket pilot.

Run from the project root:

    python scripts/download_vix_bucket_option_quotes.py \
        --download \
        --max-cost-usd 0.05

The script:

- reads the exact-leg option manifest;
- re-estimates only missing quote requests;
- skips dates for which Databento estimates zero BBO-1s records;
- enforces a combined hard cost ceiling;
- reuses existing raw DBN files;
- selects the latest valid quote at or before 3:45 p.m. New York time;
- calculates natural, midpoint, and optimistic spread credits;
- converts midpoint credit to MNQ dollars using the $2 multiplier;
- tests the 8.25-point credit floor; and
- summarizes results by VIX bucket.

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
RESULTS_DIR = ROOT / "results" / "vix_bucket_credit_pilot"
RAW_DIR = ROOT / "data" / "raw" / "databento" / "quotes"

MANIFEST_PATH = (
    RESULTS_DIR / "option_quote_manifest_all_expirations.csv"
)
SELECTION_PATH = (
    RESULTS_DIR / "selected_ladder_spreads_all_expirations.csv"
)

SNAPSHOT_CSV = RESULTS_DIR / "option_quote_snapshot.csv"
SPREAD_RESULTS_CSV = RESULTS_DIR / "spread_credit_results.csv"
BUCKET_SUMMARY_CSV = RESULTS_DIR / "bucket_credit_summary.csv"
AUDIT_JSON = RESULTS_DIR / "option_quote_download_audit.json"
SUMMARY_TXT = RESULTS_DIR / "option_quote_analysis_summary.txt"


MNQ_MULTIPLIER = 2.0
CREDIT_FLOOR_POINTS = 8.25
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
    """Load and validate the final exact-leg manifest and selections."""

    for path in (MANIFEST_PATH, SELECTION_PATH):
        if not path.is_file():
            raise FileNotFoundError(f"Missing required file: {path}")

    manifest = pd.read_csv(MANIFEST_PATH)
    selections = pd.read_csv(SELECTION_PATH)

    manifest_required = {
        "entry_date",
        "bucket",
        "vix_close",
        "ladder_distance_pct",
        "expiration",
        "dte_calendar",
        "underlying",
        "underlying_mid",
        "ladder_target_strike",
        "selected_short_strike",
        "selected_long_strike",
        "leg_name",
        "strike_price",
        "raw_symbol",
        "target_entry_timestamp_utc",
    }

    selection_required = {
        "entry_date",
        "bucket",
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

    numeric_columns = (
        "vix_close",
        "ladder_distance_pct",
        "dte_calendar",
        "underlying_mid",
        "ladder_target_strike",
        "selected_short_strike",
        "selected_long_strike",
        "strike_price",
    )

    for column in numeric_columns:
        manifest[column] = pd.to_numeric(
            manifest[column],
            errors="coerce",
        )

    if manifest.empty:
        raise ValueError("The final option manifest is empty.")

    return (
        manifest.sort_values(["entry_date", "leg_name"]).reset_index(
            drop=True
        ),
        selections.sort_values("entry_date").reset_index(drop=True),
    )


def raw_quote_path(entry_date: str) -> Path:
    """Return the ignored raw DBN path for one exact-leg request."""

    return RAW_DIR / (
        f"GLBX_MDP3_{entry_date}_MNQ_option_legs_bbo-1s.dbn.zst"
    )


def quote_window(
    target_utc: pd.Timestamp,
) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Return the ten-minute quote window centered on target time."""

    return (
        target_utc - pd.Timedelta(minutes=5),
        target_utc + pd.Timedelta(minutes=5),
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
    """Download and save one exact-symbol BBO-1s request."""

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


def empty_snapshot(
    symbols: list[str],
) -> pd.DataFrame:
    """Create an unavailable snapshot for a zero-record request."""

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


def build_bucket_summary(
    spread_results: pd.DataFrame,
) -> pd.DataFrame:
    """Aggregate usable midpoint credits by VIX bucket."""

    usable = spread_results.loc[
        spread_results["complete_quotes"]
        & spread_results["mid_credit_points"].notna()
    ].copy()

    if usable.empty:
        return pd.DataFrame(
            columns=[
                "bucket",
                "sample_dates",
                "mean_vix",
                "median_mid_credit_points",
                "minimum_mid_credit_points",
                "maximum_mid_credit_points",
                "median_mid_credit_dollars",
                "credit_floor_pass_count",
                "credit_floor_pass_rate",
            ]
        )

    rows: list[dict[str, Any]] = []

    for bucket, group in usable.groupby(
        "bucket",
        sort=False,
    ):
        rows.append(
            {
                "bucket": bucket,
                "sample_dates": len(group),
                "mean_vix": float(group["vix_close"].mean()),
                "median_mid_credit_points": float(
                    group["mid_credit_points"].median()
                ),
                "minimum_mid_credit_points": float(
                    group["mid_credit_points"].min()
                ),
                "maximum_mid_credit_points": float(
                    group["mid_credit_points"].max()
                ),
                "median_mid_credit_dollars": float(
                    group["mid_credit_dollars"].median()
                ),
                "credit_floor_pass_count": int(
                    group["mid_credit_floor_pass"].sum()
                ),
                "credit_floor_pass_rate": float(
                    group["mid_credit_floor_pass"].mean()
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
    output["_order"] = output["bucket"].map(
        desired_order
    ).fillna(999)

    return (
        output.sort_values(["_order", "bucket"])
        .drop(columns="_order")
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
    """Download or reuse exact option quotes and calculate credits."""

    import databento as db

    args = build_parser().parse_args()

    print(
        "MNQ PCS Regime Research — VIX-Bucket Option Quote Download"
    )
    print("=" * 70)

    manifest, selections = load_inputs()
    client = db.Historical(api_key())

    requests: list[dict[str, Any]] = []

    print("\nOption quote files and current estimates")
    print("-" * 134)

    for entry_date, group in manifest.groupby(
        "entry_date",
        sort=True,
    ):
        symbols = sorted(
            group["raw_symbol"].drop_duplicates().tolist()
        )

        if len(symbols) != 2:
            raise ValueError(
                f"{entry_date} has {len(symbols)} distinct symbols; "
                "expected exactly two."
            )

        target_values = group[
            "target_entry_timestamp_utc"
        ].drop_duplicates()

        if len(target_values) != 1:
            raise ValueError(
                f"{entry_date} has inconsistent target timestamps."
            )

        target_utc = pd.Timestamp(target_values.iloc[0])
        start_utc, end_utc = quote_window(target_utc)
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
                "group": group.copy(),
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

        record_text = (
            "local"
            if records == -1
            else f"{records:,}"
        )

        print(
            f"{entry_date} | "
            f"{action:<20} | "
            f"Symbols={len(symbols)} | "
            f"Cost=${cost:,.6f} | "
            f"Records={record_text}"
        )

    missing_downloads = [
        request
        for request in requests
        if not request["reuse"]
        and request["estimated_record_count"] > 0
    ]

    total_cost = sum(
        request["estimated_cost_usd"]
        for request in missing_downloads
    )

    zero_record_dates = [
        request["entry_date"]
        for request in requests
        if not request["reuse"]
        and request["estimated_record_count"] == 0
    ]

    print(
        f"\nTotal estimated cost for downloadable files: "
        f"${total_cost:,.6f}"
    )

    if zero_record_dates:
        print(
            "Dates with zero estimated BBO records: "
            + ", ".join(zero_record_dates)
        )

    if missing_downloads and not args.download:
        raise RuntimeError(
            "Missing files require --download and --max-cost-usd."
        )

    if missing_downloads and args.max_cost_usd <= 0:
        raise RuntimeError(
            "A positive --max-cost-usd is required."
        )

    if total_cost > args.max_cost_usd:
        raise RuntimeError(
            f"Blocked: ${total_cost:,.6f} exceeds "
            f"${args.max_cost_usd:,.6f}."
        )

    for request in missing_downloads:
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
    spread_rows: list[dict[str, Any]] = []

    print("\nOption quote availability")
    print("-" * 150)

    for request in requests:
        entry_date = request["entry_date"]
        group = request["group"]
        symbols = request["symbols"]

        if request["output_path"].is_file():
            quotes = normalize_quotes(
                db,
                request["output_path"],
            )

            snapshot = select_snapshot(
                quotes,
                symbols,
                request["target_utc"],
            )
        else:
            snapshot = empty_snapshot(symbols)

        first = group.iloc[0]

        metadata = group[
            [
                "entry_date",
                "bucket",
                "vix_close",
                "ladder_distance_pct",
                "expiration",
                "dte_calendar",
                "underlying",
                "underlying_mid",
                "ladder_target_strike",
                "selected_short_strike",
                "selected_long_strike",
                "leg_name",
                "strike_price",
                "raw_symbol",
            ]
        ].copy()

        snapshot_output = metadata.merge(
            snapshot,
            on="raw_symbol",
            how="left",
            validate="one_to_one",
        )

        snapshot_output["target_entry_timestamp_utc"] = request[
            "target_utc"
        ]

        snapshot_frames.append(snapshot_output)

        print(
            f"{entry_date} | "
            f"Bucket={first['bucket']:<10} | "
            f"VIX={float(first['vix_close']):>6.2f}"
        )

        for row in snapshot_output.sort_values(
            "leg_name"
        ).itertuples(index=False):
            if bool(row.quote_available):
                print(
                    f"  {row.leg_name:<10} | "
                    f"{row.raw_symbol:<18} | "
                    f"{row.quote_timestamp_utc} | "
                    f"Age={row.quote_age_seconds:>6.1f}s | "
                    f"Bid={row.bid:>8.2f} | "
                    f"Ask={row.ask:>8.2f}"
                )
            else:
                print(
                    f"  {row.leg_name:<10} | "
                    f"{row.raw_symbol:<18} | "
                    f"NO VALID QUOTE | "
                    f"Records={int(row.records_received)}"
                )

        short = snapshot_output.loc[
            snapshot_output["leg_name"] == "short_put"
        ].iloc[0]

        long = snapshot_output.loc[
            snapshot_output["leg_name"] == "long_put"
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
                    pd.Timestamp(short["quote_timestamp_utc"])
                    - pd.Timestamp(long["quote_timestamp_utc"])
                ).total_seconds()
            )
        else:
            natural_credit = np.nan
            mid_credit = np.nan
            optimistic_credit = np.nan
            quote_width = np.nan
            timestamp_gap = np.nan

        spread_rows.append(
            {
                "entry_date": entry_date,
                "bucket": str(first["bucket"]),
                "vix_close": float(first["vix_close"]),
                "ladder_distance_pct": float(
                    first["ladder_distance_pct"]
                ),
                "expiration": first["expiration"],
                "dte_calendar": int(first["dte_calendar"]),
                "underlying": str(first["underlying"]),
                "underlying_mid": float(
                    first["underlying_mid"]
                ),
                "ladder_target_strike": float(
                    first["ladder_target_strike"]
                ),
                "selected_short_strike": float(
                    first["selected_short_strike"]
                ),
                "selected_long_strike": float(
                    first["selected_long_strike"]
                ),
                "short_symbol": str(short["raw_symbol"]),
                "long_symbol": str(long["raw_symbol"]),
                "complete_quotes": complete,
                "short_quote_age_seconds": short[
                    "quote_age_seconds"
                ],
                "long_quote_age_seconds": long[
                    "quote_age_seconds"
                ],
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
                "natural_credit_floor_pass": (
                    bool(
                        natural_credit
                        >= CREDIT_FLOOR_POINTS
                    )
                    if pd.notna(natural_credit)
                    else False
                ),
                "mid_credit_floor_pass": (
                    bool(
                        mid_credit
                        >= CREDIT_FLOOR_POINTS
                    )
                    if pd.notna(mid_credit)
                    else False
                ),
                "optimistic_credit_floor_pass": (
                    bool(
                        optimistic_credit
                        >= CREDIT_FLOOR_POINTS
                    )
                    if pd.notna(optimistic_credit)
                    else False
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

    spread_results = pd.DataFrame(
        spread_rows
    ).sort_values(
        "entry_date"
    ).reset_index(drop=True)

    bucket_summary = build_bucket_summary(
        spread_results
    )

    print("\nSpread credit results")
    print("-" * 190)

    for row in spread_results.itertuples(index=False):
        if row.complete_quotes:
            print(
                f"{row.entry_date} | "
                f"Bucket={row.bucket:<10} | "
                f"VIX={row.vix_close:>6.2f} | "
                f"DTE={row.dte_calendar:>2} | "
                f"Spread={row.selected_short_strike:,.0f}/"
                f"{row.selected_long_strike:,.0f} | "
                f"Natural={row.natural_credit_points:>7.2f} | "
                f"Mid={row.mid_credit_points:>7.2f} "
                f"(${row.mid_credit_dollars:>6.2f}) | "
                f"Optimistic={row.optimistic_credit_points:>7.2f} | "
                f"Mid floor="
                f"{'PASS' if row.mid_credit_floor_pass else 'FAIL'}"
            )
        else:
            print(
                f"{row.entry_date} | "
                f"Bucket={row.bucket:<10} | "
                f"NO COMPLETE QUOTE | "
                f"Status={row.data_status}"
            )

    print("\nBucket summary")
    print("-" * 142)

    if bucket_summary.empty:
        print("No complete spread quotes were available.")
    else:
        for row in bucket_summary.itertuples(index=False):
            print(
                f"{row.bucket:<10} | "
                f"N={row.sample_dates:>2} | "
                f"Mean VIX={row.mean_vix:>6.2f} | "
                f"Median mid={row.median_mid_credit_points:>6.2f} pts "
                f"(${row.median_mid_credit_dollars:>6.2f}) | "
                f"Range={row.minimum_mid_credit_points:>6.2f}–"
                f"{row.maximum_mid_credit_points:>6.2f} | "
                f"Floor pass={row.credit_floor_pass_count}/"
                f"{row.sample_dates}"
            )

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    snapshot_all.to_csv(
        SNAPSHOT_CSV,
        index=False,
    )

    spread_results.to_csv(
        SPREAD_RESULTS_CSV,
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
        "credit_floor_points": CREDIT_FLOOR_POINTS,
        "estimated_downloadable_cost_usd": total_cost,
        "zero_record_dates": zero_record_dates,
        "complete_spread_quotes": int(
            spread_results["complete_quotes"].sum()
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

    non_executable = selections.loc[
        selections["selection_status"] != "Selected",
        "entry_date",
    ].tolist()

    summary_lines = [
        "MNQ PCS REGIME RESEARCH — VIX-BUCKET OPTION QUOTES",
        "=" * 66,
        "",
        f"Executable spread dates requested: {len(requests)}",
        (
            "Complete spread quotes: "
            f"{int(spread_results['complete_quotes'].sum())}"
        ),
        (
            "Dates with zero BBO records: "
            + (
                ", ".join(zero_record_dates)
                if zero_record_dates
                else "None"
            )
        ),
        (
            "Non-executable ladder dates: "
            + (
                ", ".join(non_executable)
                if non_executable
                else "None"
            )
        ),
        (
            "Midpoint credit-floor passes: "
            f"{int(spread_results['mid_credit_floor_pass'].sum())}"
        ),
        "",
        (
            "Midpoint credits are indicative estimates, not "
            "guaranteed fills."
        ),
    ]

    SUMMARY_TXT.write_text(
        "\n".join(summary_lines) + "\n",
        encoding="utf-8",
    )

    print("\nOutputs")
    print("-" * 106)
    print(f"Option quote snapshot: {SNAPSHOT_CSV}")
    print(f"Spread credit results: {SPREAD_RESULTS_CSV}")
    print(f"Bucket summary: {BUCKET_SUMMARY_CSV}")
    print(f"Download audit: {AUDIT_JSON}")
    print(f"Analysis summary: {SUMMARY_TXT}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

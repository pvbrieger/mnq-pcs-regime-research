"""Download MNQ futures quotes and select exact ladder spreads.

Run from the project root:

    python scripts/download_vix_bucket_underlying_quotes.py \
        --download \
        --max-cost-usd 0.75

This script:

- reads the preferred-expiration underlying manifest;
- re-estimates only missing futures quote files;
- enforces a combined hard cost ceiling;
- reuses existing raw DBN files;
- selects the latest valid BBO at or before 3:45 p.m. New York time;
- calculates the ladder target from the actual MNQ futures midpoint;
- selects the highest exact 150-point short strike at or below that target; and
- writes the exact option-leg manifest for the final quote-cost estimate.

No option quotes are downloaded in this checkpoint.
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
RAW_QUOTE_DIR = ROOT / "data" / "raw" / "databento" / "quotes"

UNDERLYING_MANIFEST_PATH = (
    RESULTS_DIR / "underlying_quote_manifest.csv"
)
CANDIDATES_PATH = (
    RESULTS_DIR / "mnq_option_definition_candidates.csv"
)

SNAPSHOT_CSV = RESULTS_DIR / "underlying_quote_snapshot.csv"
SPREAD_SELECTION_CSV = RESULTS_DIR / "selected_ladder_spreads.csv"
OPTION_MANIFEST_CSV = RESULTS_DIR / "option_quote_manifest.csv"
AUDIT_JSON = RESULTS_DIR / "underlying_quote_download_audit.json"
SUMMARY_TXT = RESULTS_DIR / "underlying_quote_and_spread_summary.txt"


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
    """Load and validate the underlying and option-definition manifests."""

    if not UNDERLYING_MANIFEST_PATH.is_file():
        raise FileNotFoundError(
            f"Missing underlying manifest: {UNDERLYING_MANIFEST_PATH}"
        )

    if not CANDIDATES_PATH.is_file():
        raise FileNotFoundError(
            f"Missing definition candidates: {CANDIDATES_PATH}"
        )

    manifest = pd.read_csv(UNDERLYING_MANIFEST_PATH)
    candidates = pd.read_csv(CANDIDATES_PATH)

    manifest_required = {
        "entry_date",
        "bucket",
        "vix_close",
        "ladder_distance_pct",
        "expiration",
        "dte_calendar",
        "underlying",
        "target_entry_timestamp_local",
        "quote_window_start_utc",
        "quote_window_end_utc",
    }

    candidate_required = {
        "entry_date",
        "expiration",
        "dte_calendar",
        "underlying",
        "strike_price",
        "raw_symbol",
    }

    for name, frame, required in (
        ("underlying manifest", manifest, manifest_required),
        ("definition candidates", candidates, candidate_required),
    ):
        missing = sorted(required.difference(frame.columns))

        if missing:
            raise ValueError(
                f"{name} is missing fields: {', '.join(missing)}"
            )

    for frame in (manifest, candidates):
        frame["entry_date"] = (
            frame["entry_date"].fillna("").astype(str).str.strip()
        )

        frame["underlying"] = (
            frame["underlying"].fillna("").astype(str).str.strip()
        )

        frame["expiration"] = pd.to_datetime(
            frame["expiration"],
            errors="coerce",
            utc=True,
        )

    candidates["raw_symbol"] = (
        candidates["raw_symbol"].fillna("").astype(str).str.strip()
    )

    candidates["strike_price"] = pd.to_numeric(
        candidates["strike_price"],
        errors="coerce",
    )

    candidates["dte_calendar"] = pd.to_numeric(
        candidates["dte_calendar"],
        errors="coerce",
    )

    for column in (
        "vix_close",
        "ladder_distance_pct",
        "dte_calendar",
    ):
        manifest[column] = pd.to_numeric(
            manifest[column],
            errors="coerce",
        )

    for column in (
        "target_entry_timestamp_local",
        "quote_window_start_utc",
        "quote_window_end_utc",
    ):
        manifest[column] = pd.to_datetime(
            manifest[column],
            errors="coerce",
            utc=True,
        )

    if manifest.empty:
        raise ValueError("The underlying quote manifest is empty.")

    return manifest.sort_values("entry_date"), candidates


def raw_quote_path(entry_date: str) -> Path:
    """Return the ignored raw futures BBO path."""

    return RAW_QUOTE_DIR / (
        f"GLBX_MDP3_{entry_date}_MNQ_underlying_bbo-1s.dbn.zst"
    )


def estimate_request(
    client: Any,
    row: pd.Series,
) -> tuple[float, int]:
    """Estimate one exact-symbol futures BBO request."""

    kwargs = dict(
        dataset="GLBX.MDP3",
        symbols=[str(row["underlying"])],
        stype_in="raw_symbol",
        schema="bbo-1s",
        start=row["quote_window_start_utc"].isoformat(),
        end=row["quote_window_end_utc"].isoformat(),
    )

    return (
        float(client.metadata.get_cost(**kwargs)),
        int(client.metadata.get_record_count(**kwargs)),
    )


def download_request(
    client: Any,
    row: pd.Series,
    output_path: Path,
) -> None:
    """Download one exact-symbol futures BBO request."""

    output_path.parent.mkdir(parents=True, exist_ok=True)

    store = client.timeseries.get_range(
        dataset="GLBX.MDP3",
        symbols=[str(row["underlying"])],
        stype_in="raw_symbol",
        schema="bbo-1s",
        start=row["quote_window_start_utc"].isoformat(),
        end=row["quote_window_end_utc"].isoformat(),
    )

    store.to_file(output_path)


def normalize_quotes(db: Any, file_path: Path) -> pd.DataFrame:
    """Load a BBO DBN file and normalize its quote fields."""

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


def select_quote(
    quotes: pd.DataFrame,
    symbol: str,
    target_utc: pd.Timestamp,
    maximum_age_minutes: float = 5.0,
) -> dict[str, Any]:
    """Select the latest valid quote at or before target time."""

    symbol_rows = quotes.loc[
        quotes["raw_symbol"] == symbol
    ].copy()

    oldest = target_utc - pd.Timedelta(
        minutes=maximum_age_minutes
    )

    valid = symbol_rows.loc[
        (symbol_rows["quote_timestamp_utc"] <= target_utc)
        & (symbol_rows["quote_timestamp_utc"] >= oldest)
        & symbol_rows["bid"].notna()
        & symbol_rows["ask"].notna()
        & (symbol_rows["bid"] >= 0)
        & (symbol_rows["ask"] >= symbol_rows["bid"])
    ]

    if valid.empty:
        return {
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

    selected = valid.iloc[-1]

    return {
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


def exact_pairs_for_expiration(
    candidates: pd.DataFrame,
    entry_date: str,
    expiration: pd.Timestamp,
    underlying: str,
    width: float = 150.0,
    tolerance: float = 0.01,
) -> pd.DataFrame:
    """Return exact 150-point pairs and their raw symbols."""

    subset = candidates.loc[
        (candidates["entry_date"] == entry_date)
        & (candidates["expiration"] == expiration)
        & (candidates["underlying"] == underlying)
    ].copy()

    if subset.empty:
        return pd.DataFrame()

    # A strike should normally map to one raw symbol in a selected series.
    # Deterministic sorting keeps the choice reproducible if duplicates occur.
    strike_map = (
        subset.sort_values(
            ["strike_price", "raw_symbol"]
        )
        .drop_duplicates(
            "strike_price",
            keep="first",
        )
        .set_index("strike_price")["raw_symbol"]
        .to_dict()
    )

    strikes = sorted(float(value) for value in strike_map)

    rows: list[dict[str, Any]] = []

    for short_strike in strikes:
        matching_longs = [
            long_strike
            for long_strike in strikes
            if abs(
                (short_strike - width) - long_strike
            ) <= tolerance
        ]

        if not matching_longs:
            continue

        long_strike = matching_longs[0]

        rows.append(
            {
                "short_strike": short_strike,
                "long_strike": long_strike,
                "short_symbol": strike_map[short_strike],
                "long_symbol": strike_map[long_strike],
            }
        )

    return pd.DataFrame(rows).sort_values(
        "short_strike"
    ).reset_index(drop=True)


def build_parser() -> argparse.ArgumentParser:
    """Build command-line parser."""

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--download",
        action="store_true",
        help="Authorize missing historical futures quote downloads.",
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
    """Download or reuse futures quotes and select ladder spreads."""

    import databento as db

    args = build_parser().parse_args()

    print(
        "MNQ PCS Regime Research — Underlying Quotes and Spread Selection"
    )
    print("=" * 72)

    manifest, candidates = load_inputs()
    client = db.Historical(api_key())

    requests: list[dict[str, Any]] = []

    print("\nUnderlying quote files and current estimates")
    print("-" * 124)

    for _, row in manifest.iterrows():
        entry_date = str(row["entry_date"])
        output_path = raw_quote_path(entry_date)

        reuse = (
            output_path.is_file()
            and not args.force_redownload
        )

        if reuse:
            cost, records = 0.0, 0
        else:
            cost, records = estimate_request(
                client,
                row,
            )

        requests.append(
            {
                "entry_date": entry_date,
                "row": row,
                "output_path": output_path,
                "reuse": reuse,
                "estimated_cost_usd": cost,
                "estimated_record_count": records,
                "downloaded_during_this_run": False,
            }
        )

        print(
            f"{entry_date} | "
            f"Underlying={row['underlying']:<6} | "
            f"{'reuse local file' if reuse else 'download required':<18} | "
            f"Cost=${cost:,.6f} | "
            f"Records={records:,}"
        )

    missing = [
        request
        for request in requests
        if not request["reuse"]
    ]

    total_cost = sum(
        request["estimated_cost_usd"]
        for request in missing
    )

    print(
        f"\nTotal estimated cost for missing files: "
        f"${total_cost:,.6f}"
    )

    if missing and not args.download:
        raise RuntimeError(
            "Missing files require --download and --max-cost-usd."
        )

    if missing and args.max_cost_usd <= 0:
        raise RuntimeError(
            "A positive --max-cost-usd is required."
        )

    if total_cost > args.max_cost_usd:
        raise RuntimeError(
            f"Blocked: ${total_cost:,.6f} exceeds "
            f"${args.max_cost_usd:,.6f}."
        )

    for request in missing:
        print(
            f"\nDownloading {request['entry_date']}; "
            f"estimated cost "
            f"${request['estimated_cost_usd']:,.6f}."
        )

        download_request(
            client,
            request["row"],
            request["output_path"],
        )

        request["downloaded_during_this_run"] = True

    snapshot_rows: list[dict[str, Any]] = []
    spread_rows: list[dict[str, Any]] = []
    option_manifest_rows: list[dict[str, Any]] = []

    print("\nUnderlying quote snapshots")
    print("-" * 142)

    for request in requests:
        row = request["row"]
        entry_date = request["entry_date"]
        underlying = str(row["underlying"])

        quotes = normalize_quotes(
            db,
            request["output_path"],
        )

        target_utc = pd.Timestamp(
            row["target_entry_timestamp_local"]
        ).tz_convert("UTC")

        quote = select_quote(
            quotes,
            underlying,
            target_utc,
        )

        snapshot_row = {
            "entry_date": entry_date,
            "bucket": str(row["bucket"]),
            "vix_close": float(row["vix_close"]),
            "ladder_distance_pct": float(
                row["ladder_distance_pct"]
            ),
            "expiration": row["expiration"],
            "dte_calendar": int(row["dte_calendar"]),
            "underlying": underlying,
            "target_entry_timestamp_utc": target_utc,
            **quote,
        }

        snapshot_rows.append(snapshot_row)

        if quote["quote_available"]:
            print(
                f"{entry_date} | "
                f"Bucket={row['bucket']:<10} | "
                f"VIX={float(row['vix_close']):>6.2f} | "
                f"{underlying:<6} | "
                f"Bid={quote['bid']:>10.2f} | "
                f"Ask={quote['ask']:>10.2f} | "
                f"Mid={quote['mid']:>10.2f} | "
                f"Age={quote['quote_age_seconds']:>6.1f}s"
            )
        else:
            print(
                f"{entry_date} | "
                f"Bucket={row['bucket']:<10} | "
                f"{underlying:<6} | "
                f"NO VALID QUOTE | "
                f"Records={quote['records_received']}"
            )

        expiration = pd.Timestamp(row["expiration"])

        pairs = exact_pairs_for_expiration(
            candidates=candidates,
            entry_date=entry_date,
            expiration=expiration,
            underlying=underlying,
        )

        if not quote["quote_available"]:
            spread_rows.append(
                {
                    "entry_date": entry_date,
                    "bucket": str(row["bucket"]),
                    "vix_close": float(row["vix_close"]),
                    "ladder_distance_pct": float(
                        row["ladder_distance_pct"]
                    ),
                    "expiration": expiration,
                    "dte_calendar": int(row["dte_calendar"]),
                    "underlying": underlying,
                    "underlying_mid": np.nan,
                    "ladder_target_strike": np.nan,
                    "selected_short_strike": np.nan,
                    "selected_long_strike": np.nan,
                    "selected_short_pct_below_mnq": np.nan,
                    "target_to_selected_short_slippage_points": np.nan,
                    "short_symbol": "",
                    "long_symbol": "",
                    "selection_status": "No valid underlying quote",
                }
            )
            continue

        target_strike = quote["mid"] * (
            1.0
            - float(row["ladder_distance_pct"])
            / 100.0
        )

        eligible_pairs = pairs.loc[
            pairs["short_strike"] <= target_strike
        ].copy()

        if eligible_pairs.empty:
            spread_rows.append(
                {
                    "entry_date": entry_date,
                    "bucket": str(row["bucket"]),
                    "vix_close": float(row["vix_close"]),
                    "ladder_distance_pct": float(
                        row["ladder_distance_pct"]
                    ),
                    "expiration": expiration,
                    "dte_calendar": int(row["dte_calendar"]),
                    "underlying": underlying,
                    "underlying_mid": quote["mid"],
                    "ladder_target_strike": target_strike,
                    "selected_short_strike": np.nan,
                    "selected_long_strike": np.nan,
                    "selected_short_pct_below_mnq": np.nan,
                    "target_to_selected_short_slippage_points": np.nan,
                    "short_symbol": "",
                    "long_symbol": "",
                    "selection_status": (
                        "No exact 150-point pair at or below target"
                    ),
                }
            )
            continue

        selected = eligible_pairs.iloc[-1]

        selected_short = float(selected["short_strike"])
        selected_long = float(selected["long_strike"])

        spread_rows.append(
            {
                "entry_date": entry_date,
                "bucket": str(row["bucket"]),
                "vix_close": float(row["vix_close"]),
                "ladder_distance_pct": float(
                    row["ladder_distance_pct"]
                ),
                "expiration": expiration,
                "dte_calendar": int(row["dte_calendar"]),
                "underlying": underlying,
                "underlying_mid": quote["mid"],
                "ladder_target_strike": target_strike,
                "selected_short_strike": selected_short,
                "selected_long_strike": selected_long,
                "selected_short_pct_below_mnq": (
                    (
                        quote["mid"] - selected_short
                    )
                    / quote["mid"]
                    * 100.0
                ),
                "target_to_selected_short_slippage_points": (
                    target_strike - selected_short
                ),
                "short_symbol": str(
                    selected["short_symbol"]
                ),
                "long_symbol": str(
                    selected["long_symbol"]
                ),
                "selection_status": "Selected",
            }
        )

        for leg_name, strike, symbol in (
            (
                "short_put",
                selected_short,
                str(selected["short_symbol"]),
            ),
            (
                "long_put",
                selected_long,
                str(selected["long_symbol"]),
            ),
        ):
            option_manifest_rows.append(
                {
                    "entry_date": entry_date,
                    "bucket": str(row["bucket"]),
                    "vix_close": float(row["vix_close"]),
                    "ladder_distance_pct": float(
                        row["ladder_distance_pct"]
                    ),
                    "expiration": expiration,
                    "dte_calendar": int(row["dte_calendar"]),
                    "underlying": underlying,
                    "underlying_mid": quote["mid"],
                    "ladder_target_strike": target_strike,
                    "selected_short_strike": selected_short,
                    "selected_long_strike": selected_long,
                    "leg_name": leg_name,
                    "strike_price": strike,
                    "raw_symbol": symbol,
                    "target_entry_timestamp_local": row[
                        "target_entry_timestamp_local"
                    ],
                    "quote_window_start_utc": row[
                        "quote_window_start_utc"
                    ],
                    "quote_window_end_utc": row[
                        "quote_window_end_utc"
                    ],
                }
            )

    snapshot = pd.DataFrame(snapshot_rows)
    spread_selection = pd.DataFrame(spread_rows)
    option_manifest = pd.DataFrame(option_manifest_rows)

    print("\nSelected ladder spreads")
    print("-" * 172)

    for row in spread_selection.itertuples(index=False):
        if row.selection_status == "Selected":
            print(
                f"{row.entry_date} | "
                f"Bucket={row.bucket:<10} | "
                f"VIX={row.vix_close:>6.2f} | "
                f"MNQ={row.underlying_mid:>10.2f} | "
                f"Target={row.ladder_target_strike:>10.2f} | "
                f"Spread={row.selected_short_strike:,.0f}/"
                f"{row.selected_long_strike:,.0f} | "
                f"Actual distance="
                f"{row.selected_short_pct_below_mnq:>5.2f}% | "
                f"Strike slippage="
                f"{row.target_to_selected_short_slippage_points:>7.2f}"
            )
        else:
            print(
                f"{row.entry_date} | "
                f"Bucket={row.bucket:<10} | "
                f"{row.selection_status}"
            )

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    snapshot.to_csv(
        SNAPSHOT_CSV,
        index=False,
    )

    spread_selection.to_csv(
        SPREAD_SELECTION_CSV,
        index=False,
    )

    option_manifest.to_csv(
        OPTION_MANIFEST_CSV,
        index=False,
    )

    selected_count = int(
        (spread_selection["selection_status"] == "Selected").sum()
    )

    audit_payload = {
        "dataset": "GLBX.MDP3",
        "schema": "bbo-1s",
        "estimated_missing_cost_usd": total_cost,
        "usable_underlying_quotes": int(
            snapshot["quote_available"].sum()
        ),
        "selected_spreads": selected_count,
        "files": [
            {
                "entry_date": request["entry_date"],
                "raw_path": str(request["output_path"]),
                "file_reused": request["reuse"],
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
        "MNQ PCS REGIME RESEARCH — UNDERLYING QUOTES AND SPREADS",
        "=" * 68,
        "",
        f"Input dates: {len(manifest)}",
        (
            "Valid underlying quotes: "
            f"{int(snapshot['quote_available'].sum())}"
        ),
        f"Exact ladder spreads selected: {selected_count}",
        f"Estimated download cost: ${total_cost:,.6f}",
        "",
        "No option quotes were downloaded.",
    ]

    SUMMARY_TXT.write_text(
        "\n".join(summary_lines) + "\n",
        encoding="utf-8",
    )

    print("\nOutputs")
    print("-" * 106)
    print(f"Underlying snapshot: {SNAPSHOT_CSV}")
    print(f"Selected spreads: {SPREAD_SELECTION_CSV}")
    print(f"Option quote manifest: {OPTION_MANIFEST_CSV}")
    print(f"Download audit: {AUDIT_JSON}")
    print(f"Summary: {SUMMARY_TXT}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

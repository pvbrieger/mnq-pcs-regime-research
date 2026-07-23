"""Download and analyze BBO-1s quotes for the April pilot dates.

Run from the project root:

    python scripts/download_databento_multidate_quotes.py \
        --download \
        --max-cost-usd 0.01

The script:

- reads the exact-symbol manifest produced by the definition step;
- re-estimates only quote files that are still missing;
- enforces a combined hard cost ceiling;
- reuses existing raw DBN files;
- selects the latest valid quote at or before 3:45 p.m. New York time;
- calculates natural, midpoint, and optimistic spread credits;
- records the actual MNQ futures midpoint and strike distances; and
- labels degraded dates as exploratory.

Raw DBN files remain under the Git-ignored raw-data directory.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
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
RESULTS_DIR = ROOT / "results" / "databento_multidate_pilot"
RAW_DIR = ROOT / "data" / "raw" / "databento" / "quotes"

SNAPSHOT_CSV = RESULTS_DIR / "quote_snapshot.csv"
SPREAD_CSV = RESULTS_DIR / "spread_credits.csv"
COMPARISON_CSV = RESULTS_DIR / "date_comparison.csv"
AUDIT_JSON = RESULTS_DIR / "quote_download_audit.json"
SUMMARY_TXT = RESULTS_DIR / "quote_analysis.txt"


def load_config() -> dict[str, Any]:
    """Load the multi-date pilot configuration."""

    if not CONFIG_PATH.is_file():
        raise FileNotFoundError(f"Missing configuration: {CONFIG_PATH}")

    with CONFIG_PATH.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def load_manifest() -> pd.DataFrame:
    """Load and validate the exact option-leg manifest."""

    if not MANIFEST_PATH.is_file():
        raise FileNotFoundError(f"Missing quote manifest: {MANIFEST_PATH}")

    frame = pd.read_csv(MANIFEST_PATH)

    required = {
        "entry_date",
        "score",
        "ndx_close",
        "expiration",
        "dte_calendar",
        "underlying",
        "candidate_type",
        "leg_name",
        "side",
        "target_short_strike",
        "selected_short_strike",
        "selected_long_strike",
        "strike_price",
        "raw_symbol",
    }

    missing = sorted(required.difference(frame.columns))

    if missing:
        raise ValueError(
            "Quote manifest is missing fields: " + ", ".join(missing)
        )

    frame["entry_date"] = (
        frame["entry_date"].fillna("").astype(str).str.strip()
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
        "side",
        "raw_symbol",
    ):
        frame[column] = (
            frame[column].fillna("").astype(str).str.strip()
        )

    for column in (
        "score",
        "ndx_close",
        "dte_calendar",
        "target_short_strike",
        "selected_short_strike",
        "selected_long_strike",
        "strike_price",
    ):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")

    if frame.empty:
        raise ValueError("The quote manifest contains no executable spreads.")

    if frame[
        [
            "expiration",
            "score",
            "ndx_close",
            "dte_calendar",
            "strike_price",
        ]
    ].isna().any(axis=None):
        raise ValueError("The quote manifest contains invalid values.")

    return frame


def api_key() -> str:
    """Load the Databento key without displaying it."""

    load_dotenv(ROOT / ".env", override=True)

    value = os.getenv("DATABENTO_API_KEY", "").strip()

    if not value.startswith("db-"):
        raise RuntimeError(
            "A valid DATABENTO_API_KEY is required in the project .env file."
        )

    return value


def quote_window(
    entry_date: str,
    config: dict[str, Any],
) -> tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp]:
    """Return target local time and UTC request window."""

    date = pd.Timestamp(entry_date)
    hour, minute = (
        int(part)
        for part in str(config.get("target_entry_time", "15:45")).split(":")
    )

    timezone = ZoneInfo(
        str(config.get("entry_timezone", "America/New_York"))
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
        minutes=float(config.get("quote_window_minutes", 10.0)) / 2.0
    )

    return (
        target_local,
        (target_local - half_window).tz_convert("UTC"),
        (target_local + half_window).tz_convert("UTC"),
    )


def raw_quote_path(dataset: str, entry_date: str) -> Path:
    """Return the ignored raw DBN path for one date."""

    return RAW_DIR / (
        f"{dataset.replace('.', '_')}_{entry_date}_bbo-1s.dbn.zst"
    )


def condition_for_date(
    client: Any,
    dataset: str,
    entry_date: str,
) -> str:
    """Return Databento's day-level dataset condition."""

    rows = client.metadata.get_dataset_condition(
        dataset=dataset,
        start_date=entry_date,
        end_date=entry_date,
    )

    if not rows:
        return "unknown"

    row = rows[0]

    if isinstance(row, dict):
        return str(row.get("condition", "unknown"))

    return str(getattr(row, "condition", "unknown"))


def estimate_request(
    client: Any,
    dataset: str,
    symbols: list[str],
    start_utc: pd.Timestamp,
    end_utc: pd.Timestamp,
) -> tuple[float, int]:
    """Estimate one exact-symbol BBO-1s request."""

    kwargs = dict(
        dataset=dataset,
        symbols=symbols,
        stype_in="raw_symbol",
        schema="bbo-1s",
        start=start_utc.isoformat(),
        end=end_utc.isoformat(),
    )

    cost = float(client.metadata.get_cost(**kwargs))
    records = int(client.metadata.get_record_count(**kwargs))

    return cost, records


def download_request(
    client: Any,
    dataset: str,
    symbols: list[str],
    start_utc: pd.Timestamp,
    end_utc: pd.Timestamp,
    output_path: Path,
) -> None:
    """Download and save one exact-symbol BBO-1s request."""

    output_path.parent.mkdir(parents=True, exist_ok=True)

    store = client.timeseries.get_range(
        dataset=dataset,
        symbols=symbols,
        stype_in="raw_symbol",
        schema="bbo-1s",
        start=start_utc.isoformat(),
        end=end_utc.isoformat(),
    )

    store.to_file(output_path)


def normalized_quotes(db: Any, raw_path: Path) -> pd.DataFrame:
    """Load a DBN file and normalize its BBO columns."""

    store = db.DBNStore.from_file(raw_path)

    frame = store.to_df(
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
            f"Unable to identify timestamp or symbol in {raw_path.name}."
        )

    required = {"bid_px_00", "ask_px_00"}
    missing = sorted(required.difference(frame.columns))

    if missing:
        raise ValueError(
            f"{raw_path.name} is missing BBO fields: {', '.join(missing)}"
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
            "bid": pd.to_numeric(frame["bid_px_00"], errors="coerce"),
            "ask": pd.to_numeric(frame["ask_px_00"], errors="coerce"),
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

    return output.dropna(subset=["quote_timestamp_utc"]).sort_values(
        ["raw_symbol", "quote_timestamp_utc"]
    )


def select_snapshot(
    quotes: pd.DataFrame,
    symbols: list[str],
    target_utc: pd.Timestamp,
    maximum_age_minutes: float,
) -> pd.DataFrame:
    """Select the latest valid quote at or before the decision time."""

    maximum_age = pd.Timedelta(minutes=maximum_age_minutes)
    rows: list[dict[str, Any]] = []

    for symbol in symbols:
        symbol_rows = quotes.loc[quotes["raw_symbol"] == symbol].copy()

        valid = symbol_rows.loc[
            (symbol_rows["quote_timestamp_utc"] <= target_utc)
            & (
                symbol_rows["quote_timestamp_utc"]
                >= target_utc - maximum_age
            )
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
        age_seconds = (
            target_utc - selected["quote_timestamp_utc"]
        ).total_seconds()

        rows.append(
            {
                "raw_symbol": symbol,
                "quote_available": True,
                "records_received": len(symbol_rows),
                "quote_timestamp_utc": selected["quote_timestamp_utc"],
                "quote_age_seconds": age_seconds,
                "bid": float(selected["bid"]),
                "ask": float(selected["ask"]),
                "mid": float((selected["bid"] + selected["ask"]) / 2.0),
                "bid_size": selected["bid_size"],
                "ask_size": selected["ask_size"],
            }
        )

    return pd.DataFrame(rows)


def analyze_date(
    entry_date: str,
    date_manifest: pd.DataFrame,
    snapshot: pd.DataFrame,
    condition: str,
    target_local: pd.Timestamp,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Build quote snapshot, spread rows, and one comparison row."""

    underlying_symbol = date_manifest["underlying"].drop_duplicates().iloc[0]

    underlying_quote = snapshot.loc[
        snapshot["raw_symbol"] == underlying_symbol
    ]

    if len(underlying_quote) != 1:
        raise ValueError(
            f"Expected one underlying quote for {entry_date}."
        )

    future_row = underlying_quote.iloc[0]
    future_mid = float(future_row["mid"]) if future_row[
        "quote_available"
    ] else np.nan

    option_metadata = date_manifest[
        [
            "entry_date",
            "score",
            "ndx_close",
            "expiration",
            "dte_calendar",
            "underlying",
            "candidate_type",
            "leg_name",
            "side",
            "target_short_strike",
            "selected_short_strike",
            "selected_long_strike",
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

    option_snapshot["dataset_condition"] = condition
    option_snapshot["target_entry_timestamp_local"] = target_local.isoformat()
    option_snapshot["instrument_role"] = "option_leg"

    underlying_output = future_row.to_frame().T
    underlying_output["entry_date"] = entry_date
    underlying_output["score"] = date_manifest["score"].iloc[0]
    underlying_output["ndx_close"] = date_manifest["ndx_close"].iloc[0]
    underlying_output["expiration"] = pd.NaT
    underlying_output["dte_calendar"] = np.nan
    underlying_output["underlying"] = underlying_symbol
    underlying_output["candidate_type"] = ""
    underlying_output["leg_name"] = ""
    underlying_output["side"] = ""
    underlying_output["target_short_strike"] = np.nan
    underlying_output["selected_short_strike"] = np.nan
    underlying_output["selected_long_strike"] = np.nan
    underlying_output["strike_price"] = np.nan
    underlying_output["dataset_condition"] = condition
    underlying_output[
        "target_entry_timestamp_local"
    ] = target_local.isoformat()
    underlying_output["instrument_role"] = "underlying_future"

    snapshot_output = pd.concat(
        [option_snapshot, underlying_output],
        ignore_index=True,
        sort=False,
    )

    spread_rows: list[dict[str, Any]] = []

    group_columns = [
        "entry_date",
        "score",
        "ndx_close",
        "expiration",
        "dte_calendar",
        "underlying",
        "candidate_type",
        "target_short_strike",
        "selected_short_strike",
        "selected_long_strike",
    ]

    for group_key, group in option_snapshot.groupby(
        group_columns,
        dropna=False,
        sort=True,
    ):
        (
            group_entry_date,
            score,
            ndx_close,
            expiration,
            dte,
            underlying,
            candidate_type,
            target_short,
            selected_short,
            selected_long,
        ) = group_key

        short = group.loc[group["leg_name"] == "short_put"].iloc[0]
        long = group.loc[group["leg_name"] == "long_put"].iloc[0]

        complete = bool(
            short["quote_available"]
            and long["quote_available"]
            and future_row["quote_available"]
        )

        if complete:
            natural_credit = float(short["bid"] - long["ask"])
            midpoint_credit = float(short["mid"] - long["mid"])
            optimistic_credit = float(short["ask"] - long["bid"])
            spread_bbo_width = optimistic_credit - natural_credit
            short_pct_below_future = (
                (future_mid - selected_short) / future_mid * 100.0
            )
            target_pct_below_future = (
                (future_mid - target_short) / future_mid * 100.0
            )
        else:
            natural_credit = np.nan
            midpoint_credit = np.nan
            optimistic_credit = np.nan
            spread_bbo_width = np.nan
            short_pct_below_future = np.nan
            target_pct_below_future = np.nan

        spread_rows.append(
            {
                "entry_date": group_entry_date,
                "dataset_condition": condition,
                "evidence_status": (
                    "exploratory"
                    if condition.lower() == "degraded"
                    else "clean_pilot"
                ),
                "score": int(score),
                "ndx_close": float(ndx_close),
                "underlying": underlying,
                "underlying_mid": future_mid,
                "ndx_to_future_basis_points": future_mid - float(ndx_close),
                "ndx_to_future_basis_pct": (
                    (future_mid / float(ndx_close) - 1.0) * 100.0
                    if complete
                    else np.nan
                ),
                "expiration": expiration,
                "dte_calendar": int(dte),
                "candidate_type": candidate_type,
                "target_short_strike": float(target_short),
                "selected_short_strike": float(selected_short),
                "selected_long_strike": float(selected_long),
                "spread_width_points": float(selected_short - selected_long),
                "target_pct_below_future": target_pct_below_future,
                "selected_short_pct_below_future": short_pct_below_future,
                "complete_quotes": complete,
                "short_symbol": short["raw_symbol"],
                "long_symbol": long["raw_symbol"],
                "short_quote_age_seconds": short["quote_age_seconds"],
                "long_quote_age_seconds": long["quote_age_seconds"],
                "short_bid": short["bid"],
                "short_ask": short["ask"],
                "long_bid": long["bid"],
                "long_ask": long["ask"],
                "natural_credit_points": natural_credit,
                "mid_credit_points": midpoint_credit,
                "optimistic_credit_points": optimistic_credit,
                "spread_bbo_width_points": spread_bbo_width,
                "natural_credit_floor_pass": (
                    bool(natural_credit >= 8.25)
                    if pd.notna(natural_credit)
                    else False
                ),
                "mid_credit_floor_pass": (
                    bool(midpoint_credit >= 8.25)
                    if pd.notna(midpoint_credit)
                    else False
                ),
            }
        )

    spread_frame = pd.DataFrame(spread_rows).sort_values(
        ["expiration", "candidate_type"]
    )

    comparison_rows: list[dict[str, Any]] = []

    for expiration, group in spread_frame.groupby("expiration", sort=True):
        standard = group.loc[
            group["candidate_type"] == "standard"
        ].iloc[0]

        defensive = group.loc[
            group["candidate_type"] == "defensive"
        ].iloc[0]

        comparison_rows.append(
            {
                "entry_date": entry_date,
                "dataset_condition": condition,
                "evidence_status": standard["evidence_status"],
                "score": int(standard["score"]),
                "ndx_close": standard["ndx_close"],
                "underlying": standard["underlying"],
                "underlying_mid": standard["underlying_mid"],
                "ndx_to_future_basis_pct": standard[
                    "ndx_to_future_basis_pct"
                ],
                "expiration": expiration,
                "dte_calendar": int(standard["dte_calendar"]),
                "standard_short_strike": standard[
                    "selected_short_strike"
                ],
                "defensive_short_strike": defensive[
                    "selected_short_strike"
                ],
                "short_strike_separation_points": (
                    standard["selected_short_strike"]
                    - defensive["selected_short_strike"]
                ),
                "short_strike_separation_pct_of_future": (
                    (
                        standard["selected_short_strike"]
                        - defensive["selected_short_strike"]
                    )
                    / standard["underlying_mid"]
                    * 100.0
                    if pd.notna(standard["underlying_mid"])
                    else np.nan
                ),
                "standard_short_pct_below_future": standard[
                    "selected_short_pct_below_future"
                ],
                "defensive_short_pct_below_future": defensive[
                    "selected_short_pct_below_future"
                ],
                "standard_natural_credit": standard[
                    "natural_credit_points"
                ],
                "defensive_natural_credit": defensive[
                    "natural_credit_points"
                ],
                "natural_credit_haircut_points": (
                    standard["natural_credit_points"]
                    - defensive["natural_credit_points"]
                ),
                "standard_mid_credit": standard["mid_credit_points"],
                "defensive_mid_credit": defensive["mid_credit_points"],
                "mid_credit_haircut_points": (
                    standard["mid_credit_points"]
                    - defensive["mid_credit_points"]
                ),
                "standard_mid_floor_pass": standard[
                    "mid_credit_floor_pass"
                ],
                "defensive_mid_floor_pass": defensive[
                    "mid_credit_floor_pass"
                ],
                "mid_haircut_within_2_91_proxy_tolerance": (
                    bool(
                        standard["mid_credit_points"]
                        - defensive["mid_credit_points"]
                        <= 2.91
                    )
                    if pd.notna(standard["mid_credit_points"])
                    and pd.notna(defensive["mid_credit_points"])
                    else False
                ),
            }
        )

    return (
        snapshot_output,
        spread_frame,
        pd.DataFrame(comparison_rows),
    )


def build_parser() -> argparse.ArgumentParser:
    """Build command-line parser."""

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--download",
        action="store_true",
        help="Authorize missing quote downloads.",
    )

    parser.add_argument(
        "--max-cost-usd",
        type=float,
        default=0.0,
        help="Maximum combined estimated cost for missing quote files.",
    )

    parser.add_argument(
        "--force-redownload",
        action="store_true",
        help="Redownload even when a raw quote DBN file exists.",
    )

    return parser


def main() -> int:
    """Download or reuse quotes and analyze both pilot dates."""

    import databento as db

    args = build_parser().parse_args()

    print(
        "MNQ PCS Regime Research — Multi-Date BBO Quote Download"
    )
    print("=" * 64)

    config = load_config()
    manifest = load_manifest()
    dataset = str(config["dataset"])
    client = db.Historical(api_key())

    requests: list[dict[str, Any]] = []

    print("\nQuote files and current estimates")
    print("-" * 126)

    for entry_date, date_rows in manifest.groupby("entry_date", sort=True):
        target_local, start_utc, end_utc = quote_window(
            entry_date,
            config,
        )

        option_symbols = sorted(
            date_rows["raw_symbol"].drop_duplicates().tolist()
        )

        underlying_symbols = sorted(
            date_rows["underlying"].drop_duplicates().tolist()
        )

        symbols = sorted(set(option_symbols + underlying_symbols))
        output_path = raw_quote_path(dataset, entry_date)

        reuse = output_path.is_file() and not args.force_redownload

        if reuse:
            cost, records = 0.0, 0
        else:
            cost, records = estimate_request(
                client,
                dataset,
                symbols,
                start_utc,
                end_utc,
            )

        condition = condition_for_date(
            client,
            dataset,
            entry_date,
        )

        requests.append(
            {
                "entry_date": entry_date,
                "date_rows": date_rows,
                "target_local": target_local,
                "start_utc": start_utc,
                "end_utc": end_utc,
                "symbols": symbols,
                "output_path": output_path,
                "reuse": reuse,
                "condition": condition,
                "estimated_cost_usd": cost,
                "estimated_record_count": records,
                "downloaded_during_this_run": False,
            }
        )

        print(
            f"{entry_date} | "
            f"Condition={condition:<9} | "
            f"{'reuse local file' if reuse else 'download required':<18} | "
            f"Symbols={len(symbols):>2} | "
            f"Cost=${cost:,.6f} | "
            f"Records={records:,}"
        )

    missing = [row for row in requests if not row["reuse"]]
    total_cost = sum(row["estimated_cost_usd"] for row in missing)

    print(f"\nTotal estimated cost for missing files: ${total_cost:,.6f}")

    if missing and not args.download:
        raise RuntimeError(
            "Missing quote files require --download and --max-cost-usd."
        )

    if missing and args.max_cost_usd <= 0:
        raise RuntimeError("A positive --max-cost-usd is required.")

    if total_cost > args.max_cost_usd:
        raise RuntimeError(
            f"Blocked: ${total_cost:,.6f} exceeds "
            f"${args.max_cost_usd:,.6f}."
        )

    for request in missing:
        print(
            f"\nDownloading {request['entry_date']}; "
            f"estimated cost ${request['estimated_cost_usd']:,.6f}."
        )

        download_request(
            client,
            dataset,
            request["symbols"],
            request["start_utc"],
            request["end_utc"],
            request["output_path"],
        )

        request["downloaded_during_this_run"] = True

    snapshots: list[pd.DataFrame] = []
    spreads: list[pd.DataFrame] = []
    comparisons: list[pd.DataFrame] = []

    print("\nQuote availability")
    print("-" * 118)

    for request in requests:
        entry_date = request["entry_date"]
        quotes = normalized_quotes(db, request["output_path"])
        target_utc = request["target_local"].tz_convert("UTC")

        snapshot = select_snapshot(
            quotes,
            request["symbols"],
            target_utc,
            maximum_age_minutes=float(
                config.get("maximum_quote_age_minutes", 5.0)
            ),
        )

        available = int(snapshot["quote_available"].sum())

        print(
            f"{entry_date} | "
            f"Condition={request['condition']:<9} | "
            f"Valid symbols={available}/{len(snapshot)}"
        )

        for row in snapshot.itertuples(index=False):
            if row.quote_available:
                print(
                    f"  {row.raw_symbol:<18} | "
                    f"{row.quote_timestamp_utc} | "
                    f"Age={row.quote_age_seconds:>6.1f}s | "
                    f"Bid={row.bid:>8.2f} | Ask={row.ask:>8.2f}"
                )
            else:
                print(
                    f"  {row.raw_symbol:<18} | "
                    f"NO VALID QUOTE | Records={row.records_received}"
                )

        snapshot_output, spread_frame, comparison_frame = analyze_date(
            entry_date,
            request["date_rows"],
            snapshot,
            request["condition"],
            request["target_local"],
        )

        snapshots.append(snapshot_output)
        spreads.append(spread_frame)
        comparisons.append(comparison_frame)

    snapshot_all = pd.concat(snapshots, ignore_index=True)
    spread_all = pd.concat(spreads, ignore_index=True)
    comparison_all = pd.concat(comparisons, ignore_index=True)

    print("\nSpread credits")
    print("-" * 174)

    for row in spread_all.itertuples(index=False):
        natural_status = "PASS" if row.natural_credit_floor_pass else "FAIL"
        mid_status = "PASS" if row.mid_credit_floor_pass else "FAIL"

        print(
            f"{row.entry_date} | "
            f"{row.evidence_status:<11} | "
            f"Exp={row.expiration} | "
            f"{row.candidate_type:<9} | "
            f"Spread={row.selected_short_strike:,.0f}/"
            f"{row.selected_long_strike:,.0f} | "
            f"Natural={row.natural_credit_points:>7.2f} "
            f"({natural_status}) | "
            f"Mid={row.mid_credit_points:>6.2f} "
            f"({mid_status}) | "
            f"Optimistic={row.optimistic_credit_points:>7.2f}"
        )

    print("\nDate comparison")
    print("-" * 180)

    for row in comparison_all.itertuples(index=False):
        tolerance_status = (
            "YES"
            if row.mid_haircut_within_2_91_proxy_tolerance
            else "NO"
        )

        print(
            f"{row.entry_date} | "
            f"{row.evidence_status:<11} | "
            f"MNQ mid={row.underlying_mid:>10.2f} | "
            f"Short separation="
            f"{row.short_strike_separation_points:>6.0f} pts "
            f"({row.short_strike_separation_pct_of_future:>5.2f}%) | "
            f"Std mid={row.standard_mid_credit:>6.2f} | "
            f"Def mid={row.defensive_mid_credit:>6.2f} | "
            f"Haircut={row.mid_credit_haircut_points:>6.2f} | "
            f"Within 2.91={tolerance_status}"
        )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    snapshot_all.to_csv(SNAPSHOT_CSV, index=False)
    spread_all.to_csv(SPREAD_CSV, index=False)
    comparison_all.to_csv(COMPARISON_CSV, index=False)

    audit_payload = {
        "dataset": dataset,
        "schema": "bbo-1s",
        "estimated_missing_cost_usd": total_cost,
        "dates": [
            {
                "entry_date": row["entry_date"],
                "condition": row["condition"],
                "symbols": row["symbols"],
                "raw_path": str(row["output_path"]),
                "file_reused": row["reuse"],
                "downloaded_during_this_run": row[
                    "downloaded_during_this_run"
                ],
                "estimated_cost_usd": row["estimated_cost_usd"],
                "estimated_record_count": row[
                    "estimated_record_count"
                ],
            }
            for row in requests
        ],
    }

    AUDIT_JSON.write_text(
        json.dumps(audit_payload, indent=2) + "\n",
        encoding="utf-8",
    )

    summary_lines = [
        "MNQ PCS REGIME RESEARCH — MULTI-DATE QUOTE PILOT",
        "=" * 62,
        "",
    ]

    for row in comparison_all.itertuples(index=False):
        summary_lines.append(
            f"{row.entry_date} | "
            f"{row.evidence_status} | "
            f"Standard mid={row.standard_mid_credit} | "
            f"Defensive mid={row.defensive_mid_credit} | "
            f"Haircut={row.mid_credit_haircut_points} | "
            f"Separation pct={row.short_strike_separation_pct_of_future}"
        )

    SUMMARY_TXT.write_text(
        "\n".join(summary_lines) + "\n",
        encoding="utf-8",
    )

    print("\nOutputs")
    print("-" * 104)
    print(f"Quote snapshot: {SNAPSHOT_CSV}")
    print(f"Spread credits: {SPREAD_CSV}")
    print(f"Date comparison: {COMPARISON_CSV}")
    print(f"Download audit: {AUDIT_JSON}")
    print(f"Analysis summary: {SUMMARY_TXT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

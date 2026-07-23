"""Download and inventory pure-MNQ definition snapshots for 13 pilot Fridays.

Run from the project root:

    python scripts/download_vix_bucket_definitions.py \
        --download \
        --max-cost-usd 5.60

This script:

- reads the reproducible 13-date selection;
- re-estimates only missing definition files;
- enforces a combined hard cost ceiling;
- reuses existing raw DBN files;
- filters to MNQ put options with 26–35 calendar DTE;
- inventories every eligible expiration and listed strike;
- records whether any exact 150-point pair exists; and
- does not yet select a short strike.

The short strike must be based on the actual MNQ midpoint at 3:45 p.m. That
underlying quote will be retrieved in the next, much cheaper checkpoint.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import pandas as pd
from dotenv import load_dotenv


ROOT = Path(__file__).resolve().parents[1]
SELECTION_PATH = (
    ROOT
    / "results"
    / "vix_bucket_credit_pilot"
    / "selected_pilot_dates.csv"
)
RAW_DIR = (
    ROOT
    / "data"
    / "raw"
    / "databento"
    / "definitions"
)
RESULTS_DIR = (
    ROOT
    / "results"
    / "vix_bucket_credit_pilot"
)

CANDIDATES_CSV = RESULTS_DIR / "mnq_option_definition_candidates.csv"
INVENTORY_CSV = RESULTS_DIR / "expiration_inventory.csv"
AUDIT_JSON = RESULTS_DIR / "definition_download_audit.json"
SUMMARY_TXT = RESULTS_DIR / "definition_inventory_summary.txt"


def api_key() -> str:
    """Load the Databento key without displaying it."""

    load_dotenv(ROOT / ".env", override=True)

    value = os.getenv("DATABENTO_API_KEY", "").strip()

    if not value.startswith("db-"):
        raise RuntimeError(
            "A valid DATABENTO_API_KEY is required in the project .env file."
        )

    return value


def load_selection() -> pd.DataFrame:
    """Load and validate the 13 selected Fridays."""

    if not SELECTION_PATH.is_file():
        raise FileNotFoundError(
            f"Missing selected-date file: {SELECTION_PATH}"
        )

    frame = pd.read_csv(SELECTION_PATH)

    required = {
        "bucket",
        "entry_date",
        "vix_close",
        "ladder_distance_pct",
        "dataset_condition",
    }

    missing = sorted(required.difference(frame.columns))

    if missing:
        raise ValueError(
            "Selected-date file is missing fields: "
            + ", ".join(missing)
        )

    frame["entry_date"] = (
        frame["entry_date"].fillna("").astype(str).str.strip()
    )

    frame["vix_close"] = pd.to_numeric(
        frame["vix_close"],
        errors="coerce",
    )

    frame["ladder_distance_pct"] = pd.to_numeric(
        frame["ladder_distance_pct"],
        errors="coerce",
    )

    if frame.empty:
        raise ValueError("The selected-date file is empty.")

    if frame["entry_date"].duplicated().any():
        raise ValueError("The selected-date file contains duplicate dates.")

    return frame.sort_values("entry_date").reset_index(drop=True)


def raw_path(entry_date: str) -> Path:
    """Return the ignored definition DBN path."""

    return RAW_DIR / (
        f"GLBX_MDP3_{entry_date}_definitions.dbn.zst"
    )


def date_window(entry_date: str) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Return one UTC calendar-day request window."""

    start = pd.Timestamp(entry_date).normalize().tz_localize("UTC")

    return start, start + pd.Timedelta(days=1)


def estimate_request(
    client: Any,
    entry_date: str,
) -> tuple[float, int]:
    """Estimate one ALL_SYMBOLS definition request."""

    start, end = date_window(entry_date)

    kwargs = dict(
        dataset="GLBX.MDP3",
        symbols="ALL_SYMBOLS",
        stype_in="raw_symbol",
        schema="definition",
        start=start.isoformat(),
        end=end.isoformat(),
    )

    return (
        float(client.metadata.get_cost(**kwargs)),
        int(client.metadata.get_record_count(**kwargs)),
    )


def download_request(
    client: Any,
    entry_date: str,
    output_path: Path,
) -> None:
    """Download and save one definition snapshot."""

    start, end = date_window(entry_date)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    store = client.timeseries.get_range(
        dataset="GLBX.MDP3",
        symbols="ALL_SYMBOLS",
        stype_in="raw_symbol",
        schema="definition",
        start=start.isoformat(),
        end=end.isoformat(),
    )

    store.to_file(output_path)


def clean_text(series: pd.Series) -> pd.Series:
    """Normalize text values."""

    return series.fillna("").astype(str).str.strip()


def load_candidates(
    db: Any,
    raw_file: Path,
    selection_row: pd.Series,
) -> pd.DataFrame:
    """Load and filter one point-in-time MNQ option definition file."""

    frame = db.DBNStore.from_file(raw_file).to_df(
        price_type="float",
        pretty_ts=True,
    ).reset_index()

    required = {
        "instrument_id",
        "raw_symbol",
        "instrument_class",
        "security_type",
        "asset",
        "underlying",
        "strike_price",
        "expiration",
    }

    missing = sorted(required.difference(frame.columns))

    if missing:
        raise ValueError(
            f"{raw_file.name} is missing fields: {', '.join(missing)}"
        )

    for column in (
        "raw_symbol",
        "instrument_class",
        "security_type",
        "asset",
        "underlying",
    ):
        frame[column] = clean_text(frame[column])

    frame["instrument_class"] = frame["instrument_class"].str.upper()
    frame["asset"] = frame["asset"].str.upper()
    frame["underlying"] = frame["underlying"].str.upper()

    frame["strike_price"] = pd.to_numeric(
        frame["strike_price"],
        errors="coerce",
    )

    frame["instrument_id"] = pd.to_numeric(
        frame["instrument_id"],
        errors="coerce",
    )

    frame["expiration"] = pd.to_datetime(
        frame["expiration"],
        errors="coerce",
        utc=True,
    )

    timestamp_column = (
        "ts_recv"
        if "ts_recv" in frame.columns
        else "ts_event"
    )

    frame["definition_timestamp_utc"] = pd.to_datetime(
        frame[timestamp_column],
        errors="coerce",
        utc=True,
    )

    entry_date = pd.Timestamp(
        selection_row["entry_date"]
    ).normalize()

    expiration_date = (
        frame["expiration"]
        .dt.tz_convert("UTC")
        .dt.tz_localize(None)
        .dt.normalize()
    )

    frame["dte_calendar"] = (
        expiration_date - entry_date
    ).dt.days

    mask = (
        (frame["instrument_class"] == "P")
        & (
            frame["underlying"].str.startswith("MNQ")
            | frame["asset"].str.startswith("MQ")
        )
        & frame["strike_price"].notna()
        & frame["instrument_id"].notna()
        & frame["expiration"].notna()
        & frame["dte_calendar"].between(
            26,
            35,
            inclusive="both",
        )
    )

    output = frame.loc[
        mask,
        [
            "definition_timestamp_utc",
            "instrument_id",
            "raw_symbol",
            "security_type",
            "asset",
            "underlying",
            "strike_price",
            "expiration",
            "dte_calendar",
        ],
    ].copy()

    output = (
        output.sort_values(
            [
                "raw_symbol",
                "instrument_id",
                "definition_timestamp_utc",
            ]
        )
        .drop_duplicates(
            ["raw_symbol", "instrument_id"],
            keep="last",
        )
        .reset_index(drop=True)
    )

    if output.empty:
        raise ValueError(
            f"No MNQ puts with 26–35 DTE were found for "
            f"{selection_row['entry_date']}."
        )

    output.insert(
        0,
        "entry_date",
        str(selection_row["entry_date"]),
    )

    output.insert(
        1,
        "bucket",
        str(selection_row["bucket"]),
    )

    output.insert(
        2,
        "vix_close",
        float(selection_row["vix_close"]),
    )

    output.insert(
        3,
        "ladder_distance_pct",
        float(selection_row["ladder_distance_pct"]),
    )

    output.insert(
        4,
        "dataset_condition",
        str(selection_row["dataset_condition"]),
    )

    return output


def count_exact_pairs(
    strikes: pd.Series,
    width: float = 150.0,
    tolerance: float = 0.01,
) -> int:
    """Count listed short strikes with an exact width-lower strike."""

    unique = sorted(
        {
            float(value)
            for value in strikes.dropna()
        }
    )

    strike_set = set(unique)

    return sum(
        1
        for short in unique
        if any(
            abs((short - width) - candidate) <= tolerance
            for candidate in strike_set
        )
    )


def build_inventory(candidates: pd.DataFrame) -> pd.DataFrame:
    """Summarize every eligible expiration."""

    rows: list[dict[str, Any]] = []

    group_columns = [
        "entry_date",
        "bucket",
        "vix_close",
        "ladder_distance_pct",
        "dataset_condition",
        "expiration",
        "dte_calendar",
        "underlying",
        "asset",
    ]

    for group_key, group in candidates.groupby(
        group_columns,
        dropna=False,
        sort=True,
    ):
        (
            entry_date,
            bucket,
            vix_close,
            ladder_distance_pct,
            dataset_condition,
            expiration,
            dte_calendar,
            underlying,
            asset,
        ) = group_key

        exact_pairs = count_exact_pairs(
            group["strike_price"]
        )

        rows.append(
            {
                "entry_date": entry_date,
                "bucket": bucket,
                "vix_close": float(vix_close),
                "ladder_distance_pct": float(
                    ladder_distance_pct
                ),
                "dataset_condition": dataset_condition,
                "expiration": expiration,
                "dte_calendar": int(dte_calendar),
                "underlying": underlying,
                "asset": asset,
                "listed_put_strikes": int(
                    group["strike_price"].nunique()
                ),
                "minimum_strike": float(
                    group["strike_price"].min()
                ),
                "maximum_strike": float(
                    group["strike_price"].max()
                ),
                "exact_150_point_pairs": int(
                    exact_pairs
                ),
                "has_any_150_point_pair": bool(
                    exact_pairs > 0
                ),
                "dte_priority_distance_from_28": abs(
                    int(dte_calendar) - 28
                ),
            }
        )

    return pd.DataFrame(rows).sort_values(
        [
            "entry_date",
            "dte_priority_distance_from_28",
            "expiration",
        ]
    ).reset_index(drop=True)


def build_parser() -> argparse.ArgumentParser:
    """Build command-line parser."""

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--download",
        action="store_true",
        help="Authorize missing historical definition downloads.",
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
    """Download or reuse definitions and build the expiration inventory."""

    import databento as db

    args = build_parser().parse_args()

    print(
        "MNQ PCS Regime Research — Pure-MNQ Definition Download"
    )
    print("=" * 62)

    selection = load_selection()
    client = db.Historical(api_key())

    requests: list[dict[str, Any]] = []

    print("\nDefinition files and current estimates")
    print("-" * 116)

    for row in selection.itertuples(index=False):
        entry_date = str(row.entry_date)
        output_path = raw_path(entry_date)

        reuse = (
            output_path.is_file()
            and not args.force_redownload
        )

        if reuse:
            cost, records = 0.0, 0
        else:
            cost, records = estimate_request(
                client,
                entry_date,
            )

        requests.append(
            {
                "entry_date": entry_date,
                "output_path": output_path,
                "reuse": reuse,
                "estimated_cost_usd": cost,
                "estimated_record_count": records,
                "downloaded_during_this_run": False,
            }
        )

        print(
            f"{entry_date} | "
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
            request["entry_date"],
            request["output_path"],
        )

        request["downloaded_during_this_run"] = True

    candidate_frames: list[pd.DataFrame] = []

    print("\nMNQ option definition inventory")
    print("-" * 142)

    for selection_row in selection.to_dict(orient="records"):
        entry_date = str(selection_row["entry_date"])

        candidates = load_candidates(
            db,
            raw_path(entry_date),
            pd.Series(selection_row),
        )

        candidate_frames.append(candidates)

        date_inventory = build_inventory(candidates)

        inventory_text = "; ".join(
            (
                f"{int(row.dte_calendar)}D "
                f"{row.expiration.date()} "
                f"strikes={row.listed_put_strikes} "
                f"pairs={row.exact_150_point_pairs} "
                f"underlying={row.underlying}"
            )
            for row in date_inventory.itertuples(index=False)
        )

        print(
            f"{entry_date} | "
            f"Bucket={selection_row['bucket']:<10} | "
            f"VIX={float(selection_row['vix_close']):>6.2f} | "
            f"{inventory_text}"
        )

    candidates_all = pd.concat(
        candidate_frames,
        ignore_index=True,
    )

    inventory_all = build_inventory(
        candidates_all
    )

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    candidates_all.to_csv(
        CANDIDATES_CSV,
        index=False,
    )

    inventory_all.to_csv(
        INVENTORY_CSV,
        index=False,
    )

    audit_payload = {
        "dataset": "GLBX.MDP3",
        "schema": "definition",
        "selected_dates": len(selection),
        "estimated_missing_cost_usd": total_cost,
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

    usable_dates = int(
        inventory_all.loc[
            inventory_all["has_any_150_point_pair"]
        ]["entry_date"].nunique()
    )

    summary_lines = [
        "MNQ PCS REGIME RESEARCH — PURE-MNQ DEFINITIONS",
        "=" * 60,
        "",
        f"Selected dates: {len(selection)}",
        f"Candidate option rows: {len(candidates_all):,}",
        f"Expiration inventory rows: {len(inventory_all):,}",
        (
            "Dates with at least one exact 150-point pair: "
            f"{usable_dates}"
        ),
        "",
        "No strike target has been selected yet.",
        (
            "The next checkpoint retrieves the actual 3:45 p.m. "
            "MNQ midpoint."
        ),
    ]

    SUMMARY_TXT.write_text(
        "\n".join(summary_lines) + "\n",
        encoding="utf-8",
    )

    print("\nOutputs")
    print("-" * 100)
    print(f"Definition candidates: {CANDIDATES_CSV}")
    print(f"Expiration inventory: {INVENTORY_CSV}")
    print(f"Download audit: {AUDIT_JSON}")
    print(f"Summary: {SUMMARY_TXT}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

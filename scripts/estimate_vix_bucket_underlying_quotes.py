"""Select preferred expirations and estimate MNQ futures quote costs.

This checkpoint downloads nothing. It:

- reads the completed definition inventory;
- excludes dates with no eligible 26–35 DTE MNQ expiration;
- chooses the expiration closest to 28 calendar DTE;
- requires at least one exact 150-point pair;
- identifies the corresponding MNQ futures symbol; and
- estimates ten minutes of BBO-1s data around 3:45 p.m. New York time.

Run:
    python scripts/estimate_vix_bucket_underlying_quotes.py
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd
from dotenv import load_dotenv


ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results" / "vix_bucket_credit_pilot"

SELECTION_PATH = RESULTS_DIR / "selected_pilot_dates.csv"
INVENTORY_PATH = RESULTS_DIR / "expiration_inventory.csv"
DATE_AUDIT_PATH = RESULTS_DIR / "date_eligibility_audit.csv"

MANIFEST_CSV = RESULTS_DIR / "underlying_quote_manifest.csv"
ESTIMATE_JSON = RESULTS_DIR / "underlying_quote_cost_estimates.json"
SUMMARY_TXT = RESULTS_DIR / "underlying_quote_estimate_summary.txt"


def api_key() -> str:
    """Load the Databento key without displaying it."""

    load_dotenv(ROOT / ".env", override=True)

    value = os.getenv("DATABENTO_API_KEY", "").strip()

    if not value.startswith("db-"):
        raise RuntimeError(
            "A valid DATABENTO_API_KEY is required in the project .env file."
        )

    return value


def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load and validate prior checkpoint outputs."""

    for path in (
        SELECTION_PATH,
        INVENTORY_PATH,
        DATE_AUDIT_PATH,
    ):
        if not path.is_file():
            raise FileNotFoundError(f"Missing required file: {path}")

    selection = pd.read_csv(SELECTION_PATH)
    inventory = pd.read_csv(INVENTORY_PATH)
    audit = pd.read_csv(DATE_AUDIT_PATH)

    required_selection = {
        "entry_date",
        "bucket",
        "vix_close",
        "ladder_distance_pct",
    }

    required_inventory = {
        "entry_date",
        "bucket",
        "vix_close",
        "ladder_distance_pct",
        "expiration",
        "dte_calendar",
        "underlying",
        "exact_150_point_pairs",
        "has_any_150_point_pair",
        "dte_priority_distance_from_28",
    }

    required_audit = {
        "entry_date",
        "eligible_26_to_35_dte",
        "status",
    }

    for name, frame, required in (
        ("selection", selection, required_selection),
        ("inventory", inventory, required_inventory),
        ("date audit", audit, required_audit),
    ):
        missing = sorted(required.difference(frame.columns))

        if missing:
            raise ValueError(
                f"{name} is missing fields: {', '.join(missing)}"
            )

    for frame in (selection, inventory, audit):
        frame["entry_date"] = (
            frame["entry_date"].fillna("").astype(str).str.strip()
        )

    inventory["expiration"] = pd.to_datetime(
        inventory["expiration"],
        errors="coerce",
        utc=True,
    )

    for column in (
        "dte_calendar",
        "exact_150_point_pairs",
        "dte_priority_distance_from_28",
    ):
        inventory[column] = pd.to_numeric(
            inventory[column],
            errors="coerce",
        )

    inventory["has_any_150_point_pair"] = (
        inventory["has_any_150_point_pair"]
        .astype(str)
        .str.lower()
        .map({"true": True, "false": False})
        .fillna(False)
    )

    audit["eligible_26_to_35_dte"] = (
        audit["eligible_26_to_35_dte"]
        .astype(str)
        .str.lower()
        .map({"true": True, "false": False})
        .fillna(False)
    )

    return selection, inventory, audit


def choose_preferred_expirations(
    inventory: pd.DataFrame,
) -> pd.DataFrame:
    """Choose one comparable expiration per usable date."""

    usable = inventory.loc[
        inventory["has_any_150_point_pair"]
        & inventory["expiration"].notna()
        & inventory["underlying"].notna()
    ].copy()

    if usable.empty:
        raise RuntimeError(
            "No date has an eligible expiration with an exact 150-point pair."
        )

    usable["underlying"] = (
        usable["underlying"].fillna("").astype(str).str.strip()
    )

    usable = usable.sort_values(
        [
            "entry_date",
            "dte_priority_distance_from_28",
            "dte_calendar",
            "expiration",
            "underlying",
        ]
    )

    preferred = (
        usable.drop_duplicates("entry_date", keep="first")
        .reset_index(drop=True)
    )

    return preferred


def quote_window(
    entry_date: str,
) -> tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp]:
    """Return local target and ten-minute UTC request window."""

    date = pd.Timestamp(entry_date)
    timezone = ZoneInfo("America/New_York")

    target_local = pd.Timestamp(
        year=date.year,
        month=date.month,
        day=date.day,
        hour=15,
        minute=45,
        tz=timezone,
    )

    half_window = pd.Timedelta(minutes=5)

    return (
        target_local,
        (target_local - half_window).tz_convert("UTC"),
        (target_local + half_window).tz_convert("UTC"),
    )


def condition_for_date(
    client: Any,
    entry_date: str,
) -> str:
    """Return Databento's day-level condition."""

    rows = client.metadata.get_dataset_condition(
        dataset="GLBX.MDP3",
        start_date=entry_date,
        end_date=entry_date,
    )

    if not rows:
        return "unknown"

    row = rows[0]

    if isinstance(row, dict):
        value = row.get("condition", "unknown")
    else:
        value = getattr(row, "condition", "unknown")

    text = str(value).strip().lower()

    if "." in text:
        text = text.rsplit(".", 1)[-1]

    return text


def estimate_request(
    client: Any,
    symbol: str,
    start_utc: pd.Timestamp,
    end_utc: pd.Timestamp,
) -> tuple[float, int]:
    """Estimate one exact-symbol BBO-1s request."""

    kwargs = dict(
        dataset="GLBX.MDP3",
        symbols=[symbol],
        stype_in="raw_symbol",
        schema="bbo-1s",
        start=start_utc.isoformat(),
        end=end_utc.isoformat(),
    )

    return (
        float(client.metadata.get_cost(**kwargs)),
        int(client.metadata.get_record_count(**kwargs)),
    )


def main() -> int:
    """Select preferred expirations and estimate underlying quote costs."""

    import databento as db

    print(
        "MNQ PCS Regime Research — Underlying Quote Estimate"
    )
    print("=" * 60)

    selection, inventory, date_audit = load_inputs()
    preferred = choose_preferred_expirations(inventory)
    client = db.Historical(api_key())

    selected_dates = set(selection["entry_date"])
    usable_dates = set(preferred["entry_date"])
    excluded_dates = sorted(selected_dates.difference(usable_dates))

    print("\nExcluded dates")
    print("-" * 92)

    if excluded_dates:
        for entry_date in excluded_dates:
            audit_rows = date_audit.loc[
                date_audit["entry_date"] == entry_date
            ]

            status = (
                audit_rows.iloc[0]["status"]
                if not audit_rows.empty
                else "No eligible expiration with an exact 150-point pair"
            )

            print(f"{entry_date} | {status}")
    else:
        print("None")

    rows: list[dict[str, Any]] = []

    print("\nPreferred expiration and underlying")
    print("-" * 148)

    for row in preferred.itertuples(index=False):
        target_local, start_utc, end_utc = quote_window(
            str(row.entry_date)
        )

        condition = condition_for_date(
            client,
            str(row.entry_date),
        )

        cost, records = estimate_request(
            client,
            str(row.underlying),
            start_utc,
            end_utc,
        )

        output = {
            "entry_date": str(row.entry_date),
            "bucket": str(row.bucket),
            "vix_close": float(row.vix_close),
            "ladder_distance_pct": float(
                row.ladder_distance_pct
            ),
            "expiration": row.expiration.isoformat(),
            "dte_calendar": int(row.dte_calendar),
            "underlying": str(row.underlying),
            "exact_150_point_pairs": int(
                row.exact_150_point_pairs
            ),
            "dataset_condition": condition,
            "target_entry_timestamp_local": (
                target_local.isoformat()
            ),
            "quote_window_start_utc": start_utc.isoformat(),
            "quote_window_end_utc": end_utc.isoformat(),
            "estimated_cost_usd": cost,
            "estimated_record_count": records,
        }

        rows.append(output)

        print(
            f"{row.entry_date} | "
            f"Bucket={row.bucket:<10} | "
            f"VIX={float(row.vix_close):>6.2f} | "
            f"Ladder={float(row.ladder_distance_pct):>4.1f}% | "
            f"Exp={row.expiration.date()} | "
            f"DTE={int(row.dte_calendar):>2} | "
            f"Underlying={row.underlying:<6} | "
            f"Pairs={int(row.exact_150_point_pairs):>3} | "
            f"Cost=${cost:,.6f} | "
            f"Records={records:,}"
        )

    manifest = pd.DataFrame(rows).sort_values(
        "entry_date"
    ).reset_index(drop=True)

    total_cost = float(
        manifest["estimated_cost_usd"].sum()
    )

    total_records = int(
        manifest["estimated_record_count"].sum()
    )

    print("\nCombined underlying quote estimate")
    print("-" * 88)
    print(f"Usable dates: {len(manifest)}")
    print(f"Excluded dates: {len(excluded_dates)}")
    print(f"Total estimated cost: ${total_cost:,.6f}")
    print(f"Total estimated records: {total_records:,}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    manifest.to_csv(
        MANIFEST_CSV,
        index=False,
    )

    payload = {
        "dataset": "GLBX.MDP3",
        "schema": "bbo-1s",
        "usable_dates": len(manifest),
        "excluded_dates": excluded_dates,
        "combined_estimated_cost_usd": total_cost,
        "combined_estimated_record_count": total_records,
        "historical_quotes_downloaded": False,
        "requests": rows,
    }

    ESTIMATE_JSON.write_text(
        json.dumps(payload, indent=2) + "\n",
        encoding="utf-8",
    )

    summary_lines = [
        "MNQ PCS REGIME RESEARCH — UNDERLYING QUOTE ESTIMATE",
        "=" * 64,
        "",
        f"Usable dates: {len(manifest)}",
        f"Excluded dates: {', '.join(excluded_dates) if excluded_dates else 'None'}",
        f"Estimated quote cost: ${total_cost:,.6f}",
        f"Estimated records: {total_records:,}",
        "",
        (
            "The next checkpoint will download only these MNQ "
            "futures quotes and calculate ladder targets."
        ),
    ]

    SUMMARY_TXT.write_text(
        "\n".join(summary_lines) + "\n",
        encoding="utf-8",
    )

    print("\nNo historical quote data was downloaded.")
    print(f"Underlying manifest: {MANIFEST_CSV}")
    print(f"Cost estimates: {ESTIMATE_JSON}")
    print(f"Summary: {SUMMARY_TXT}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

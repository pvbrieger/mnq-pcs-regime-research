"""Estimate exact option-quote costs for the executable VIX-bucket spreads.

This checkpoint downloads nothing.

Run:
    python scripts/estimate_vix_bucket_option_quotes.py

The script reads the final all-expiration option manifest and estimates ten
minutes of BBO-1s data around 3:45 p.m. New York time for the exact two option
legs selected on each executable date.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import pandas as pd
from dotenv import load_dotenv


ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results" / "vix_bucket_credit_pilot"

MANIFEST_PATH = (
    RESULTS_DIR / "option_quote_manifest_all_expirations.csv"
)
SELECTION_PATH = (
    RESULTS_DIR / "selected_ladder_spreads_all_expirations.csv"
)

ESTIMATE_JSON = RESULTS_DIR / "option_quote_cost_estimates.json"
SUMMARY_TXT = RESULTS_DIR / "option_quote_estimate_summary.txt"


def api_key() -> str:
    """Load the Databento API key without displaying it."""

    load_dotenv(ROOT / ".env", override=True)

    value = os.getenv("DATABENTO_API_KEY", "").strip()

    if not value.startswith("db-"):
        raise RuntimeError(
            "A valid DATABENTO_API_KEY is required in the project .env file."
        )

    return value


def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load and validate the exact-leg manifest and date selections."""

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
        "selected_short_strike",
        "selected_long_strike",
        "leg_name",
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

    for column in (
        "vix_close",
        "ladder_distance_pct",
        "dte_calendar",
        "underlying_mid",
        "selected_short_strike",
        "selected_long_strike",
    ):
        manifest[column] = pd.to_numeric(
            manifest[column],
            errors="coerce",
        )

    if manifest.empty:
        raise ValueError("The option quote manifest is empty.")

    return (
        manifest.sort_values(["entry_date", "leg_name"]).reset_index(
            drop=True
        ),
        selections.sort_values("entry_date").reset_index(drop=True),
    )


def normalize_condition(value: Any) -> str:
    """Normalize Databento dataset-condition text."""

    text = str(value or "unknown").strip().lower()

    if "." in text:
        text = text.rsplit(".", 1)[-1]

    return text


def condition_for_date(
    client: Any,
    entry_date: str,
) -> str:
    """Return Databento's day-level dataset condition."""

    rows = client.metadata.get_dataset_condition(
        dataset="GLBX.MDP3",
        start_date=entry_date,
        end_date=entry_date,
    )

    if not rows:
        return "unknown"

    row = rows[0]

    if isinstance(row, dict):
        return normalize_condition(row.get("condition"))

    return normalize_condition(
        getattr(row, "condition", "unknown")
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


def main() -> int:
    """Estimate exact option-leg quote costs."""

    import databento as db

    print(
        "MNQ PCS Regime Research — VIX-Bucket Option Quote Estimate"
    )
    print("=" * 70)

    manifest, selections = load_inputs()
    client = db.Historical(api_key())

    unselected = selections.loc[
        selections["selection_status"] != "Selected",
        ["entry_date", "bucket", "selection_status"],
    ]

    print("\nDates without an executable ladder spread")
    print("-" * 114)

    if unselected.empty:
        print("None")
    else:
        for row in unselected.itertuples(index=False):
            print(
                f"{row.entry_date} | "
                f"Bucket={row.bucket:<10} | "
                f"{row.selection_status}"
            )

    results: list[dict[str, Any]] = []

    print("\nExact option-leg quote estimates")
    print("-" * 180)

    for entry_date, group in manifest.groupby(
        "entry_date",
        sort=True,
    ):
        symbols = sorted(
            group["raw_symbol"].drop_duplicates().tolist()
        )

        if len(symbols) != 2:
            raise ValueError(
                f"{entry_date} has {len(symbols)} distinct option symbols; "
                "expected exactly two."
            )

        target_utc = group[
            "target_entry_timestamp_utc"
        ].drop_duplicates()

        if len(target_utc) != 1:
            raise ValueError(
                f"{entry_date} has inconsistent target timestamps."
            )

        target_utc = pd.Timestamp(target_utc.iloc[0])
        start_utc = target_utc - pd.Timedelta(minutes=5)
        end_utc = target_utc + pd.Timedelta(minutes=5)

        condition = condition_for_date(
            client,
            entry_date,
        )

        cost, records = estimate_request(
            client,
            symbols,
            start_utc,
            end_utc,
        )

        first = group.iloc[0]

        short_symbol = group.loc[
            group["leg_name"] == "short_put",
            "raw_symbol",
        ].iloc[0]

        long_symbol = group.loc[
            group["leg_name"] == "long_put",
            "raw_symbol",
        ].iloc[0]

        result = {
            "entry_date": entry_date,
            "bucket": str(first["bucket"]),
            "vix_close": float(first["vix_close"]),
            "ladder_distance_pct": float(
                first["ladder_distance_pct"]
            ),
            "expiration": first["expiration"].isoformat(),
            "dte_calendar": int(first["dte_calendar"]),
            "underlying": str(first["underlying"]),
            "underlying_mid": float(first["underlying_mid"]),
            "selected_short_strike": float(
                first["selected_short_strike"]
            ),
            "selected_long_strike": float(
                first["selected_long_strike"]
            ),
            "short_symbol": str(short_symbol),
            "long_symbol": str(long_symbol),
            "target_entry_timestamp_utc": target_utc.isoformat(),
            "quote_window_start_utc": start_utc.isoformat(),
            "quote_window_end_utc": end_utc.isoformat(),
            "dataset_condition": condition,
            "estimated_cost_usd": cost,
            "estimated_record_count": records,
        }

        results.append(result)

        print(
            f"{entry_date} | "
            f"Bucket={first['bucket']:<10} | "
            f"VIX={float(first['vix_close']):>6.2f} | "
            f"Exp={first['expiration'].date()} | "
            f"DTE={int(first['dte_calendar']):>2} | "
            f"Spread={float(first['selected_short_strike']):,.0f}/"
            f"{float(first['selected_long_strike']):,.0f} | "
            f"Condition={condition:<9} | "
            f"Cost=${cost:,.6f} | "
            f"Records={records:,}"
        )

        print(
            f"  Short={short_symbol} | Long={long_symbol}"
        )

    total_cost = sum(
        row["estimated_cost_usd"]
        for row in results
    )

    total_records = sum(
        row["estimated_record_count"]
        for row in results
    )

    print("\nCombined option quote estimate")
    print("-" * 88)
    print(f"Executable dates: {len(results)}")
    print(f"Non-executable dates: {len(unselected)}")
    print(f"Total estimated cost: ${total_cost:,.6f}")
    print(f"Total estimated records: {total_records:,}")

    payload = {
        "dataset": "GLBX.MDP3",
        "schema": "bbo-1s",
        "executable_dates": len(results),
        "non_executable_dates": unselected.to_dict(orient="records"),
        "combined_estimated_cost_usd": total_cost,
        "combined_estimated_record_count": total_records,
        "historical_option_quotes_downloaded": False,
        "requests": results,
    }

    ESTIMATE_JSON.write_text(
        json.dumps(payload, indent=2) + "\n",
        encoding="utf-8",
    )

    summary_lines = [
        "MNQ PCS REGIME RESEARCH — OPTION QUOTE ESTIMATE",
        "=" * 62,
        "",
        f"Executable dates: {len(results)}",
        f"Non-executable dates: {len(unselected)}",
        f"Estimated quote cost: ${total_cost:,.6f}",
        f"Estimated records: {total_records:,}",
        "",
        "No historical option quotes were downloaded.",
    ]

    SUMMARY_TXT.write_text(
        "\n".join(summary_lines) + "\n",
        encoding="utf-8",
    )

    print("\nNo historical option quote data was downloaded.")
    print(f"Cost estimates: {ESTIMATE_JSON}")
    print(f"Summary: {SUMMARY_TXT}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

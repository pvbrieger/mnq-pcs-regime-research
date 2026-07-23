"""Finish the pure-MNQ definition inventory without new downloads.

This script reads the 13 raw definition DBN files already downloaded. A date
with no MNQ put expiration in the required 26–35 calendar-DTE window is
recorded as ineligible rather than causing the full inventory to stop.

Run:
    python scripts/inventory_vix_bucket_definitions.py
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd


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
DATE_AUDIT_CSV = RESULTS_DIR / "date_eligibility_audit.csv"
SUMMARY_TXT = RESULTS_DIR / "definition_inventory_summary.txt"


def load_selection() -> pd.DataFrame:
    """Load and validate the selected-date file."""

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
        frame["entry_date"]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    return frame.sort_values("entry_date").reset_index(drop=True)


def raw_path(entry_date: str) -> Path:
    """Return the raw definition DBN path."""

    return RAW_DIR / (
        f"GLBX_MDP3_{entry_date}_definitions.dbn.zst"
    )


def clean_text(series: pd.Series) -> pd.Series:
    """Normalize string fields."""

    return series.fillna("").astype(str).str.strip()


def load_date_candidates(
    db: Any,
    file_path: Path,
    selection_row: pd.Series,
) -> pd.DataFrame:
    """Load and filter one point-in-time definition snapshot."""

    if not file_path.is_file():
        raise FileNotFoundError(
            f"Missing raw definition file: {file_path}"
        )

    frame = db.DBNStore.from_file(file_path).to_df(
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
            f"{file_path.name} is missing fields: "
            + ", ".join(missing)
        )

    for column in (
        "raw_symbol",
        "instrument_class",
        "security_type",
        "asset",
        "underlying",
    ):
        frame[column] = clean_text(frame[column])

    frame["instrument_class"] = (
        frame["instrument_class"].str.upper()
    )

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
        return output

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
    """Count listed short strikes with an exact 150-point-lower strike."""

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
            abs((short - width) - candidate)
            <= tolerance
            for candidate in strike_set
        )
    )


def build_inventory(candidates: pd.DataFrame) -> pd.DataFrame:
    """Summarize each eligible expiration."""

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


def main() -> int:
    """Inventory all downloaded dates and record ineligible dates."""

    import databento as db

    print(
        "MNQ PCS Regime Research — Complete Definition Inventory"
    )
    print("=" * 62)

    selection = load_selection()

    candidate_frames: list[pd.DataFrame] = []
    date_audit_rows: list[dict[str, Any]] = []

    print("\nMNQ option definition inventory")
    print("-" * 150)

    for selection_row in selection.to_dict(orient="records"):
        entry_date = str(selection_row["entry_date"])

        candidates = load_date_candidates(
            db,
            raw_path(entry_date),
            pd.Series(selection_row),
        )

        if candidates.empty:
            date_audit_rows.append(
                {
                    "entry_date": entry_date,
                    "bucket": selection_row["bucket"],
                    "vix_close": float(
                        selection_row["vix_close"]
                    ),
                    "ladder_distance_pct": float(
                        selection_row[
                            "ladder_distance_pct"
                        ]
                    ),
                    "eligible_26_to_35_dte": False,
                    "candidate_rows": 0,
                    "expiration_count": 0,
                    "expirations_with_150_point_pair": 0,
                    "status": (
                        "No MNQ put expiration in 26–35 calendar DTE"
                    ),
                }
            )

            print(
                f"{entry_date} | "
                f"Bucket={selection_row['bucket']:<10} | "
                f"VIX={float(selection_row['vix_close']):>6.2f} | "
                "INELIGIBLE — no MNQ puts with 26–35 DTE"
            )

            continue

        candidate_frames.append(candidates)

        date_inventory = build_inventory(candidates)

        usable_expirations = int(
            date_inventory[
                "has_any_150_point_pair"
            ].sum()
        )

        date_audit_rows.append(
            {
                "entry_date": entry_date,
                "bucket": selection_row["bucket"],
                "vix_close": float(
                    selection_row["vix_close"]
                ),
                "ladder_distance_pct": float(
                    selection_row[
                        "ladder_distance_pct"
                    ]
                ),
                "eligible_26_to_35_dte": True,
                "candidate_rows": len(candidates),
                "expiration_count": len(date_inventory),
                "expirations_with_150_point_pair": (
                    usable_expirations
                ),
                "status": (
                    "Eligible"
                    if usable_expirations > 0
                    else "No exact 150-point pair"
                ),
            }
        )

        inventory_text = "; ".join(
            (
                f"{int(row.dte_calendar)}D "
                f"{row.expiration.date()} "
                f"strikes={row.listed_put_strikes} "
                f"pairs={row.exact_150_point_pairs} "
                f"underlying={row.underlying}"
            )
            for row in date_inventory.itertuples(
                index=False
            )
        )

        print(
            f"{entry_date} | "
            f"Bucket={selection_row['bucket']:<10} | "
            f"VIX={float(selection_row['vix_close']):>6.2f} | "
            f"{inventory_text}"
        )

    if not candidate_frames:
        raise RuntimeError(
            "No selected date produced an eligible MNQ option inventory."
        )

    candidates_all = pd.concat(
        candidate_frames,
        ignore_index=True,
    )

    inventory_all = build_inventory(
        candidates_all
    )

    date_audit = pd.DataFrame(
        date_audit_rows
    ).sort_values(
        [
            "bucket",
            "entry_date",
        ]
    ).reset_index(drop=True)

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

    date_audit.to_csv(
        DATE_AUDIT_CSV,
        index=False,
    )

    eligible_dates = int(
        date_audit[
            "eligible_26_to_35_dte"
        ].sum()
    )

    usable_dates = int(
        (
            date_audit[
                "expirations_with_150_point_pair"
            ]
            > 0
        ).sum()
    )

    ineligible_dates = date_audit.loc[
        ~date_audit[
            "eligible_26_to_35_dte"
        ],
        "entry_date",
    ].tolist()

    summary_lines = [
        "MNQ PCS REGIME RESEARCH — COMPLETE DEFINITION INVENTORY",
        "=" * 66,
        "",
        f"Selected dates: {len(selection)}",
        f"Dates with 26–35 DTE MNQ puts: {eligible_dates}",
        (
            "Dates with at least one exact 150-point pair: "
            f"{usable_dates}"
        ),
        (
            "Dates requiring replacement: "
            + (
                ", ".join(ineligible_dates)
                if ineligible_dates
                else "None"
            )
        ),
        "",
        "No new historical data was downloaded.",
    ]

    SUMMARY_TXT.write_text(
        "\n".join(summary_lines) + "\n",
        encoding="utf-8",
    )

    print("\nOutputs")
    print("-" * 104)

    print(
        f"Definition candidates: {CANDIDATES_CSV}"
    )

    print(
        f"Expiration inventory: {INVENTORY_CSV}"
    )

    print(
        f"Date eligibility audit: {DATE_AUDIT_CSV}"
    )

    print(
        f"Summary: {SUMMARY_TXT}"
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

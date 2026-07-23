"""Build and estimate the closer-strike MNQ option-quote test.

This checkpoint downloads nothing.

For every pilot Friday with a valid 3:45 p.m. MNQ futures midpoint, the script
tests four target distances:

- the current VIX-ladder distance;
- 1 percentage point closer to MNQ;
- 2 percentage points closer; and
- 3 percentage points closer.

A single expiration is chosen per date to keep the within-date comparison clean.
The chosen expiration:

1. supports the largest number of requested distances;
2. is closest to 28 calendar DTE;
3. uses the lower DTE as the next tie-breaker; and
4. uses the earlier expiration as the final tie-breaker.

For each executable distance, the highest exact 150-point short strike at or
below the target is selected. The script joins the historical NDX loss surface
and estimates ten minutes of exact-leg BBO-1s data around 3:45 p.m.

Run:
    python scripts/estimate_vix_closer_strike_quotes.py
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from dotenv import load_dotenv


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "config" / "vix_closer_strike_quote_test.yaml"

PILOT_RESULTS_DIR = ROOT / "results" / "vix_bucket_credit_pilot"
RISK_RESULTS_DIR = ROOT / "results" / "vix_distance_risk_surface"
RESULTS_DIR = ROOT / "results" / "vix_closer_strike_quote_test"

SNAPSHOT_PATH = PILOT_RESULTS_DIR / "underlying_quote_snapshot.csv"
CANDIDATES_PATH = (
    PILOT_RESULTS_DIR / "mnq_option_definition_candidates.csv"
)
RISK_SURFACE_PATH = RISK_RESULTS_DIR / "risk_distance_surface.csv"

FEASIBILITY_CSV = RESULTS_DIR / "distance_expiration_feasibility.csv"
SELECTION_CSV = RESULTS_DIR / "selected_distance_spreads.csv"
OPTION_MANIFEST_CSV = RESULTS_DIR / "option_quote_manifest.csv"
ESTIMATE_JSON = RESULTS_DIR / "option_quote_cost_estimates.json"
SUMMARY_TXT = RESULTS_DIR / "selection_and_cost_summary.txt"


def load_config() -> dict[str, Any]:
    """Load test configuration."""

    if not CONFIG_PATH.is_file():
        raise FileNotFoundError(f"Missing configuration: {CONFIG_PATH}")

    with CONFIG_PATH.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def api_key() -> str:
    """Load the Databento key without displaying it."""

    load_dotenv(ROOT / ".env", override=True)

    value = os.getenv("DATABENTO_API_KEY", "").strip()

    if not value.startswith("db-"):
        raise RuntimeError(
            "A valid DATABENTO_API_KEY is required in the project .env file."
        )

    return value


def bool_series(series: pd.Series) -> pd.Series:
    """Normalize common boolean representations."""

    return (
        series.astype(str)
        .str.strip()
        .str.lower()
        .map(
            {
                "true": True,
                "false": False,
                "1": True,
                "0": False,
            }
        )
        .fillna(False)
    )


def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load and validate prior checkpoint outputs."""

    for path in (
        SNAPSHOT_PATH,
        CANDIDATES_PATH,
        RISK_SURFACE_PATH,
    ):
        if not path.is_file():
            raise FileNotFoundError(f"Missing required file: {path}")

    snapshot = pd.read_csv(SNAPSHOT_PATH)
    candidates = pd.read_csv(CANDIDATES_PATH)
    risk = pd.read_csv(RISK_SURFACE_PATH)

    snapshot_required = {
        "entry_date",
        "bucket",
        "vix_close",
        "ladder_distance_pct",
        "underlying",
        "quote_available",
        "mid",
        "target_entry_timestamp_utc",
    }

    candidates_required = {
        "entry_date",
        "expiration",
        "dte_calendar",
        "underlying",
        "strike_price",
        "raw_symbol",
    }

    risk_required = {
        "vix_bucket",
        "distance_pct",
        "sample_count",
        "short_breach_rate",
        "max_loss_rate",
        "average_intrinsic_loss_points",
        "average_intrinsic_loss_ci_low",
        "average_intrinsic_loss_ci_high",
    }

    for name, frame, required in (
        ("underlying snapshot", snapshot, snapshot_required),
        ("definition candidates", candidates, candidates_required),
        ("risk surface", risk, risk_required),
    ):
        missing = sorted(required.difference(frame.columns))

        if missing:
            raise ValueError(
                f"{name} is missing fields: {', '.join(missing)}"
            )

    for frame in (snapshot, candidates):
        frame["entry_date"] = (
            frame["entry_date"].fillna("").astype(str).str.strip()
        )

        frame["underlying"] = (
            frame["underlying"].fillna("").astype(str).str.strip()
        )

    snapshot["quote_available"] = bool_series(
        snapshot["quote_available"]
    )

    for column in (
        "vix_close",
        "ladder_distance_pct",
        "mid",
    ):
        snapshot[column] = pd.to_numeric(
            snapshot[column],
            errors="coerce",
        )

    snapshot["target_entry_timestamp_utc"] = pd.to_datetime(
        snapshot["target_entry_timestamp_utc"],
        errors="coerce",
        utc=True,
    )

    candidates["expiration"] = pd.to_datetime(
        candidates["expiration"],
        errors="coerce",
        utc=True,
    )

    candidates["dte_calendar"] = pd.to_numeric(
        candidates["dte_calendar"],
        errors="coerce",
    )

    candidates["strike_price"] = pd.to_numeric(
        candidates["strike_price"],
        errors="coerce",
    )

    candidates["raw_symbol"] = (
        candidates["raw_symbol"].fillna("").astype(str).str.strip()
    )

    risk["distance_pct"] = pd.to_numeric(
        risk["distance_pct"],
        errors="coerce",
    )

    numeric_risk_columns = (
        "sample_count",
        "short_breach_rate",
        "max_loss_rate",
        "average_intrinsic_loss_points",
        "average_intrinsic_loss_ci_low",
        "average_intrinsic_loss_ci_high",
    )

    for column in numeric_risk_columns:
        risk[column] = pd.to_numeric(
            risk[column],
            errors="coerce",
        )

    snapshot = snapshot.loc[
        snapshot["quote_available"]
        & snapshot["mid"].notna()
        & snapshot["target_entry_timestamp_utc"].notna()
    ].copy()

    if snapshot.empty:
        raise RuntimeError("No valid MNQ futures snapshots are available.")

    return (
        snapshot.sort_values("entry_date").reset_index(drop=True),
        candidates,
        risk,
    )


def requested_distances(
    bucket: str,
    config: dict[str, Any],
) -> list[tuple[str, float, float]]:
    """Return labels, offsets, and target distances for one bucket."""

    ladder_map = {
        str(key): float(value)
        for key, value in config["current_ladder_distances_pct"].items()
    }

    if bucket not in ladder_map:
        raise KeyError(f"No current ladder distance configured for {bucket}.")

    current = ladder_map[bucket]

    output: list[tuple[str, float, float]] = []

    for offset in config["closer_offsets_pct"]:
        offset_value = float(offset)
        distance = round(current - offset_value, 10)

        label = (
            "current"
            if np.isclose(offset_value, 0.0)
            else f"{offset_value:g}pct_closer"
        )

        output.append((label, offset_value, distance))

    return output


def exact_pairs(
    group: pd.DataFrame,
    spread_width: float,
    tolerance: float,
) -> pd.DataFrame:
    """Return exact-width put-spread pairs in one expiration."""

    strike_map = (
        group.dropna(subset=["strike_price"])
        .sort_values(["strike_price", "raw_symbol"])
        .drop_duplicates("strike_price", keep="first")
        .set_index("strike_price")["raw_symbol"]
        .to_dict()
    )

    strikes = sorted(float(value) for value in strike_map)
    rows: list[dict[str, Any]] = []

    for short_strike in strikes:
        desired_long = short_strike - spread_width

        matches = [
            strike
            for strike in strikes
            if abs(strike - desired_long) <= tolerance
        ]

        if not matches:
            continue

        long_strike = matches[0]

        rows.append(
            {
                "short_strike": short_strike,
                "long_strike": long_strike,
                "short_symbol": strike_map[short_strike],
                "long_symbol": strike_map[long_strike],
            }
        )

    if not rows:
        return pd.DataFrame(
            columns=[
                "short_strike",
                "long_strike",
                "short_symbol",
                "long_symbol",
            ]
        )

    return pd.DataFrame(rows).sort_values(
        "short_strike"
    ).reset_index(drop=True)


def risk_row_for_distance(
    risk: pd.DataFrame,
    bucket: str,
    distance: float,
) -> pd.Series:
    """Resolve one historical-risk row with floating-point tolerance."""

    rows = risk.loc[
        (risk["vix_bucket"] == bucket)
        & np.isclose(
            risk["distance_pct"],
            distance,
            atol=1e-9,
        )
    ]

    if len(rows) != 1:
        raise ValueError(
            f"Expected one risk-surface row for {bucket} at "
            f"{distance:.2f}%, found {len(rows)}."
        )

    return rows.iloc[0]


def dataset_condition(
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
    symbols: list[str],
    start_utc: pd.Timestamp,
    end_utc: pd.Timestamp,
) -> tuple[float, int]:
    """Estimate one exact-symbol BBO-1s request."""

    if not symbols:
        return 0.0, 0

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
    """Select comparison spreads and estimate exact option-quote costs."""

    import databento as db

    print(
        "MNQ PCS Regime Research — Closer-Strike Quote-Test Estimate"
    )
    print("=" * 72)

    config = load_config()
    snapshot, candidates, risk = load_inputs()
    client = db.Historical(api_key())

    spread_width = float(config["spread_width_points"])
    tolerance = float(config["strike_tolerance_points"])
    target_dte = int(config["target_dte_calendar"])
    quote_half_window = pd.Timedelta(
        minutes=float(config["quote_window_minutes"]) / 2.0
    )

    feasibility_rows: list[dict[str, Any]] = []
    selection_rows: list[dict[str, Any]] = []
    manifest_rows: list[dict[str, Any]] = []
    estimate_rows: list[dict[str, Any]] = []

    print("\nDate-level expiration selection")
    print("-" * 178)

    for snap in snapshot.itertuples(index=False):
        entry_date = str(snap.entry_date)
        bucket = str(snap.bucket)
        mnq_mid = float(snap.mid)
        underlying = str(snap.underlying)
        distances = requested_distances(bucket, config)

        date_candidates = candidates.loc[
            (candidates["entry_date"] == entry_date)
            & (candidates["underlying"] == underlying)
            & candidates["expiration"].notna()
            & candidates["dte_calendar"].between(
                int(config["minimum_dte_calendar"]),
                int(config["maximum_dte_calendar"]),
                inclusive="both",
            )
        ].copy()

        if date_candidates.empty:
            print(
                f"{entry_date} | Bucket={bucket:<10} | "
                "No eligible definitions for quoted underlying"
            )
            continue

        expiration_summaries: list[dict[str, Any]] = []
        pair_cache: dict[tuple[pd.Timestamp, int], pd.DataFrame] = {}

        for (expiration, dte), group in date_candidates.groupby(
            ["expiration", "dte_calendar"],
            sort=True,
        ):
            dte_int = int(dte)
            pairs = exact_pairs(
                group,
                spread_width=spread_width,
                tolerance=tolerance,
            )

            pair_cache[(pd.Timestamp(expiration), dte_int)] = pairs
            feasible_count = 0

            for distance_label, closer_offset, distance in distances:
                target = mnq_mid * (1.0 - distance / 100.0)

                eligible = pairs.loc[
                    pairs["short_strike"] <= target
                ].copy()

                feasible = not eligible.empty

                if feasible:
                    selected = eligible.iloc[-1]
                    selected_short = float(
                        selected["short_strike"]
                    )
                    selected_long = float(
                        selected["long_strike"]
                    )
                    short_symbol = str(
                        selected["short_symbol"]
                    )
                    long_symbol = str(
                        selected["long_symbol"]
                    )
                    feasible_count += 1
                else:
                    selected_short = np.nan
                    selected_long = np.nan
                    short_symbol = ""
                    long_symbol = ""

                feasibility_rows.append(
                    {
                        "entry_date": entry_date,
                        "bucket": bucket,
                        "vix_close": float(snap.vix_close),
                        "underlying": underlying,
                        "underlying_mid": mnq_mid,
                        "expiration": pd.Timestamp(expiration),
                        "dte_calendar": dte_int,
                        "distance_label": distance_label,
                        "closer_offset_pct": closer_offset,
                        "target_distance_pct": distance,
                        "target_short_strike": target,
                        "exact_pair_count": len(pairs),
                        "feasible": feasible,
                        "selected_short_strike": selected_short,
                        "selected_long_strike": selected_long,
                        "short_symbol": short_symbol,
                        "long_symbol": long_symbol,
                    }
                )

            expiration_summaries.append(
                {
                    "expiration": pd.Timestamp(expiration),
                    "dte_calendar": dte_int,
                    "feasible_distance_count": feasible_count,
                    "distance_from_target_dte": abs(
                        dte_int - target_dte
                    ),
                }
            )

        expiration_choice = (
            pd.DataFrame(expiration_summaries)
            .sort_values(
                [
                    "feasible_distance_count",
                    "distance_from_target_dte",
                    "dte_calendar",
                    "expiration",
                ],
                ascending=[False, True, True, True],
            )
            .iloc[0]
        )

        chosen_expiration = pd.Timestamp(
            expiration_choice["expiration"]
        )
        chosen_dte = int(
            expiration_choice["dte_calendar"]
        )
        chosen_pairs = pair_cache[
            (chosen_expiration, chosen_dte)
        ]

        chosen_count = int(
            expiration_choice["feasible_distance_count"]
        )

        print(
            f"{entry_date} | "
            f"Bucket={bucket:<10} | "
            f"VIX={float(snap.vix_close):>6.2f} | "
            f"MNQ={mnq_mid:>10.2f} | "
            f"Exp={chosen_expiration.date()} | "
            f"DTE={chosen_dte:>2} | "
            f"Executable distances={chosen_count}/{len(distances)}"
        )

        target_utc = pd.Timestamp(
            snap.target_entry_timestamp_utc
        )

        for distance_label, closer_offset, distance in distances:
            target = mnq_mid * (1.0 - distance / 100.0)

            eligible = chosen_pairs.loc[
                chosen_pairs["short_strike"] <= target
            ].copy()

            risk_row = risk_row_for_distance(
                risk,
                bucket,
                distance,
            )

            if eligible.empty:
                selection_rows.append(
                    {
                        "entry_date": entry_date,
                        "bucket": bucket,
                        "vix_close": float(snap.vix_close),
                        "underlying": underlying,
                        "underlying_mid": mnq_mid,
                        "expiration": chosen_expiration,
                        "dte_calendar": chosen_dte,
                        "distance_label": distance_label,
                        "closer_offset_pct": closer_offset,
                        "target_distance_pct": distance,
                        "target_short_strike": target,
                        "selected_short_strike": np.nan,
                        "selected_long_strike": np.nan,
                        "actual_short_distance_pct": np.nan,
                        "target_to_selected_short_slippage_points": np.nan,
                        "short_symbol": "",
                        "long_symbol": "",
                        "historical_sample_count": int(
                            risk_row["sample_count"]
                        ),
                        "historical_breach_rate": float(
                            risk_row["short_breach_rate"]
                        ),
                        "historical_max_loss_rate": float(
                            risk_row["max_loss_rate"]
                        ),
                        "historical_mean_intrinsic_loss_points": float(
                            risk_row[
                                "average_intrinsic_loss_points"
                            ]
                        ),
                        "historical_loss_ci_low": float(
                            risk_row[
                                "average_intrinsic_loss_ci_low"
                            ]
                        ),
                        "historical_loss_ci_high": float(
                            risk_row[
                                "average_intrinsic_loss_ci_high"
                            ]
                        ),
                        "selection_status": (
                            "No exact 150-point pair at or below target "
                            "in chosen expiration"
                        ),
                    }
                )
                continue

            selected = eligible.iloc[-1]
            selected_short = float(
                selected["short_strike"]
            )
            selected_long = float(
                selected["long_strike"]
            )
            short_symbol = str(
                selected["short_symbol"]
            )
            long_symbol = str(
                selected["long_symbol"]
            )

            actual_distance = (
                (mnq_mid - selected_short)
                / mnq_mid
                * 100.0
            )

            selection_rows.append(
                {
                    "entry_date": entry_date,
                    "bucket": bucket,
                    "vix_close": float(snap.vix_close),
                    "underlying": underlying,
                    "underlying_mid": mnq_mid,
                    "expiration": chosen_expiration,
                    "dte_calendar": chosen_dte,
                    "distance_label": distance_label,
                    "closer_offset_pct": closer_offset,
                    "target_distance_pct": distance,
                    "target_short_strike": target,
                    "selected_short_strike": selected_short,
                    "selected_long_strike": selected_long,
                    "actual_short_distance_pct": actual_distance,
                    "target_to_selected_short_slippage_points": (
                        target - selected_short
                    ),
                    "short_symbol": short_symbol,
                    "long_symbol": long_symbol,
                    "historical_sample_count": int(
                        risk_row["sample_count"]
                    ),
                    "historical_breach_rate": float(
                        risk_row["short_breach_rate"]
                    ),
                    "historical_max_loss_rate": float(
                        risk_row["max_loss_rate"]
                    ),
                    "historical_mean_intrinsic_loss_points": float(
                        risk_row[
                            "average_intrinsic_loss_points"
                        ]
                    ),
                    "historical_loss_ci_low": float(
                        risk_row[
                            "average_intrinsic_loss_ci_low"
                        ]
                    ),
                    "historical_loss_ci_high": float(
                        risk_row[
                            "average_intrinsic_loss_ci_high"
                        ]
                    ),
                    "selection_status": "Selected",
                }
            )

            for leg_name, strike, symbol in (
                (
                    "short_put",
                    selected_short,
                    short_symbol,
                ),
                (
                    "long_put",
                    selected_long,
                    long_symbol,
                ),
            ):
                manifest_rows.append(
                    {
                        "entry_date": entry_date,
                        "bucket": bucket,
                        "vix_close": float(snap.vix_close),
                        "underlying": underlying,
                        "underlying_mid": mnq_mid,
                        "expiration": chosen_expiration,
                        "dte_calendar": chosen_dte,
                        "distance_label": distance_label,
                        "closer_offset_pct": closer_offset,
                        "target_distance_pct": distance,
                        "target_short_strike": target,
                        "selected_short_strike": selected_short,
                        "selected_long_strike": selected_long,
                        "actual_short_distance_pct": actual_distance,
                        "leg_name": leg_name,
                        "strike_price": strike,
                        "raw_symbol": symbol,
                        "target_entry_timestamp_utc": target_utc,
                    }
                )

        date_manifest = pd.DataFrame(
            [
                row
                for row in manifest_rows
                if row["entry_date"] == entry_date
            ]
        )

        symbols = (
            sorted(
                date_manifest["raw_symbol"]
                .drop_duplicates()
                .tolist()
            )
            if not date_manifest.empty
            else []
        )

        start_utc = target_utc - quote_half_window
        end_utc = target_utc + quote_half_window
        condition = dataset_condition(
            client,
            entry_date,
        )

        cost, records = estimate_request(
            client,
            symbols,
            start_utc,
            end_utc,
        )

        estimate_rows.append(
            {
                "entry_date": entry_date,
                "bucket": bucket,
                "dataset_condition": condition,
                "chosen_expiration": chosen_expiration,
                "chosen_dte_calendar": chosen_dte,
                "requested_distance_count": len(distances),
                "executable_distance_count": chosen_count,
                "unique_option_symbols": len(symbols),
                "symbols": symbols,
                "target_entry_timestamp_utc": target_utc,
                "quote_window_start_utc": start_utc,
                "quote_window_end_utc": end_utc,
                "estimated_cost_usd": cost,
                "estimated_record_count": records,
            }
        )

        print(
            f"  Quote estimate | "
            f"Unique symbols={len(symbols):>2} | "
            f"Condition={condition:<9} | "
            f"Cost=${cost:,.6f} | "
            f"Records={records:,}"
        )

    feasibility = pd.DataFrame(feasibility_rows)
    selections = pd.DataFrame(selection_rows).sort_values(
        ["entry_date", "closer_offset_pct"]
    ).reset_index(drop=True)
    manifest = pd.DataFrame(manifest_rows).sort_values(
        ["entry_date", "closer_offset_pct", "leg_name"]
    ).reset_index(drop=True)
    estimates = pd.DataFrame(estimate_rows).sort_values(
        "entry_date"
    ).reset_index(drop=True)

    print("\nSelected comparison spreads")
    print("-" * 196)

    for row in selections.itertuples(index=False):
        if row.selection_status == "Selected":
            print(
                f"{row.entry_date} | "
                f"Bucket={row.bucket:<10} | "
                f"{row.distance_label:<12} | "
                f"Target distance={row.target_distance_pct:>4.1f}% | "
                f"Spread={row.selected_short_strike:,.0f}/"
                f"{row.selected_long_strike:,.0f} | "
                f"Actual distance={row.actual_short_distance_pct:>5.2f}% | "
                f"Hist loss={row.historical_mean_intrinsic_loss_points:>6.2f} "
                f"[{row.historical_loss_ci_low:>5.2f}, "
                f"{row.historical_loss_ci_high:>5.2f}] | "
                f"Breach={row.historical_breach_rate * 100:>5.2f}%"
            )
        else:
            print(
                f"{row.entry_date} | "
                f"Bucket={row.bucket:<10} | "
                f"{row.distance_label:<12} | "
                f"Target distance={row.target_distance_pct:>4.1f}% | "
                f"{row.selection_status}"
            )

    total_cost = float(
        estimates["estimated_cost_usd"].sum()
    )

    total_records = int(
        estimates["estimated_record_count"].sum()
    )

    print("\nCombined option quote estimate")
    print("-" * 96)
    print(f"Dates evaluated: {len(estimates)}")
    print(
        "Executable distance-date combinations: "
        f"{int((selections['selection_status'] == 'Selected').sum())}"
    )
    print(f"Unique date-level requests: {len(estimates)}")
    print(f"Total estimated cost: ${total_cost:,.6f}")
    print(f"Total estimated records: {total_records:,}")

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    feasibility.to_csv(
        FEASIBILITY_CSV,
        index=False,
    )

    selections.to_csv(
        SELECTION_CSV,
        index=False,
    )

    manifest.to_csv(
        OPTION_MANIFEST_CSV,
        index=False,
    )

    payload = {
        "dataset": "GLBX.MDP3",
        "schema": "bbo-1s",
        "spread_width_points": spread_width,
        "closer_offsets_pct": [
            float(value)
            for value in config["closer_offsets_pct"]
        ],
        "combined_estimated_cost_usd": total_cost,
        "combined_estimated_record_count": total_records,
        "historical_option_quotes_downloaded": False,
        "requests": estimates.to_dict(orient="records"),
    }

    ESTIMATE_JSON.write_text(
        json.dumps(
            payload,
            indent=2,
            default=str,
        )
        + "\n",
        encoding="utf-8",
    )

    summary_lines = [
        "MNQ PCS REGIME RESEARCH — CLOSER-STRIKE QUOTE TEST",
        "=" * 68,
        "",
        f"Dates evaluated: {len(estimates)}",
        (
            "Executable distance-date combinations: "
            f"{int((selections['selection_status'] == 'Selected').sum())}"
        ),
        f"Estimated quote cost: ${total_cost:,.6f}",
        f"Estimated records: {total_records:,}",
        "",
        (
            "A single expiration is used per date so market-credit changes "
            "across distances are not confounded by DTE."
        ),
        (
            "Historical expected loss is joined at the requested target "
            "distance. Because the selected strike is rounded at or below "
            "the target, this is generally conservative."
        ),
        "",
        "No historical option quotes were downloaded.",
    ]

    SUMMARY_TXT.write_text(
        "\n".join(summary_lines) + "\n",
        encoding="utf-8",
    )

    print("\nNo historical option quote data was downloaded.")
    print(f"Expiration feasibility: {FEASIBILITY_CSV}")
    print(f"Selected comparison spreads: {SELECTION_CSV}")
    print(f"Exact option manifest: {OPTION_MANIFEST_CSV}")
    print(f"Cost estimates: {ESTIMATE_JSON}")
    print(f"Summary: {SUMMARY_TXT}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

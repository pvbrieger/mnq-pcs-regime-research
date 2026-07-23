"""Re-evaluate every eligible expiration using the downloaded MNQ midpoint.

The prior checkpoint chose the expiration closest to 28 DTE before testing
whether its listed strike range reached the VIX-ladder target. This local-only
audit reverses that order:

1. Calculate the ladder target from the downloaded 3:45 p.m. MNQ midpoint.
2. Test every 26–35 DTE expiration tied to that quoted MNQ contract.
3. Keep expirations with an exact 150-point pair at or below the target.
4. Among feasible expirations, choose the one closest to 28 DTE.

No Databento request is made and no data is downloaded.

Run:
    python scripts/reselect_vix_bucket_spreads_all_expirations.py
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results" / "vix_bucket_credit_pilot"

SNAPSHOT_PATH = RESULTS_DIR / "underlying_quote_snapshot.csv"
CANDIDATES_PATH = RESULTS_DIR / "mnq_option_definition_candidates.csv"

AUDIT_CSV = RESULTS_DIR / "all_expiration_spread_feasibility.csv"
SELECTION_CSV = RESULTS_DIR / "selected_ladder_spreads_all_expirations.csv"
OPTION_MANIFEST_CSV = (
    RESULTS_DIR / "option_quote_manifest_all_expirations.csv"
)
SUMMARY_TXT = (
    RESULTS_DIR / "all_expiration_spread_selection_summary.txt"
)


def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load and validate futures snapshots and definition candidates."""

    for path in (SNAPSHOT_PATH, CANDIDATES_PATH):
        if not path.is_file():
            raise FileNotFoundError(f"Missing required file: {path}")

    snapshot = pd.read_csv(SNAPSHOT_PATH)
    candidates = pd.read_csv(CANDIDATES_PATH)

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

    candidate_required = {
        "entry_date",
        "bucket",
        "vix_close",
        "ladder_distance_pct",
        "expiration",
        "dte_calendar",
        "underlying",
        "strike_price",
        "raw_symbol",
    }

    for name, frame, required in (
        ("underlying snapshot", snapshot, snapshot_required),
        ("definition candidates", candidates, candidate_required),
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

    snapshot["quote_available"] = (
        snapshot["quote_available"]
        .astype(str)
        .str.lower()
        .map({"true": True, "false": False})
        .fillna(False)
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

    return (
        snapshot.sort_values("entry_date").reset_index(drop=True),
        candidates,
    )


def exact_pairs(
    group: pd.DataFrame,
    width: float = 150.0,
    tolerance: float = 0.01,
) -> pd.DataFrame:
    """Return every exact-width pair in one option series."""

    strike_map = (
        group.dropna(subset=["strike_price"])
        .sort_values(["strike_price", "raw_symbol"])
        .drop_duplicates("strike_price", keep="first")
        .set_index("strike_price")["raw_symbol"]
        .to_dict()
    )

    strikes = sorted(float(value) for value in strike_map)
    strike_set = set(strikes)
    rows: list[dict[str, Any]] = []

    for short_strike in strikes:
        desired_long = short_strike - width

        long_matches = [
            strike
            for strike in strike_set
            if abs(strike - desired_long) <= tolerance
        ]

        if not long_matches:
            continue

        long_strike = sorted(long_matches)[0]

        rows.append(
            {
                "short_strike": short_strike,
                "long_strike": long_strike,
                "short_symbol": strike_map[short_strike],
                "long_symbol": strike_map[long_strike],
            }
        )

    return pd.DataFrame(rows)


def main() -> int:
    """Audit all expirations and select the best feasible one per date."""

    print(
        "MNQ PCS Regime Research — All-Expiration Spread Reselection"
    )
    print("=" * 70)

    snapshot, candidates = load_inputs()

    audit_rows: list[dict[str, Any]] = []
    selection_rows: list[dict[str, Any]] = []
    option_manifest_rows: list[dict[str, Any]] = []

    print("\nAll-expiration feasibility")
    print("-" * 184)

    for snap in snapshot.itertuples(index=False):
        entry_date = str(snap.entry_date)

        if not bool(snap.quote_available) or pd.isna(snap.mid):
            selection_rows.append(
                {
                    "entry_date": entry_date,
                    "bucket": snap.bucket,
                    "vix_close": snap.vix_close,
                    "ladder_distance_pct": snap.ladder_distance_pct,
                    "underlying": snap.underlying,
                    "underlying_mid": np.nan,
                    "ladder_target_strike": np.nan,
                    "expiration": pd.NaT,
                    "dte_calendar": np.nan,
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

        target = float(snap.mid) * (
            1.0 - float(snap.ladder_distance_pct) / 100.0
        )

        date_candidates = candidates.loc[
            (candidates["entry_date"] == entry_date)
            & (candidates["underlying"] == str(snap.underlying))
            & candidates["expiration"].notna()
            & candidates["dte_calendar"].between(
                26,
                35,
                inclusive="both",
            )
        ].copy()

        feasible_choices: list[dict[str, Any]] = []

        if date_candidates.empty:
            selection_rows.append(
                {
                    "entry_date": entry_date,
                    "bucket": snap.bucket,
                    "vix_close": float(snap.vix_close),
                    "ladder_distance_pct": float(
                        snap.ladder_distance_pct
                    ),
                    "underlying": snap.underlying,
                    "underlying_mid": float(snap.mid),
                    "ladder_target_strike": target,
                    "expiration": pd.NaT,
                    "dte_calendar": np.nan,
                    "selected_short_strike": np.nan,
                    "selected_long_strike": np.nan,
                    "selected_short_pct_below_mnq": np.nan,
                    "target_to_selected_short_slippage_points": np.nan,
                    "short_symbol": "",
                    "long_symbol": "",
                    "selection_status": (
                        "No 26–35 DTE definitions for quoted underlying"
                    ),
                }
            )
            continue

        group_columns = ["expiration", "dte_calendar"]

        for (expiration, dte), group in date_candidates.groupby(
            group_columns,
            sort=True,
        ):
            pairs = exact_pairs(group)

            if pairs.empty:
                pair_count = 0
                eligible = pairs
            else:
                pair_count = len(pairs)
                eligible = pairs.loc[
                    pairs["short_strike"] <= target
                ].copy()

            if eligible.empty:
                selected_short = np.nan
                selected_long = np.nan
                short_symbol = ""
                long_symbol = ""
                slippage = np.nan
                feasible = False
            else:
                chosen = eligible.sort_values(
                    "short_strike"
                ).iloc[-1]

                selected_short = float(
                    chosen["short_strike"]
                )
                selected_long = float(
                    chosen["long_strike"]
                )
                short_symbol = str(
                    chosen["short_symbol"]
                )
                long_symbol = str(
                    chosen["long_symbol"]
                )
                slippage = target - selected_short
                feasible = True

                feasible_choices.append(
                    {
                        "expiration": expiration,
                        "dte_calendar": int(dte),
                        "selected_short_strike": selected_short,
                        "selected_long_strike": selected_long,
                        "short_symbol": short_symbol,
                        "long_symbol": long_symbol,
                        "target_to_selected_short_slippage_points": (
                            slippage
                        ),
                    }
                )

            audit_rows.append(
                {
                    "entry_date": entry_date,
                    "bucket": snap.bucket,
                    "vix_close": float(snap.vix_close),
                    "ladder_distance_pct": float(
                        snap.ladder_distance_pct
                    ),
                    "underlying": snap.underlying,
                    "underlying_mid": float(snap.mid),
                    "ladder_target_strike": target,
                    "expiration": expiration,
                    "dte_calendar": int(dte),
                    "exact_150_point_pairs": pair_count,
                    "pairs_at_or_below_target": int(
                        len(eligible)
                    ),
                    "feasible": feasible,
                    "best_short_at_or_below_target": (
                        selected_short
                    ),
                    "best_long_strike": selected_long,
                    "target_to_best_short_slippage_points": (
                        slippage
                    ),
                }
            )

            result_text = (
                f"{selected_short:,.0f}/{selected_long:,.0f}"
                if feasible
                else "not executable"
            )

            print(
                f"{entry_date} | "
                f"Bucket={snap.bucket:<10} | "
                f"VIX={float(snap.vix_close):>6.2f} | "
                f"MNQ={float(snap.mid):>10.2f} | "
                f"Target={target:>10.2f} | "
                f"Exp={pd.Timestamp(expiration).date()} | "
                f"DTE={int(dte):>2} | "
                f"Pairs={pair_count:>3} | "
                f"At/below={len(eligible):>3} | "
                f"Best={result_text}"
            )

        if not feasible_choices:
            selection_rows.append(
                {
                    "entry_date": entry_date,
                    "bucket": snap.bucket,
                    "vix_close": float(snap.vix_close),
                    "ladder_distance_pct": float(
                        snap.ladder_distance_pct
                    ),
                    "underlying": snap.underlying,
                    "underlying_mid": float(snap.mid),
                    "ladder_target_strike": target,
                    "expiration": pd.NaT,
                    "dte_calendar": np.nan,
                    "selected_short_strike": np.nan,
                    "selected_long_strike": np.nan,
                    "selected_short_pct_below_mnq": np.nan,
                    "target_to_selected_short_slippage_points": np.nan,
                    "short_symbol": "",
                    "long_symbol": "",
                    "selection_status": (
                        "No exact 150-point pair at or below target "
                        "in any eligible expiration"
                    ),
                }
            )
            continue

        feasible_frame = pd.DataFrame(feasible_choices)

        feasible_frame["distance_from_28"] = (
            feasible_frame["dte_calendar"] - 28
        ).abs()

        chosen = feasible_frame.sort_values(
            [
                "distance_from_28",
                "dte_calendar",
                "expiration",
            ]
        ).iloc[0]

        short_strike = float(
            chosen["selected_short_strike"]
        )
        long_strike = float(
            chosen["selected_long_strike"]
        )

        selection_rows.append(
            {
                "entry_date": entry_date,
                "bucket": snap.bucket,
                "vix_close": float(snap.vix_close),
                "ladder_distance_pct": float(
                    snap.ladder_distance_pct
                ),
                "underlying": snap.underlying,
                "underlying_mid": float(snap.mid),
                "ladder_target_strike": target,
                "expiration": chosen["expiration"],
                "dte_calendar": int(
                    chosen["dte_calendar"]
                ),
                "selected_short_strike": short_strike,
                "selected_long_strike": long_strike,
                "selected_short_pct_below_mnq": (
                    (float(snap.mid) - short_strike)
                    / float(snap.mid)
                    * 100.0
                ),
                "target_to_selected_short_slippage_points": float(
                    chosen[
                        "target_to_selected_short_slippage_points"
                    ]
                ),
                "short_symbol": str(
                    chosen["short_symbol"]
                ),
                "long_symbol": str(
                    chosen["long_symbol"]
                ),
                "selection_status": "Selected",
            }
        )

        for leg_name, strike, symbol in (
            (
                "short_put",
                short_strike,
                str(chosen["short_symbol"]),
            ),
            (
                "long_put",
                long_strike,
                str(chosen["long_symbol"]),
            ),
        ):
            option_manifest_rows.append(
                {
                    "entry_date": entry_date,
                    "bucket": snap.bucket,
                    "vix_close": float(snap.vix_close),
                    "ladder_distance_pct": float(
                        snap.ladder_distance_pct
                    ),
                    "expiration": chosen["expiration"],
                    "dte_calendar": int(
                        chosen["dte_calendar"]
                    ),
                    "underlying": snap.underlying,
                    "underlying_mid": float(snap.mid),
                    "ladder_target_strike": target,
                    "selected_short_strike": short_strike,
                    "selected_long_strike": long_strike,
                    "leg_name": leg_name,
                    "strike_price": strike,
                    "raw_symbol": symbol,
                    "target_entry_timestamp_utc": (
                        snap.target_entry_timestamp_utc
                    ),
                }
            )

    audit = pd.DataFrame(audit_rows)
    selection = pd.DataFrame(selection_rows).sort_values(
        "entry_date"
    ).reset_index(drop=True)
    option_manifest = pd.DataFrame(
        option_manifest_rows
    ).sort_values(
        ["entry_date", "leg_name"]
    ).reset_index(drop=True)

    selected_count = int(
        (selection["selection_status"] == "Selected").sum()
    )

    print("\nFinal all-expiration selections")
    print("-" * 176)

    for row in selection.itertuples(index=False):
        if row.selection_status == "Selected":
            print(
                f"{row.entry_date} | "
                f"Bucket={row.bucket:<10} | "
                f"VIX={row.vix_close:>6.2f} | "
                f"MNQ={row.underlying_mid:>10.2f} | "
                f"Exp={pd.Timestamp(row.expiration).date()} | "
                f"DTE={int(row.dte_calendar):>2} | "
                f"Spread={row.selected_short_strike:,.0f}/"
                f"{row.selected_long_strike:,.0f} | "
                f"Actual distance="
                f"{row.selected_short_pct_below_mnq:>5.2f}%"
            )
        else:
            print(
                f"{row.entry_date} | "
                f"Bucket={row.bucket:<10} | "
                f"{row.selection_status}"
            )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    audit.to_csv(AUDIT_CSV, index=False)
    selection.to_csv(SELECTION_CSV, index=False)
    option_manifest.to_csv(
        OPTION_MANIFEST_CSV,
        index=False,
    )

    unselected = selection.loc[
        selection["selection_status"] != "Selected",
        ["entry_date", "bucket", "selection_status"],
    ]

    summary_lines = [
        "MNQ PCS REGIME RESEARCH — ALL-EXPIRATION RESELECTION",
        "=" * 68,
        "",
        f"Underlying dates evaluated: {len(snapshot)}",
        f"Executable ladder spreads selected: {selected_count}",
        f"Dates still unavailable: {len(unselected)}",
        "",
    ]

    for row in unselected.itertuples(index=False):
        summary_lines.append(
            f"{row.entry_date} | {row.bucket} | "
            f"{row.selection_status}"
        )

    summary_lines.extend(
        [
            "",
            "No Databento request was made.",
            "No historical data was downloaded.",
        ]
    )

    SUMMARY_TXT.write_text(
        "\n".join(summary_lines) + "\n",
        encoding="utf-8",
    )

    print("\nOutputs")
    print("-" * 108)
    print(f"Expiration feasibility: {AUDIT_CSV}")
    print(f"Final spread selections: {SELECTION_CSV}")
    print(f"Final option manifest: {OPTION_MANIFEST_CSV}")
    print(f"Summary: {SUMMARY_TXT}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

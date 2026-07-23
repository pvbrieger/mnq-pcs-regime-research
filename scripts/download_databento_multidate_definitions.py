"""Download and process two Databento definition snapshots.

Run:
    python scripts/download_databento_multidate_definitions.py \
        --download --max-cost-usd 1.25

Raw DBN files are reused when already present. Historical downloads are blocked
when the current combined estimate exceeds the explicit cost ceiling.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import pandas as pd
import yaml
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "config" / "databento_multidate_pilot.yaml"
RAW_DIR = ROOT / "data" / "raw" / "databento" / "definitions"
RESULTS = ROOT / "results" / "databento_multidate_pilot"

CANDIDATES_CSV = RESULTS / "mnq_option_definition_candidates.csv"
FEASIBILITY_CSV = RESULTS / "spread_feasibility_audit.csv"
MANIFEST_CSV = RESULTS / "quote_request_manifest.csv"
AUDIT_JSON = RESULTS / "definition_download_audit.json"
SUMMARY_TXT = RESULTS / "definition_download_summary.txt"


def load_config() -> dict[str, Any]:
    with CONFIG.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def api_key() -> str:
    load_dotenv(ROOT / ".env", override=True)
    value = os.getenv("DATABENTO_API_KEY", "").strip()
    if not value.startswith("db-"):
        raise RuntimeError("A valid DATABENTO_API_KEY is required in .env.")
    return value


def raw_path(dataset: str, entry_date: str) -> Path:
    return RAW_DIR / (
        f"{dataset.replace('.', '_')}_{entry_date}_definitions.dbn.zst"
    )


def date_window(entry_date: str) -> tuple[pd.Timestamp, pd.Timestamp]:
    start = pd.Timestamp(entry_date).normalize().tz_localize("UTC")
    return start, start + pd.Timedelta(days=1)


def estimate_request(
    client: Any,
    dataset: str,
    entry_date: str,
) -> tuple[float, int]:
    start, end = date_window(entry_date)
    kwargs = dict(
        dataset=dataset,
        symbols="ALL_SYMBOLS",
        stype_in="raw_symbol",
        schema="definition",
        start=start.isoformat(),
        end=end.isoformat(),
    )
    cost = float(client.metadata.get_cost(**kwargs))
    records = int(client.metadata.get_record_count(**kwargs))
    return cost, records


def download_request(
    client: Any,
    dataset: str,
    entry_date: str,
    output: Path,
) -> None:
    start, end = date_window(entry_date)
    store = client.timeseries.get_range(
        dataset=dataset,
        symbols="ALL_SYMBOLS",
        stype_in="raw_symbol",
        schema="definition",
        start=start.isoformat(),
        end=end.isoformat(),
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    store.to_file(output)


def clean_text(series: pd.Series) -> pd.Series:
    return series.fillna("").astype(str).str.strip()


def load_candidates(
    db: Any,
    file_path: Path,
    pilot: dict[str, Any],
    dte_min: int,
    dte_max: int,
) -> pd.DataFrame:
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
            f"{file_path.name} is missing fields: {', '.join(missing)}"
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
        frame["strike_price"], errors="coerce"
    )
    frame["instrument_id"] = pd.to_numeric(
        frame["instrument_id"], errors="coerce"
    )
    frame["expiration"] = pd.to_datetime(
        frame["expiration"], errors="coerce", utc=True
    )

    ts_column = "ts_recv" if "ts_recv" in frame.columns else "ts_event"
    frame["definition_timestamp"] = pd.to_datetime(
        frame[ts_column], errors="coerce", utc=True
    )

    entry_date = pd.Timestamp(pilot["entry_date"]).normalize()
    expiration_dates = (
        frame["expiration"]
        .dt.tz_convert("UTC")
        .dt.tz_localize(None)
        .dt.normalize()
    )
    frame["dte_calendar"] = (expiration_dates - entry_date).dt.days

    mask = (
        (frame["instrument_class"] == "P")
        & (
            frame["underlying"].str.startswith("MNQ")
            | frame["asset"].str.startswith("MQ")
        )
        & frame["strike_price"].notna()
        & frame["instrument_id"].notna()
        & frame["expiration"].notna()
        & frame["dte_calendar"].between(dte_min, dte_max)
    )

    output = frame.loc[
        mask,
        [
            "definition_timestamp",
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
            ["raw_symbol", "instrument_id", "definition_timestamp"]
        )
        .drop_duplicates(["raw_symbol", "instrument_id"], keep="last")
        .reset_index(drop=True)
    )

    if output.empty:
        raise ValueError(
            f"No MNQ put definitions found for {pilot['entry_date']}."
        )

    output.insert(0, "entry_date", str(pilot["entry_date"]))
    output.insert(1, "score", int(pilot["score"]))
    output.insert(2, "ndx_close", float(pilot["ndx_close"]))
    return output


def exact_row(
    frame: pd.DataFrame,
    strike: float,
    tolerance: float,
) -> pd.Series | None:
    matches = frame.loc[
        (frame["strike_price"] - strike).abs() <= tolerance
    ]
    if matches.empty:
        return None
    return matches.sort_values(["strike_price", "raw_symbol"]).iloc[0]


def choose_spread(
    frame: pd.DataFrame,
    target: float,
    width: float,
    tolerance: float,
) -> dict[str, Any]:
    shorts = frame.loc[
        frame["strike_price"] <= target + tolerance
    ].copy()

    if shorts.empty:
        return {
            "feasible": False,
            "reason": "No listed short at or below target",
        }

    shorts["distance"] = target - shorts["strike_price"]
    shorts = shorts.sort_values(["distance", "raw_symbol"])
    nearest = float(shorts.iloc[0]["strike_price"])

    for _, short in shorts.iterrows():
        short_strike = float(short["strike_price"])
        long = exact_row(frame, short_strike - width, tolerance)
        if long is not None:
            return {
                "feasible": True,
                "reason": "Executable",
                "nearest": nearest,
                "short_strike": short_strike,
                "long_strike": float(long["strike_price"]),
                "short_row": short,
                "long_row": long,
            }

    return {
        "feasible": False,
        "reason": f"No exact {width:.0f}-point pair at or below target",
        "nearest": nearest,
    }


def build_outputs(
    candidates: pd.DataFrame,
    config: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    width = float(config["spread_width_points"])
    tolerance = float(config.get("strike_match_tolerance_points", 0.01))
    pilots = {str(item["entry_date"]): item for item in config["pilots"]}

    audits: list[dict[str, Any]] = []
    selections: dict[tuple[Any, ...], dict[str, Any]] = {}

    group_cols = [
        "entry_date",
        "expiration",
        "underlying",
        "asset",
        "dte_calendar",
    ]

    for group_key, group in candidates.groupby(group_cols, dropna=False):
        entry_date, expiration, underlying, asset, dte = group_key
        pilot = pilots[str(entry_date)]

        for name, target in (
            ("standard", float(pilot["standard_target_short_strike"])),
            ("defensive", float(pilot["defensive_target_short_strike"])),
        ):
            result = choose_spread(group, target, width, tolerance)
            key = (*group_key, name)
            selections[key] = result

            short_strike = result.get("short_strike")
            audits.append(
                {
                    "entry_date": entry_date,
                    "expiration": expiration,
                    "underlying": underlying,
                    "asset": asset,
                    "dte_calendar": int(dte),
                    "candidate_type": name,
                    "target_short_strike": target,
                    "feasible": bool(result["feasible"]),
                    "reason": result["reason"],
                    "nearest_unconstrained_short_strike": result.get("nearest"),
                    "selected_short_strike": short_strike,
                    "selected_long_strike": result.get("long_strike"),
                    "short_distance_below_target_points": (
                        target - short_strike
                        if short_strike is not None
                        else None
                    ),
                }
            )

    audit = pd.DataFrame(audits)
    audit["expiration_included_in_quote_pilot"] = False
    manifest_rows: list[dict[str, Any]] = []

    for base_key, group in audit.groupby(group_cols, dropna=False):
        status = dict(zip(group["candidate_type"], group["feasible"]))
        included = bool(status.get("standard")) and bool(
            status.get("defensive")
        )
        audit.loc[
            group.index, "expiration_included_in_quote_pilot"
        ] = included

        if not included:
            continue

        entry_date, expiration, underlying, asset, dte = base_key
        pilot = pilots[str(entry_date)]

        for name, target in (
            ("standard", float(pilot["standard_target_short_strike"])),
            ("defensive", float(pilot["defensive_target_short_strike"])),
        ):
            result = selections[(*base_key, name)]
            short = result["short_row"]
            long = result["long_row"]

            for leg_name, side, row in (
                ("short_put", "sell", short),
                ("long_put", "buy", long),
            ):
                manifest_rows.append(
                    {
                        "entry_date": entry_date,
                        "score": int(pilot["score"]),
                        "ndx_close": float(pilot["ndx_close"]),
                        "expiration": expiration,
                        "dte_calendar": int(dte),
                        "underlying": underlying,
                        "asset": asset,
                        "candidate_type": name,
                        "target_short_strike": target,
                        "selected_short_strike": result["short_strike"],
                        "selected_long_strike": result["long_strike"],
                        "short_distance_below_target_points": (
                            target - result["short_strike"]
                        ),
                        "spread_width_points": (
                            result["short_strike"] - result["long_strike"]
                        ),
                        "leg_name": leg_name,
                        "side": side,
                        "strike_price": float(row["strike_price"]),
                        "raw_symbol": str(row["raw_symbol"]),
                        "instrument_id": int(row["instrument_id"]),
                    }
                )

    manifest = pd.DataFrame(manifest_rows)
    return audit, manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--max-cost-usd", type=float, default=0.0)
    parser.add_argument("--force-redownload", action="store_true")
    args = parser.parse_args()

    print("MNQ PCS Regime Research — Multi-Date Definition Download")
    print("=" * 64)

    import databento as db

    config = load_config()
    client = db.Historical(api_key())
    dataset = str(config["dataset"])

    file_audit: list[dict[str, Any]] = []
    missing: list[tuple[dict[str, Any], Path, float, int]] = []

    print("\nDefinition files and estimates")
    print("-" * 118)

    for pilot in config["pilots"]:
        entry_date = str(pilot["entry_date"])
        output = raw_path(dataset, entry_date)
        reuse = output.is_file() and not args.force_redownload

        if reuse:
            cost, records = 0.0, 0
        else:
            cost, records = estimate_request(client, dataset, entry_date)
            missing.append((pilot, output, cost, records))

        print(
            f"{entry_date} | "
            f"{'reuse local file' if reuse else 'download required':<18} | "
            f"Cost=${cost:,.6f} | Records={records:,}"
        )

        file_audit.append(
            {
                "entry_date": entry_date,
                "raw_path": str(output),
                "file_reused": reuse,
                "estimated_cost_usd": cost,
                "estimated_record_count": records,
                "downloaded_during_this_run": False,
            }
        )

    total_cost = sum(item[2] for item in missing)
    print(f"\nTotal estimated cost for missing files: ${total_cost:,.6f}")

    if missing and not args.download:
        raise RuntimeError(
            "Missing files require --download and --max-cost-usd."
        )
    if missing and args.max_cost_usd <= 0:
        raise RuntimeError("A positive --max-cost-usd is required.")
    if total_cost > args.max_cost_usd:
        raise RuntimeError(
            f"Blocked: ${total_cost:,.6f} exceeds "
            f"${args.max_cost_usd:,.6f}."
        )

    for pilot, output, cost, _ in missing:
        entry_date = str(pilot["entry_date"])
        print(f"\nDownloading {entry_date}; estimated cost ${cost:,.6f}.")
        download_request(client, dataset, entry_date, output)
        for row in file_audit:
            if row["entry_date"] == entry_date:
                row["downloaded_during_this_run"] = True

    frames: list[pd.DataFrame] = []

    print("\nPoint-in-time MNQ put definitions")
    print("-" * 96)

    for pilot in config["pilots"]:
        entry_date = str(pilot["entry_date"])
        frame = load_candidates(
            db,
            raw_path(dataset, entry_date),
            pilot,
            int(config["dte_calendar_min"]),
            int(config["dte_calendar_max"]),
        )
        frames.append(frame)
        print(
            f"{entry_date} | Candidates={len(frame):>4} | "
            f"Expirations={frame['expiration'].nunique():>2} | "
            f"Underlyings={', '.join(sorted(frame['underlying'].unique()))}"
        )

    candidates = pd.concat(frames, ignore_index=True)
    feasibility, manifest = build_outputs(candidates, config)

    print("\nSpread feasibility")
    print("-" * 160)

    for row in feasibility.itertuples(index=False):
        spread = (
            f"{row.selected_short_strike:,.0f}/"
            f"{row.selected_long_strike:,.0f}"
            if row.feasible
            else "not executable"
        )
        print(
            f"{row.entry_date} | Exp={row.expiration} | "
            f"DTE={row.dte_calendar:>2} | {row.candidate_type:<9} | "
            f"Target={row.target_short_strike:>10,.2f} | "
            f"Spread={spread:<20} | "
            f"Pilot={'included' if row.expiration_included_in_quote_pilot else 'excluded'}"
        )

    RESULTS.mkdir(parents=True, exist_ok=True)
    candidates.to_csv(CANDIDATES_CSV, index=False)
    feasibility.to_csv(FEASIBILITY_CSV, index=False)
    manifest.to_csv(MANIFEST_CSV, index=False)

    AUDIT_JSON.write_text(
        json.dumps(
            {
                "dataset": dataset,
                "estimated_missing_cost_usd": total_cost,
                "files": file_audit,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    included = (
        feasibility.loc[
            feasibility["expiration_included_in_quote_pilot"]
        ]
        .groupby("entry_date")["expiration"]
        .nunique()
        .to_dict()
    )

    summary = [
        "MNQ PCS REGIME RESEARCH — MULTI-DATE DEFINITIONS",
        "=" * 60,
        f"Candidate rows: {len(candidates):,}",
        f"Feasibility rows: {len(feasibility):,}",
        f"Quote-manifest rows: {len(manifest):,}",
    ]
    for pilot in config["pilots"]:
        date = str(pilot["entry_date"])
        summary.append(
            f"{date} included expirations: {included.get(date, 0)}"
        )

    SUMMARY_TXT.write_text(
        "\n".join(summary) + "\n",
        encoding="utf-8",
    )

    print("\nOutputs")
    print("-" * 96)
    print(f"Definition candidates: {CANDIDATES_CSV}")
    print(f"Feasibility audit: {FEASIBILITY_CSV}")
    print(f"Quote manifest: {MANIFEST_CSV}")
    print(f"Download audit: {AUDIT_JSON}")
    print(f"Summary: {SUMMARY_TXT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Select a reproducible pure-MNQ VIX-bucket credit pilot.

This checkpoint downloads nothing. It:

- starts on the MNQ options launch date, 2020-08-31;
- uses only Fridays through the fixed VIX data cutoff, 2026-07-02;
- excludes observations within 0.50 VIX points of bucket boundaries;
- selects two temporally separated Fridays per bucket when possible;
- selects the sole qualifying Friday above VIX 40;
- requires Databento condition `available`;
- uses a fixed random seed; and
- estimates one day of ALL_SYMBOLS definitions for each selected Friday.

Run:
    python scripts/select_vix_bucket_credit_pilot.py
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
CONFIG_PATH = ROOT / "config" / "vix_bucket_credit_pilot.yaml"
RESULTS_DIR = ROOT / "results" / "vix_bucket_credit_pilot"

SELECTION_CSV = RESULTS_DIR / "selected_pilot_dates.csv"
CANDIDATE_COUNTS_CSV = RESULTS_DIR / "candidate_population_counts.csv"
ESTIMATE_JSON = RESULTS_DIR / "definition_cost_estimates.json"
SUMMARY_TXT = RESULTS_DIR / "selection_summary.txt"


def load_config() -> dict[str, Any]:
    """Load the pilot configuration."""

    if not CONFIG_PATH.is_file():
        raise FileNotFoundError(f"Missing configuration: {CONFIG_PATH}")

    with CONFIG_PATH.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def load_environment() -> tuple[str, str]:
    """Load shared-data root and Databento key."""

    load_dotenv(ROOT / ".env", override=True)

    shared_root = os.getenv("SHARED_MARKET_DATA", "").strip()
    api_key = os.getenv("DATABENTO_API_KEY", "").strip()

    if not shared_root:
        raise RuntimeError("SHARED_MARKET_DATA is missing from .env.")

    if not api_key.startswith("db-"):
        raise RuntimeError(
            "A valid DATABENTO_API_KEY is required in the project .env file."
        )

    return shared_root, api_key


def find_vix_csv(shared_root: str) -> Path:
    """Find the largest usable VIX CSV under the shared-data root."""

    root = Path(shared_root).expanduser()

    if not root.exists():
        raise FileNotFoundError(f"Shared-data root does not exist: {root}")

    candidates = sorted(
        {
            path
            for pattern in ("**/*VIX*.csv", "**/*vix*.csv")
            for path in root.glob(pattern)
            if path.is_file()
        }
    )

    usable: list[tuple[int, Path]] = []

    for path in candidates:
        try:
            preview = pd.read_csv(path, nrows=5)
        except Exception:
            continue

        columns = {
            str(column).strip().lower()
            for column in preview.columns
        }

        if {"date", "close"}.issubset(columns):
            try:
                row_count = (
                    sum(1 for _ in path.open("r", encoding="utf-8")) - 1
                )
            except UnicodeDecodeError:
                row_count = len(pd.read_csv(path))

            usable.append((row_count, path))

    if not usable:
        raise FileNotFoundError(
            "No VIX CSV with Date and Close columns was found under "
            f"{root}."
        )

    return sorted(
        usable,
        key=lambda item: (item[0], str(item[1])),
    )[-1][1]


def load_vix_data(path: Path) -> pd.DataFrame:
    """Load and normalize daily VIX observations."""

    frame = pd.read_csv(path)

    lookup = {
        str(column).strip().lower(): column
        for column in frame.columns
    }

    output = pd.DataFrame(
        {
            "date": pd.to_datetime(
                frame[lookup["date"]],
                errors="coerce",
            ).dt.normalize(),
            "vix_close": pd.to_numeric(
                frame[lookup["close"]],
                errors="coerce",
            ),
        }
    )

    return (
        output.dropna(subset=["date", "vix_close"])
        .sort_values("date")
        .drop_duplicates("date", keep="last")
        .reset_index(drop=True)
    )


def bucket_mask(
    values: pd.Series,
    lower: float | None,
    upper: float | None,
    boundary_buffer: float,
) -> pd.Series:
    """Apply a VIX bucket with a boundary buffer."""

    mask = pd.Series(True, index=values.index)

    if lower is not None:
        mask &= values >= lower + boundary_buffer

    if upper is not None:
        mask &= values < upper - boundary_buffer

    return mask


def normalize_condition(value: Any) -> str:
    """Normalize Databento condition text."""

    text = str(value or "unknown").strip().lower()

    if "." in text:
        text = text.rsplit(".", 1)[-1]

    return text


def dataset_condition(
    client: Any,
    dataset: str,
    date_text: str,
) -> str:
    """Return Databento's condition for one date."""

    rows = client.metadata.get_dataset_condition(
        dataset=dataset,
        start_date=date_text,
        end_date=date_text,
    )

    if not rows:
        return "unknown"

    row = rows[0]

    if isinstance(row, dict):
        return normalize_condition(row.get("condition"))

    return normalize_condition(
        getattr(row, "condition", "unknown")
    )


def definition_estimate(
    client: Any,
    dataset: str,
    date_text: str,
) -> tuple[float, int]:
    """Estimate one UTC day of ALL_SYMBOLS definitions."""

    start = pd.Timestamp(date_text).tz_localize("UTC")
    end = start + pd.Timedelta(days=1)

    kwargs = dict(
        dataset=dataset,
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


def choose_available_date(
    candidates: pd.DataFrame,
    client: Any,
    dataset: str,
    allowed_conditions: set[str],
    condition_cache: dict[str, str],
    seed: int,
) -> tuple[pd.Series, str]:
    """Choose a reproducibly randomized date with an accepted condition."""

    randomized = candidates.sample(
        frac=1.0,
        random_state=seed,
    ).reset_index(drop=True)

    for row in randomized.itertuples(index=False):
        date_text = row.date.date().isoformat()

        if date_text not in condition_cache:
            condition_cache[date_text] = dataset_condition(
                client,
                dataset,
                date_text,
            )

        condition = condition_cache[date_text]

        if condition in allowed_conditions:
            return pd.Series(row._asdict()), condition

    raise RuntimeError(
        "No date with an accepted Databento condition remained."
    )


def select_bucket_dates(
    candidates: pd.DataFrame,
    desired_count: int,
    client: Any,
    dataset: str,
    allowed_conditions: set[str],
    condition_cache: dict[str, str],
    seed: int,
) -> list[tuple[pd.Series, str, str]]:
    """Select one or two temporally separated dates."""

    if len(candidates) < desired_count:
        raise RuntimeError(
            f"Bucket has only {len(candidates)} candidates but "
            f"{desired_count} were requested."
        )

    candidates = candidates.sort_values("date").reset_index(drop=True)

    if desired_count == 1:
        row, condition = choose_available_date(
            candidates,
            client,
            dataset,
            allowed_conditions,
            condition_cache,
            seed,
        )
        return [(row, condition, "single_available_sample")]

    midpoint = len(candidates) // 2

    early = candidates.iloc[:midpoint].copy()
    late = candidates.iloc[midpoint:].copy()

    early_row, early_condition = choose_available_date(
        early,
        client,
        dataset,
        allowed_conditions,
        condition_cache,
        seed,
    )

    late_row, late_condition = choose_available_date(
        late,
        client,
        dataset,
        allowed_conditions,
        condition_cache,
        seed + 1,
    )

    return [
        (early_row, early_condition, "earlier_half"),
        (late_row, late_condition, "later_half"),
    ]


def main() -> int:
    """Select the pure-MNQ pilot and estimate definition costs."""

    import databento as db

    print(
        "MNQ PCS Regime Research — Pure-MNQ VIX-Bucket Pilot Selection"
    )
    print("=" * 74)

    config = load_config()
    shared_root, key = load_environment()
    client = db.Historical(key)

    dataset = str(config["dataset"])
    vix_path = find_vix_csv(shared_root)
    vix = load_vix_data(vix_path)

    launch_date = pd.Timestamp(
        config["mnq_options_launch_date"]
    )

    end_date = pd.Timestamp(
        config["selection_end_date"]
    )

    boundary_buffer = float(
        config["boundary_buffer_points"]
    )

    random_seed = int(config["random_seed"])

    allowed_conditions = {
        normalize_condition(value)
        for value in config["accepted_dataset_conditions"]
    }

    eligible = vix.loc[
        (vix["date"] >= launch_date)
        & (vix["date"] <= end_date)
        & (vix["date"].dt.weekday == 4)
    ].copy()

    print(f"VIX source: {vix_path}")
    print(
        f"MNQ options study range: {launch_date.date()} through "
        f"{end_date.date()}"
    )
    print(
        f"Boundary buffer: {boundary_buffer:.2f} VIX points | "
        f"Random seed: {random_seed}"
    )

    condition_cache: dict[str, str] = {}
    selections: list[dict[str, Any]] = []
    population_rows: list[dict[str, Any]] = []

    print("\nCandidate populations and selected dates")
    print("-" * 148)

    for bucket_index, bucket in enumerate(config["buckets"]):
        label = str(bucket["label"])
        lower = (
            float(bucket["minimum"])
            if bucket.get("minimum") is not None
            else None
        )
        upper = (
            float(bucket["maximum"])
            if bucket.get("maximum") is not None
            else None
        )
        desired_count = int(bucket["sample_count"])

        candidates = eligible.loc[
            bucket_mask(
                eligible["vix_close"],
                lower,
                upper,
                boundary_buffer,
            )
        ].copy()

        chosen = select_bucket_dates(
            candidates=candidates,
            desired_count=desired_count,
            client=client,
            dataset=dataset,
            allowed_conditions=allowed_conditions,
            condition_cache=condition_cache,
            seed=random_seed + bucket_index * 100,
        )

        chosen_dates: list[str] = []

        for row, condition, segment in chosen:
            date_text = row["date"].date().isoformat()
            chosen_dates.append(date_text)

            selections.append(
                {
                    "bucket": label,
                    "selection_segment": segment,
                    "entry_date": date_text,
                    "vix_close": float(row["vix_close"]),
                    "ladder_distance_pct": float(
                        bucket["ladder_distance_pct"]
                    ),
                    "dataset_condition": condition,
                    "definition_cost_usd": np.nan,
                    "definition_record_count": np.nan,
                }
            )

        population_rows.append(
            {
                "bucket": label,
                "ladder_distance_pct": float(
                    bucket["ladder_distance_pct"]
                ),
                "candidate_fridays": len(candidates),
                "requested_samples": desired_count,
                "selected_samples": len(chosen),
            }
        )

        print(
            f"{label:<10} | "
            f"Candidates={len(candidates):>3} | "
            f"Requested={desired_count} | "
            f"Selected={', '.join(chosen_dates)}"
        )

    selection = pd.DataFrame(selections).sort_values(
        ["bucket", "entry_date"]
    ).reset_index(drop=True)

    print("\nDefinition estimates")
    print("-" * 132)

    for index, row in selection.iterrows():
        cost, records = definition_estimate(
            client,
            dataset,
            str(row["entry_date"]),
        )

        selection.loc[index, "definition_cost_usd"] = cost
        selection.loc[index, "definition_record_count"] = records

        print(
            f"{row['bucket']:<10} | "
            f"{row['entry_date']} | "
            f"VIX={row['vix_close']:>6.2f} | "
            f"Ladder={row['ladder_distance_pct']:>4.1f}% | "
            f"Condition={row['dataset_condition']:<9} | "
            f"Cost=${cost:>8.6f} | "
            f"Records={records:,}"
        )

    total_cost = float(
        selection["definition_cost_usd"].sum()
    )
    total_records = int(
        selection["definition_record_count"].sum()
    )

    print("\nCombined definition estimate")
    print("-" * 84)
    print(f"Selected dates: {len(selection)}")
    print(f"Total estimated cost: ${total_cost:,.6f}")
    print(f"Total estimated records: {total_records:,}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    population = pd.DataFrame(population_rows)

    selection.to_csv(SELECTION_CSV, index=False)
    population.to_csv(CANDIDATE_COUNTS_CSV, index=False)

    payload = {
        "dataset": dataset,
        "vix_source": str(vix_path),
        "mnq_options_launch_date": launch_date.date().isoformat(),
        "selection_end_date": end_date.date().isoformat(),
        "boundary_buffer_points": boundary_buffer,
        "random_seed": random_seed,
        "accepted_dataset_conditions": sorted(allowed_conditions),
        "selected_dates": selection.to_dict(orient="records"),
        "combined_estimated_definition_cost_usd": total_cost,
        "combined_estimated_definition_record_count": total_records,
        "historical_data_downloaded": False,
    }

    ESTIMATE_JSON.write_text(
        json.dumps(payload, indent=2, default=str) + "\n",
        encoding="utf-8",
    )

    summary_lines = [
        "MNQ PCS REGIME RESEARCH — PURE-MNQ VIX-BUCKET PILOT",
        "=" * 66,
        "",
        f"VIX source: {vix_path}",
        f"Study begins: {launch_date.date()}",
        f"Study cutoff: {end_date.date()}",
        f"Selected dates: {len(selection)}",
        f"Definition cost estimate: ${total_cost:,.6f}",
        f"Definition record estimate: {total_records:,}",
        "",
        "Selected dates",
        "-" * 66,
    ]

    for row in selection.itertuples(index=False):
        summary_lines.append(
            f"{row.bucket} | {row.entry_date} | "
            f"VIX={row.vix_close:.2f} | "
            f"Ladder={row.ladder_distance_pct:.1f}% | "
            f"Condition={row.dataset_condition}"
        )

    SUMMARY_TXT.write_text(
        "\n".join(summary_lines) + "\n",
        encoding="utf-8",
    )

    print("\nNo historical data was downloaded.")
    print(f"Selected dates: {SELECTION_CSV}")
    print(f"Candidate counts: {CANDIDATE_COUNTS_CSV}")
    print(f"Cost estimates: {ESTIMATE_JSON}")
    print(f"Summary: {SUMMARY_TXT}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Screen individual Friday regime factors against the PCS baseline."""

from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Any

import duckdb
import numpy as np
import pandas as pd
import yaml
from scipy.stats import fisher_exact
from statsmodels.stats.proportion import proportion_confint


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATABASE_PATH = PROJECT_ROOT / "database" / "research.duckdb"
FACTOR_CONFIG_PATH = PROJECT_ROOT / "config" / "factor_definitions.yaml"
RESULTS_DIR = PROJECT_ROOT / "results" / "factor_screening"
CSV_OUTPUT = RESULTS_DIR / "factor_screening_results.csv"
SUMMARY_OUTPUT = RESULTS_DIR / "factor_screening_summary.txt"

SUBPERIODS = (
    ("2005–2012", "2005-01-01", "2012-12-31"),
    ("2013–2019", "2013-01-01", "2019-12-31"),
    ("2020–2026", "2020-01-01", "2026-12-31"),
)


def safe_divide(numerator: float, denominator: float) -> float:
    """Return a safe ratio."""

    if denominator == 0 or pd.isna(denominator):
        return np.nan

    return numerator / denominator


def evaluate_rule(series: pd.Series, operator: str, value: Any) -> pd.Series:
    """Evaluate one configured factor condition."""

    operators = {
        "lt": series < value,
        "le": series <= value,
        "gt": series > value,
        "ge": series >= value,
        "eq": series == value,
        "ne": series != value,
    }

    if operator not in operators:
        raise ValueError(f"Unsupported operator: {operator}")

    return operators[operator].fillna(False)


def evaluate_factor(df: pd.DataFrame, factor: dict[str, Any]) -> pd.Series:
    """Evaluate all conditions belonging to one factor."""

    condition = pd.Series(True, index=df.index)

    for rule in factor["all"]:
        column = rule["column"]

        if column not in df.columns:
            raise ValueError(
                f"Factor '{factor['id']}' references missing column '{column}'."
            )

        condition &= evaluate_rule(
            df[column],
            str(rule["operator"]),
            rule["value"],
        )

    return condition


def wilson_interval(successes: int, observations: int) -> tuple[float, float]:
    """Calculate a Wilson 95% confidence interval."""

    if observations == 0:
        return np.nan, np.nan

    lower, upper = proportion_confint(
        count=successes,
        nobs=observations,
        alpha=0.05,
        method="wilson",
    )

    return float(lower), float(upper)


def classify_factor(row: dict[str, Any]) -> str:
    """Assign an exploratory evidence classification."""

    if row["observations"] < 50 or row["failures"] < 2:
        return "Insufficient failure sample"

    if (
        row["observations"] >= 100
        and row["failures"] >= 3
        and row["failure_rate_lift"] >= 1.50
        and row["failure_capture_efficiency"] >= 1.50
        and row["higher_risk_subperiods"] >= 2
    ):
        return "Candidate risk flag"

    if (
        row["failure_rate_lift"] >= 1.25
        and row["failure_capture_efficiency"] >= 1.25
        and row["failures"] >= 2
    ):
        return "Watchlist"

    return "No clear individual edge"


def main() -> int:
    """Run the individual-factor screening program."""

    print("MNQ PCS Regime Research — Individual Factor Screening")
    print("=" * 56)

    if not DATABASE_PATH.is_file():
        print(f"ERROR: Missing database: {DATABASE_PATH}")
        return 1

    if not FACTOR_CONFIG_PATH.is_file():
        print(f"ERROR: Missing factor configuration: {FACTOR_CONFIG_PATH}")
        return 1

    with FACTOR_CONFIG_PATH.open("r", encoding="utf-8") as file_handle:
        factor_config = yaml.safe_load(file_handle)

    factors = factor_config.get("factors", [])

    if not factors:
        print("ERROR: No factors were defined.")
        return 1

    connection = duckdb.connect(str(DATABASE_PATH))

    try:
        df = connection.execute(
            """
            SELECT *
            FROM pcs.friday_regime_features
            WHERE complete_core_features = TRUE
            ORDER BY entry_date
            """
        ).fetchdf()

        df["entry_date"] = pd.to_datetime(df["entry_date"])

        boolean_columns = (
            "expiration_failure",
            "short_strike_touched",
            "clustered_expiration_failure",
            "below_sma20",
            "below_sma40",
            "sma20_falling",
            "sma40_falling",
            "vix_above_sma20",
        )

        for column in boolean_columns:
            df[column] = df[column].fillna(False).astype(bool)

        total_entries = len(df)
        total_failures = int(df["expiration_failure"].sum())
        total_touches = int(df["short_strike_touched"].sum())

        baseline_failure_rate = float(df["expiration_failure"].mean())
        baseline_touch_rate = float(df["short_strike_touched"].mean())

        print(
            f"Complete observations: {total_entries:,} | "
            f"Failures: {total_failures} ({baseline_failure_rate:.2%}) | "
            f"Touches: {total_touches} ({baseline_touch_rate:.2%})"
        )

        results: list[dict[str, Any]] = []

        for factor in factors:
            condition = evaluate_factor(df, factor)

            in_group = df.loc[condition]
            out_group = df.loc[~condition]

            observations = len(in_group)
            out_observations = len(out_group)

            failures = int(in_group["expiration_failure"].sum())
            out_failures = int(out_group["expiration_failure"].sum())

            touches = int(in_group["short_strike_touched"].sum())
            clustered_failures = int(
                in_group["clustered_expiration_failure"].sum()
            )

            failure_rate = safe_divide(failures, observations)
            out_failure_rate = safe_divide(out_failures, out_observations)
            touch_rate = safe_divide(touches, observations)

            observation_share = safe_divide(observations, total_entries)
            failure_capture = safe_divide(failures, total_failures)

            failure_rate_lift = safe_divide(
                failure_rate,
                baseline_failure_rate,
            )

            failure_capture_efficiency = safe_divide(
                failure_capture,
                observation_share,
            )

            lower_ci, upper_ci = wilson_interval(
                failures,
                observations,
            )

            contingency = [
                [failures, observations - failures],
                [out_failures, out_observations - out_failures],
            ]

            odds_ratio, fisher_p_value = fisher_exact(
                contingency,
                alternative="two-sided",
            )

            subperiod_values: dict[str, Any] = {}
            higher_risk_subperiods = 0

            for period_label, start_date, end_date in SUBPERIODS:
                period_mask = df["entry_date"].between(
                    pd.Timestamp(start_date),
                    pd.Timestamp(end_date),
                )

                period_in = df.loc[period_mask & condition]
                period_out = df.loc[period_mask & ~condition]

                period_in_failures = int(
                    period_in["expiration_failure"].sum()
                )

                period_out_failures = int(
                    period_out["expiration_failure"].sum()
                )

                period_in_rate = safe_divide(
                    period_in_failures,
                    len(period_in),
                )

                period_out_rate = safe_divide(
                    period_out_failures,
                    len(period_out),
                )

                if (
                    not pd.isna(period_in_rate)
                    and not pd.isna(period_out_rate)
                    and period_in_rate > period_out_rate
                ):
                    higher_risk_subperiods += 1

                period_key = period_label.replace("–", "_")

                subperiod_values[f"{period_key}_n"] = len(period_in)
                subperiod_values[
                    f"{period_key}_failures"
                ] = period_in_failures
                subperiod_values[
                    f"{period_key}_failure_rate"
                ] = period_in_rate

            result: dict[str, Any] = {
                "factor_id": factor["id"],
                "factor_label": factor["label"],
                "category": factor["category"],
                "observations": observations,
                "observation_share": observation_share,
                "touches": touches,
                "touch_rate": touch_rate,
                "touch_rate_lift": safe_divide(
                    touch_rate,
                    baseline_touch_rate,
                ),
                "failures": failures,
                "failure_rate": failure_rate,
                "failure_rate_ci_lower": lower_ci,
                "failure_rate_ci_upper": upper_ci,
                "failure_rate_lift": failure_rate_lift,
                "complement_failure_rate": out_failure_rate,
                "failure_capture": failure_capture,
                "failure_capture_efficiency": (
                    failure_capture_efficiency
                ),
                "clustered_failures": clustered_failures,
                "clustered_share_of_failures": safe_divide(
                    clustered_failures,
                    failures,
                ),
                "average_expiration_return": float(
                    in_group["expiration_return"].mean()
                ),
                "fifth_percentile_expiration_return": float(
                    in_group["expiration_return"].quantile(0.05)
                ),
                "first_percentile_expiration_return": float(
                    in_group["expiration_return"].quantile(0.01)
                ),
                "fifth_percentile_maximum_adverse_return": float(
                    in_group["maximum_adverse_return"].quantile(0.05)
                ),
                "odds_ratio": float(odds_ratio),
                "fisher_p_value": float(fisher_p_value),
                "higher_risk_subperiods": higher_risk_subperiods,
                **subperiod_values,
            }

            result["classification"] = classify_factor(result)
            results.append(result)

        results_df = pd.DataFrame(results)

        classification_rank = {
            "Candidate risk flag": 0,
            "Watchlist": 1,
            "No clear individual edge": 2,
            "Insufficient failure sample": 3,
        }

        results_df["classification_rank"] = (
            results_df["classification"].map(classification_rank)
        )

        results_df = results_df.sort_values(
            [
                "classification_rank",
                "failure_rate_lift",
                "failure_capture_efficiency",
                "observations",
            ],
            ascending=[True, False, False, False],
        ).reset_index(drop=True)

        results_df.insert(
            0,
            "screen_rank",
            np.arange(1, len(results_df) + 1),
        )

        results_df = results_df.drop(
            columns=["classification_rank"]
        )

        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        results_df.to_csv(CSV_OUTPUT, index=False)

        connection.register(
            "factor_screening_dataframe",
            results_df,
        )

        connection.execute(
            """
            CREATE OR REPLACE TABLE
                pcs.factor_screening_results AS
            SELECT *
            FROM factor_screening_dataframe
            ORDER BY screen_rank
            """
        )

        candidate_count = int(
            (results_df["classification"] == "Candidate risk flag").sum()
        )

        watchlist_count = int(
            (results_df["classification"] == "Watchlist").sum()
        )

        summary_lines = [
            "MNQ PCS REGIME RESEARCH — INDIVIDUAL FACTOR SCREENING",
            "=" * 58,
            "",
            f"Observations: {total_entries:,}",
            f"Baseline failures: {total_failures} ({baseline_failure_rate:.2%})",
            f"Baseline touches: {total_touches} ({baseline_touch_rate:.2%})",
            f"Candidate risk flags: {candidate_count}",
            f"Watchlist factors: {watchlist_count}",
            "",
            "Top-ranked factors",
            "-" * 58,
        ]

        top_results = results_df.head(10)

        for row in top_results.itertuples(index=False):
            summary_lines.append(
                f"{row.screen_rank:>2}. {row.factor_label} | "
                f"{row.classification} | "
                f"N={row.observations} | "
                f"Failures={row.failures} "
                f"({row.failure_rate:.2%}) | "
                f"Lift={row.failure_rate_lift:.2f}x | "
                f"Captured={row.failure_capture:.1%}"
            )

        SUMMARY_OUTPUT.write_text(
            "\n".join(summary_lines) + "\n",
            encoding="utf-8",
        )

        print("\nTop-ranked factors")
        print("-" * 110)

        for row in top_results.itertuples(index=False):
            print(
                f"{row.screen_rank:>2}. "
                f"{row.factor_label:<48} | "
                f"{row.classification:<27} | "
                f"N={row.observations:>4} | "
                f"Fail={row.failures:>2} "
                f"({row.failure_rate:>6.2%}) | "
                f"Lift={row.failure_rate_lift:>4.2f}x"
            )

        print(f"\nCandidate risk flags: {candidate_count}")
        print(f"Watchlist factors: {watchlist_count}")
        print(f"CSV output: {CSV_OUTPUT}")
        print(f"Summary output: {SUMMARY_OUTPUT}")
        print("DuckDB table: pcs.factor_screening_results")
        print("Individual factor screening completed successfully.")

        return 0

    except Exception as exc:
        print(f"\nERROR: {exc}")
        return 1

    finally:
        connection.close()


if __name__ == "__main__":
    sys.exit(main())

"""Validate the candidate Friday PCS regime score."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import duckdb
import numpy as np
import pandas as pd
import yaml
from scipy.stats import fisher_exact


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATABASE_PATH = PROJECT_ROOT / "database" / "research.duckdb"
CONFIG_PATH = PROJECT_ROOT / "config" / "candidate_regime_score.yaml"

RESULTS_DIR = PROJECT_ROOT / "results" / "regime_validation"
SUBPERIOD_OUTPUT = RESULTS_DIR / "subperiod_validation.csv"
CLUSTER_OUTPUT = RESULTS_DIR / "leave_one_cluster_out.csv"
SUMMARY_OUTPUT = RESULTS_DIR / "regime_validation_summary.txt"


def safe_divide(numerator: float, denominator: float) -> float:
    """Return a ratio without raising on a zero denominator."""

    if denominator == 0 or pd.isna(denominator):
        return np.nan

    return numerator / denominator


def evaluate_rule(
    series: pd.Series,
    operator: str,
    value: Any,
) -> pd.Series:
    """Evaluate one configured regime condition."""

    operations = {
        "lt": series < value,
        "le": series <= value,
        "gt": series > value,
        "ge": series >= value,
        "eq": series == value,
        "ne": series != value,
    }

    if operator not in operations:
        raise ValueError(f"Unsupported operator: {operator}")

    return operations[operator].fillna(False)


def summarize_regimes(
    dataframe: pd.DataFrame,
    normal_maximum: int,
    elevated_minimum: int,
) -> dict[str, Any]:
    """Summarize normal and elevated regime outcomes."""

    normal = dataframe[
        dataframe["candidate_score"] <= normal_maximum
    ]

    elevated = dataframe[
        dataframe["candidate_score"] >= elevated_minimum
    ]

    normal_failures = int(normal["expiration_failure"].sum())
    elevated_failures = int(elevated["expiration_failure"].sum())

    normal_rate = safe_divide(normal_failures, len(normal))
    elevated_rate = safe_divide(elevated_failures, len(elevated))

    contingency = [
        [elevated_failures, len(elevated) - elevated_failures],
        [normal_failures, len(normal) - normal_failures],
    ]

    if len(normal) and len(elevated):
        odds_ratio, fisher_p_value = fisher_exact(
            contingency,
            alternative="two-sided",
        )
    else:
        odds_ratio = np.nan
        fisher_p_value = np.nan

    return {
        "total_observations": len(dataframe),
        "total_failures": int(
            dataframe["expiration_failure"].sum()
        ),
        "normal_observations": len(normal),
        "normal_failures": normal_failures,
        "normal_failure_rate": normal_rate,
        "elevated_observations": len(elevated),
        "elevated_failures": elevated_failures,
        "elevated_failure_rate": elevated_rate,
        "elevated_rate_lift_vs_normal": safe_divide(
            elevated_rate,
            normal_rate,
        ),
        "elevated_failure_capture": safe_divide(
            elevated_failures,
            int(dataframe["expiration_failure"].sum()),
        ),
        "odds_ratio": float(odds_ratio),
        "fisher_p_value": float(fisher_p_value),
    }


def build_failure_clusters(
    failure_rows: pd.DataFrame,
) -> list[dict[str, Any]]:
    """Group overlapping failure positions into market-stress clusters."""

    failures = failure_rows.sort_values(
        ["entry_date", "expiration_proxy_date"]
    ).reset_index(drop=True)

    clusters: list[dict[str, Any]] = []
    current_indexes: list[int] = []
    current_end: pd.Timestamp | None = None

    for index, row in failures.iterrows():
        entry_date = pd.Timestamp(row["entry_date"])
        expiration_date = pd.Timestamp(
            row["expiration_proxy_date"]
        )

        if current_end is None or entry_date > current_end:
            if current_indexes:
                cluster = failures.loc[current_indexes]

                clusters.append(
                    {
                        "cluster_id": len(clusters) + 1,
                        "cluster_start": cluster[
                            "entry_date"
                        ].min(),
                        "cluster_end": cluster[
                            "expiration_proxy_date"
                        ].max(),
                        "failure_entries": len(cluster),
                        "failure_entry_dates": ", ".join(
                            cluster["entry_date"]
                            .dt.strftime("%Y-%m-%d")
                            .tolist()
                        ),
                    }
                )

            current_indexes = [index]
            current_end = expiration_date
        else:
            current_indexes.append(index)
            current_end = max(current_end, expiration_date)

    if current_indexes:
        cluster = failures.loc[current_indexes]

        clusters.append(
            {
                "cluster_id": len(clusters) + 1,
                "cluster_start": cluster["entry_date"].min(),
                "cluster_end": cluster[
                    "expiration_proxy_date"
                ].max(),
                "failure_entries": len(cluster),
                "failure_entry_dates": ", ".join(
                    cluster["entry_date"]
                    .dt.strftime("%Y-%m-%d")
                    .tolist()
                ),
            }
        )

    return clusters


def main() -> int:
    """Run subperiod and leave-one-cluster-out validation."""

    print("MNQ PCS Regime Research — Candidate Score Validation")
    print("=" * 56)

    if not DATABASE_PATH.is_file():
        print(f"ERROR: Missing database: {DATABASE_PATH}")
        return 1

    if not CONFIG_PATH.is_file():
        print(f"ERROR: Missing configuration: {CONFIG_PATH}")
        return 1

    with CONFIG_PATH.open("r", encoding="utf-8") as file_handle:
        config = yaml.safe_load(file_handle)

    factors = config["factors"]
    normal_maximum = int(
        config["regimes"]["normal"]["maximum_score"]
    )
    elevated_minimum = int(
        config["regimes"]["elevated"]["minimum_score"]
    )

    connection = duckdb.connect(str(DATABASE_PATH))

    try:
        dataframe = connection.execute(
            """
            SELECT *
            FROM pcs.friday_regime_features
            WHERE complete_core_features = TRUE
            ORDER BY entry_date
            """
        ).fetchdf()

        dataframe["entry_date"] = pd.to_datetime(
            dataframe["entry_date"]
        )

        dataframe["expiration_proxy_date"] = pd.to_datetime(
            dataframe["expiration_proxy_date"]
        )

        dataframe["expiration_failure"] = (
            dataframe["expiration_failure"]
            .fillna(False)
            .astype(bool)
        )

        factor_columns: list[str] = []

        for factor in factors:
            column_name = f"flag_{factor['id']}"

            if factor["column"] not in dataframe.columns:
                raise ValueError(
                    f"Missing feature column: {factor['column']}"
                )

            dataframe[column_name] = evaluate_rule(
                dataframe[factor["column"]],
                str(factor["operator"]),
                factor["value"],
            )

            factor_columns.append(column_name)

        dataframe["candidate_score"] = (
            dataframe[factor_columns]
            .astype(int)
            .sum(axis=1)
        )

        overall = summarize_regimes(
            dataframe,
            normal_maximum,
            elevated_minimum,
        )

        print(
            f"Complete observations: "
            f"{overall['total_observations']:,}"
        )

        print(
            f"Total failures: {overall['total_failures']} "
            f"({overall['total_failures'] / overall['total_observations']:.2%})"
        )

        print("\nOverall regime comparison")
        print("-" * 86)

        print(
            f"Normal, score 0–{normal_maximum}: "
            f"N={overall['normal_observations']} | "
            f"Failures={overall['normal_failures']} "
            f"({overall['normal_failure_rate']:.2%})"
        )

        print(
            f"Elevated, score {elevated_minimum}–3: "
            f"N={overall['elevated_observations']} | "
            f"Failures={overall['elevated_failures']} "
            f"({overall['elevated_failure_rate']:.2%}) | "
            f"Lift={overall['elevated_rate_lift_vs_normal']:.2f}x | "
            f"Fisher p={overall['fisher_p_value']:.4f}"
        )

        subperiod_rows: list[dict[str, Any]] = []

        print("\nHistorical subperiod validation")
        print("-" * 104)

        for period in config["subperiods"]:
            start_date = pd.Timestamp(period["start_date"])
            end_date = pd.Timestamp(period["end_date"])

            period_data = dataframe[
                dataframe["entry_date"].between(
                    start_date,
                    end_date,
                )
            ]

            result = summarize_regimes(
                period_data,
                normal_maximum,
                elevated_minimum,
            )

            result["period"] = period["label"]
            result["start_date"] = start_date
            result["end_date"] = end_date

            subperiod_rows.append(result)

            print(
                f"{period['label']} | "
                f"Normal: N={result['normal_observations']:>3}, "
                f"Fail={result['normal_failures']:>2} "
                f"({result['normal_failure_rate']:>6.2%}) | "
                f"Elevated: N={result['elevated_observations']:>3}, "
                f"Fail={result['elevated_failures']:>2} "
                f"({result['elevated_failure_rate']:>6.2%}) | "
                f"Lift={result['elevated_rate_lift_vs_normal']:>5.2f}x | "
                f"p={result['fisher_p_value']:.4f}"
            )

        subperiods = pd.DataFrame(subperiod_rows)

        failure_rows = dataframe[
            dataframe["expiration_failure"]
        ].copy()

        clusters = build_failure_clusters(failure_rows)

        cluster_rows: list[dict[str, Any]] = []

        print("\nLeave-one-failure-cluster-out validation")
        print("-" * 112)

        for cluster in clusters:
            overlap_mask = (
                dataframe["entry_date"]
                <= cluster["cluster_end"]
            ) & (
                dataframe["expiration_proxy_date"]
                >= cluster["cluster_start"]
            )

            reduced = dataframe.loc[~overlap_mask].copy()

            result = summarize_regimes(
                reduced,
                normal_maximum,
                elevated_minimum,
            )

            result.update(cluster)
            result["removed_observations"] = int(
                overlap_mask.sum()
            )

            cluster_rows.append(result)

            print(
                f"Cluster {cluster['cluster_id']:>2} "
                f"{cluster['cluster_start'].date()} to "
                f"{cluster['cluster_end'].date()} | "
                f"Removed N={result['removed_observations']:>3} | "
                f"Normal={result['normal_failure_rate']:>6.2%} | "
                f"Elevated={result['elevated_failure_rate']:>6.2%} | "
                f"Lift={result['elevated_rate_lift_vs_normal']:>5.2f}x"
            )

        cluster_results = pd.DataFrame(cluster_rows)

        RESULTS_DIR.mkdir(parents=True, exist_ok=True)

        subperiods.to_csv(
            SUBPERIOD_OUTPUT,
            index=False,
        )

        cluster_results.to_csv(
            CLUSTER_OUTPUT,
            index=False,
        )

        higher_in_subperiods = int(
            (
                subperiods["elevated_failure_rate"]
                > subperiods["normal_failure_rate"]
            ).sum()
        )

        higher_after_cluster_removal = int(
            (
                cluster_results["elevated_failure_rate"]
                > cluster_results["normal_failure_rate"]
            ).sum()
        )

        minimum_cluster_lift = float(
            cluster_results[
                "elevated_rate_lift_vs_normal"
            ].min()
        )

        median_cluster_lift = float(
            cluster_results[
                "elevated_rate_lift_vs_normal"
            ].median()
        )

        summary_lines = [
            "MNQ PCS REGIME RESEARCH — CANDIDATE SCORE VALIDATION",
            "=" * 58,
            "",
            (
                f"Overall normal failure rate: "
                f"{overall['normal_failure_rate']:.2%}"
            ),
            (
                f"Overall elevated failure rate: "
                f"{overall['elevated_failure_rate']:.2%}"
            ),
            (
                f"Overall elevated lift: "
                f"{overall['elevated_rate_lift_vs_normal']:.2f}x"
            ),
            (
                f"Overall Fisher p-value: "
                f"{overall['fisher_p_value']:.4f}"
            ),
            "",
            (
                f"Subperiods with elevated rate above normal: "
                f"{higher_in_subperiods}/{len(subperiods)}"
            ),
            (
                f"Cluster-removal tests with elevated rate above normal: "
                f"{higher_after_cluster_removal}/{len(cluster_results)}"
            ),
            (
                f"Minimum leave-one-cluster-out lift: "
                f"{minimum_cluster_lift:.2f}x"
            ),
            (
                f"Median leave-one-cluster-out lift: "
                f"{median_cluster_lift:.2f}x"
            ),
        ]

        SUMMARY_OUTPUT.write_text(
            "\n".join(summary_lines) + "\n",
            encoding="utf-8",
        )

        connection.register(
            "subperiod_validation_dataframe",
            subperiods,
        )

        connection.register(
            "cluster_validation_dataframe",
            cluster_results,
        )

        connection.execute(
            """
            CREATE OR REPLACE TABLE
                pcs.regime_subperiod_validation AS
            SELECT *
            FROM subperiod_validation_dataframe
            """
        )

        connection.execute(
            """
            CREATE OR REPLACE TABLE
                pcs.regime_cluster_validation AS
            SELECT *
            FROM cluster_validation_dataframe
            """
        )

        print("\nValidation summary")
        print("-" * 72)

        print(
            f"Elevated risk exceeded normal risk in "
            f"{higher_in_subperiods}/{len(subperiods)} subperiods."
        )

        print(
            f"Elevated risk exceeded normal risk in "
            f"{higher_after_cluster_removal}/"
            f"{len(cluster_results)} cluster-removal tests."
        )

        print(
            f"Leave-one-cluster-out lift range: "
            f"{minimum_cluster_lift:.2f}x minimum; "
            f"{median_cluster_lift:.2f}x median."
        )

        print(f"\nSubperiod output: {SUBPERIOD_OUTPUT}")
        print(f"Cluster output: {CLUSTER_OUTPUT}")
        print(f"Summary output: {SUMMARY_OUTPUT}")
        print("Candidate score validation completed successfully.")

        return 0

    except Exception as exc:
        print(f"\nERROR: {exc}")
        return 1

    finally:
        connection.close()


if __name__ == "__main__":
    sys.exit(main())

"""Analyze overlap and independence among candidate PCS regime factors."""

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


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATABASE_PATH = PROJECT_ROOT / "database" / "research.duckdb"
FACTOR_CONFIG_PATH = PROJECT_ROOT / "config" / "factor_definitions.yaml"

RESULTS_DIR = PROJECT_ROOT / "results" / "overlap_analysis"
PAIRWISE_OUTPUT = RESULTS_DIR / "candidate_pairwise_overlap.csv"
COMBINATION_OUTPUT = RESULTS_DIR / "candidate_combinations.csv"
SCORE_OUTPUT = RESULTS_DIR / "candidate_flag_scores.csv"
CLUSTER_OUTPUT = RESULTS_DIR / "failure_clusters.csv"
SUMMARY_OUTPUT = RESULTS_DIR / "overlap_analysis_summary.txt"


def safe_divide(numerator: float, denominator: float) -> float:
    """Return a safe ratio."""

    if denominator == 0 or pd.isna(denominator):
        return np.nan

    return numerator / denominator


def evaluate_rule(
    series: pd.Series,
    operator: str,
    value: Any,
) -> pd.Series:
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


def evaluate_factor(
    dataframe: pd.DataFrame,
    factor: dict[str, Any],
) -> pd.Series:
    """Evaluate every rule belonging to one factor."""

    condition = pd.Series(True, index=dataframe.index)

    for rule in factor["all"]:
        column = rule["column"]

        if column not in dataframe.columns:
            raise ValueError(
                f"Factor '{factor['id']}' references missing column "
                f"'{column}'."
            )

        condition &= evaluate_rule(
            dataframe[column],
            str(rule["operator"]),
            rule["value"],
        )

    return condition


def phi_coefficient(
    first: pd.Series,
    second: pd.Series,
) -> float:
    """Calculate the phi correlation between two Boolean conditions."""

    first_numeric = first.astype(int)
    second_numeric = second.astype(int)

    if (
        first_numeric.nunique() < 2
        or second_numeric.nunique() < 2
    ):
        return np.nan

    return float(first_numeric.corr(second_numeric))


def summarize_group(
    group: pd.DataFrame,
    total_entries: int,
    total_failures: int,
    baseline_failure_rate: float,
) -> dict[str, Any]:
    """Create common outcome statistics for one regime group."""

    observations = len(group)
    failures = int(group["expiration_failure"].sum())
    touches = int(group["short_strike_touched"].sum())

    failure_rate = safe_divide(failures, observations)
    touch_rate = safe_divide(touches, observations)

    return {
        "observations": observations,
        "observation_share": safe_divide(
            observations,
            total_entries,
        ),
        "failures": failures,
        "failure_rate": failure_rate,
        "failure_rate_lift": safe_divide(
            failure_rate,
            baseline_failure_rate,
        ),
        "failure_capture": safe_divide(
            failures,
            total_failures,
        ),
        "touches": touches,
        "touch_rate": touch_rate,
        "clustered_failures": int(
            group["clustered_expiration_failure"].sum()
        ),
        "average_expiration_return": float(
            group["expiration_return"].mean()
        ) if observations else np.nan,
        "fifth_percentile_expiration_return": float(
            group["expiration_return"].quantile(0.05)
        ) if observations else np.nan,
        "first_percentile_expiration_return": float(
            group["expiration_return"].quantile(0.01)
        ) if observations else np.nan,
        "fifth_percentile_maximum_adverse_return": float(
            group["maximum_adverse_return"].quantile(0.05)
        ) if observations else np.nan,
    }


def build_failure_clusters(
    failure_rows: pd.DataFrame,
    candidate_ids: list[str],
    candidate_labels: dict[str, str],
) -> pd.DataFrame:
    """Group overlapping failure trades into connected clusters."""

    if failure_rows.empty:
        return pd.DataFrame()

    failures = failure_rows.sort_values(
        ["entry_date", "expiration_proxy_date"]
    ).reset_index(drop=True)

    clusters: list[list[int]] = []
    current_cluster: list[int] = []
    current_cluster_end: pd.Timestamp | None = None

    for index, row in failures.iterrows():
        entry_date = pd.Timestamp(row["entry_date"])
        expiration_date = pd.Timestamp(
            row["expiration_proxy_date"]
        )

        if (
            current_cluster_end is None
            or entry_date > current_cluster_end
        ):
            if current_cluster:
                clusters.append(current_cluster)

            current_cluster = [index]
            current_cluster_end = expiration_date
        else:
            current_cluster.append(index)
            current_cluster_end = max(
                current_cluster_end,
                expiration_date,
            )

    if current_cluster:
        clusters.append(current_cluster)

    output_rows: list[dict[str, Any]] = []

    for cluster_number, indexes in enumerate(clusters, start=1):
        cluster = failures.loc[indexes].copy()

        signatures = cluster["regime_signature"].tolist()
        active_scores = cluster["candidate_flag_count"].tolist()

        row: dict[str, Any] = {
            "cluster_id": cluster_number,
            "cluster_start_entry": cluster["entry_date"].min(),
            "cluster_last_entry": cluster["entry_date"].max(),
            "cluster_end_expiration": (
                cluster["expiration_proxy_date"].max()
            ),
            "failure_entries": len(cluster),
            "entry_dates": ", ".join(
                cluster["entry_date"]
                .dt.strftime("%Y-%m-%d")
                .tolist()
            ),
            "regime_signatures": " | ".join(signatures),
            "maximum_active_flags": int(max(active_scores)),
            "average_active_flags": float(
                np.mean(active_scores)
            ),
            "all_failures_flagged": bool(
                (cluster["candidate_flag_count"] > 0).all()
            ),
            "any_failure_flagged": bool(
                (cluster["candidate_flag_count"] > 0).any()
            ),
        }

        for factor_id in candidate_ids:
            row[f"any_{factor_id}"] = bool(
                cluster[factor_id].any()
            )

            row[f"count_{factor_id}"] = int(
                cluster[factor_id].sum()
            )

            row[f"label_{factor_id}"] = candidate_labels[
                factor_id
            ]

        output_rows.append(row)

    return pd.DataFrame(output_rows)


def main() -> int:
    """Run candidate overlap and independence analysis."""

    print("MNQ PCS Regime Research — Candidate Overlap Analysis")
    print("=" * 55)

    if not DATABASE_PATH.is_file():
        print(f"ERROR: Missing database: {DATABASE_PATH}")
        return 1

    if not FACTOR_CONFIG_PATH.is_file():
        print(
            f"ERROR: Missing factor definitions: "
            f"{FACTOR_CONFIG_PATH}"
        )
        return 1

    with FACTOR_CONFIG_PATH.open(
        "r",
        encoding="utf-8",
    ) as file_handle:
        factor_config = yaml.safe_load(file_handle)

    configured_factors = {
        factor["id"]: factor
        for factor in factor_config.get("factors", [])
    }

    connection = duckdb.connect(str(DATABASE_PATH))

    try:
        features = connection.execute(
            """
            SELECT *
            FROM pcs.friday_regime_features
            WHERE complete_core_features = TRUE
            ORDER BY entry_date
            """
        ).fetchdf()

        screening = connection.execute(
            """
            SELECT
                factor_id,
                factor_label,
                classification,
                screen_rank
            FROM pcs.factor_screening_results
            WHERE classification = 'Candidate risk flag'
            ORDER BY screen_rank
            """
        ).fetchdf()

        if len(screening) < 2:
            raise ValueError(
                "At least two candidate factors are required."
            )

        features["entry_date"] = pd.to_datetime(
            features["entry_date"]
        )

        features["expiration_proxy_date"] = pd.to_datetime(
            features["expiration_proxy_date"]
        )

        boolean_columns = (
            "expiration_failure",
            "short_strike_touched",
            "clustered_expiration_failure",
        )

        for column in boolean_columns:
            features[column] = (
                features[column]
                .fillna(False)
                .astype(bool)
            )

        candidate_ids = screening["factor_id"].tolist()

        candidate_labels = dict(
            zip(
                screening["factor_id"],
                screening["factor_label"],
            )
        )

        for factor_id in candidate_ids:
            if factor_id not in configured_factors:
                raise ValueError(
                    f"Candidate factor '{factor_id}' is missing "
                    "from factor_definitions.yaml."
                )

            features[factor_id] = evaluate_factor(
                features,
                configured_factors[factor_id],
            )

        total_entries = len(features)
        total_failures = int(
            features["expiration_failure"].sum()
        )

        baseline_failure_rate = float(
            features["expiration_failure"].mean()
        )

        features["candidate_flag_count"] = (
            features[candidate_ids]
            .astype(int)
            .sum(axis=1)
        )

        def signature_for_row(row: pd.Series) -> str:
            active = [
                candidate_labels[factor_id]
                for factor_id in candidate_ids
                if bool(row[factor_id])
            ]

            return "None" if not active else " + ".join(active)

        features["regime_signature"] = features.apply(
            signature_for_row,
            axis=1,
        )

        print(
            f"Complete observations: {total_entries:,} | "
            f"Failures: {total_failures} "
            f"({baseline_failure_rate:.2%})"
        )

        print("\nCandidate factors")
        print("-" * 70)

        for factor_id in candidate_ids:
            factor_count = int(features[factor_id].sum())
            factor_failures = int(
                features.loc[
                    features[factor_id],
                    "expiration_failure",
                ].sum()
            )

            print(
                f" - {candidate_labels[factor_id]} | "
                f"N={factor_count} | "
                f"Failures={factor_failures}"
            )

        pairwise_rows: list[dict[str, Any]] = []

        for first_index, first_id in enumerate(candidate_ids):
            for second_id in candidate_ids[first_index + 1:]:
                first_mask = features[first_id]
                second_mask = features[second_id]

                both_mask = first_mask & second_mask
                first_only_mask = first_mask & ~second_mask
                second_only_mask = second_mask & ~first_mask
                neither_mask = ~first_mask & ~second_mask

                first_failures = int(
                    features.loc[
                        first_mask,
                        "expiration_failure",
                    ].sum()
                )

                second_failures = int(
                    features.loc[
                        second_mask,
                        "expiration_failure",
                    ].sum()
                )

                both_failures = int(
                    features.loc[
                        both_mask,
                        "expiration_failure",
                    ].sum()
                )

                both_summary = summarize_group(
                    features.loc[both_mask],
                    total_entries,
                    total_failures,
                    baseline_failure_rate,
                )

                first_only_summary = summarize_group(
                    features.loc[first_only_mask],
                    total_entries,
                    total_failures,
                    baseline_failure_rate,
                )

                second_only_summary = summarize_group(
                    features.loc[second_only_mask],
                    total_entries,
                    total_failures,
                    baseline_failure_rate,
                )

                neither_summary = summarize_group(
                    features.loc[neither_mask],
                    total_entries,
                    total_failures,
                    baseline_failure_rate,
                )

                both_observations = int(both_mask.sum())
                union_observations = int(
                    (first_mask | second_mask).sum()
                )

                contingency = [
                    [
                        both_summary["failures"],
                        (
                            both_summary["observations"]
                            - both_summary["failures"]
                        ),
                    ],
                    [
                        neither_summary["failures"],
                        (
                            neither_summary["observations"]
                            - neither_summary["failures"]
                        ),
                    ],
                ]

                _, fisher_p_value = fisher_exact(
                    contingency,
                    alternative="two-sided",
                )

                pairwise_rows.append(
                    {
                        "factor_a_id": first_id,
                        "factor_a_label": candidate_labels[
                            first_id
                        ],
                        "factor_b_id": second_id,
                        "factor_b_label": candidate_labels[
                            second_id
                        ],
                        "factor_a_observations": int(
                            first_mask.sum()
                        ),
                        "factor_b_observations": int(
                            second_mask.sum()
                        ),
                        "both_observations": both_observations,
                        "both_share_of_factor_a": safe_divide(
                            both_observations,
                            int(first_mask.sum()),
                        ),
                        "both_share_of_factor_b": safe_divide(
                            both_observations,
                            int(second_mask.sum()),
                        ),
                        "jaccard_overlap": safe_divide(
                            both_observations,
                            union_observations,
                        ),
                        "phi_correlation": phi_coefficient(
                            first_mask,
                            second_mask,
                        ),
                        "factor_a_failures": first_failures,
                        "factor_b_failures": second_failures,
                        "shared_failures": both_failures,
                        "shared_failure_share_a": safe_divide(
                            both_failures,
                            first_failures,
                        ),
                        "shared_failure_share_b": safe_divide(
                            both_failures,
                            second_failures,
                        ),
                        "both_failure_rate": both_summary[
                            "failure_rate"
                        ],
                        "factor_a_only_failure_rate": (
                            first_only_summary["failure_rate"]
                        ),
                        "factor_b_only_failure_rate": (
                            second_only_summary["failure_rate"]
                        ),
                        "neither_failure_rate": neither_summary[
                            "failure_rate"
                        ],
                        "both_failure_rate_lift": both_summary[
                            "failure_rate_lift"
                        ],
                        "both_failure_capture": both_summary[
                            "failure_capture"
                        ],
                        "both_vs_neither_fisher_p": float(
                            fisher_p_value
                        ),
                    }
                )

        pairwise = pd.DataFrame(pairwise_rows)

        exact_combination_rows: list[dict[str, Any]] = []

        for signature, group in features.groupby(
            "regime_signature",
            sort=False,
        ):
            summary = summarize_group(
                group,
                total_entries,
                total_failures,
                baseline_failure_rate,
            )

            exact_combination_rows.append(
                {
                    "regime_signature": signature,
                    "active_flag_count": int(
                        group["candidate_flag_count"].iloc[0]
                    ),
                    **summary,
                }
            )

        combinations = pd.DataFrame(
            exact_combination_rows
        ).sort_values(
            [
                "active_flag_count",
                "failure_rate",
                "observations",
            ],
            ascending=[True, False, False],
        ).reset_index(drop=True)

        score_rows: list[dict[str, Any]] = []

        maximum_score = int(
            features["candidate_flag_count"].max()
        )

        for score in range(maximum_score + 1):
            group = features[
                features["candidate_flag_count"] == score
            ]

            summary = summarize_group(
                group,
                total_entries,
                total_failures,
                baseline_failure_rate,
            )

            score_rows.append(
                {
                    "candidate_flag_count": score,
                    **summary,
                }
            )

        scores = pd.DataFrame(score_rows)

        failure_rows = features[
            features["expiration_failure"]
        ].copy()

        clusters = build_failure_clusters(
            failure_rows,
            candidate_ids,
            candidate_labels,
        )

        RESULTS_DIR.mkdir(
            parents=True,
            exist_ok=True,
        )

        pairwise.to_csv(
            PAIRWISE_OUTPUT,
            index=False,
        )

        combinations.to_csv(
            COMBINATION_OUTPUT,
            index=False,
        )

        scores.to_csv(
            SCORE_OUTPUT,
            index=False,
        )

        clusters.to_csv(
            CLUSTER_OUTPUT,
            index=False,
        )

        connection.register(
            "candidate_pairwise_dataframe",
            pairwise,
        )

        connection.register(
            "candidate_combinations_dataframe",
            combinations,
        )

        connection.register(
            "candidate_scores_dataframe",
            scores,
        )

        connection.register(
            "failure_clusters_dataframe",
            clusters,
        )

        connection.execute(
            """
            CREATE OR REPLACE TABLE
                pcs.candidate_pairwise_overlap AS
            SELECT *
            FROM candidate_pairwise_dataframe
            """
        )

        connection.execute(
            """
            CREATE OR REPLACE TABLE
                pcs.candidate_combinations AS
            SELECT *
            FROM candidate_combinations_dataframe
            """
        )

        connection.execute(
            """
            CREATE OR REPLACE TABLE
                pcs.candidate_flag_scores AS
            SELECT *
            FROM candidate_scores_dataframe
            ORDER BY candidate_flag_count
            """
        )

        connection.execute(
            """
            CREATE OR REPLACE TABLE
                pcs.failure_clusters AS
            SELECT *
            FROM failure_clusters_dataframe
            ORDER BY cluster_id
            """
        )

        summary_lines = [
            "MNQ PCS REGIME RESEARCH — CANDIDATE OVERLAP ANALYSIS",
            "=" * 59,
            "",
            f"Complete observations: {total_entries:,}",
            (
                f"Baseline failures: {total_failures} "
                f"({baseline_failure_rate:.2%})"
            ),
            f"Candidate factors tested: {len(candidate_ids)}",
            "",
            "Failure rate by number of active candidate flags",
            "-" * 59,
        ]

        for row in scores.itertuples(index=False):
            summary_lines.append(
                f"{row.candidate_flag_count} active | "
                f"N={row.observations} | "
                f"Failures={row.failures} "
                f"({row.failure_rate:.2%}) | "
                f"Lift={row.failure_rate_lift:.2f}x | "
                f"Captured={row.failure_capture:.1%}"
            )

        summary_lines.extend(
            [
                "",
                "Pairwise overlap",
                "-" * 59,
            ]
        )

        for row in pairwise.itertuples(index=False):
            summary_lines.append(
                f"{row.factor_a_label} + "
                f"{row.factor_b_label} | "
                f"N={row.both_observations} | "
                f"Failures={row.shared_failures} | "
                f"Failure rate={row.both_failure_rate:.2%} | "
                f"Jaccard={row.jaccard_overlap:.2f} | "
                f"Phi={row.phi_correlation:.2f}"
            )

        summary_lines.extend(
            [
                "",
                "Failure clusters",
                "-" * 59,
                (
                    f"Distinct failure clusters: "
                    f"{len(clusters)}"
                ),
                (
                    f"Clusters containing multiple failures: "
                    f"{int((clusters['failure_entries'] > 1).sum())}"
                ),
            ]
        )

        SUMMARY_OUTPUT.write_text(
            "\n".join(summary_lines) + "\n",
            encoding="utf-8",
        )

        print("\nFailure rate by active candidate flags")
        print("-" * 92)

        for row in scores.itertuples(index=False):
            print(
                f"{row.candidate_flag_count} active | "
                f"N={row.observations:>4} | "
                f"Failures={row.failures:>2} "
                f"({row.failure_rate:>6.2%}) | "
                f"Lift={row.failure_rate_lift:>4.2f}x | "
                f"Captured={row.failure_capture:>5.1%}"
            )

        print("\nPairwise overlap")
        print("-" * 118)

        for row in pairwise.itertuples(index=False):
            print(
                f"{row.factor_a_label:<47} + "
                f"{row.factor_b_label:<47} | "
                f"N={row.both_observations:>3} | "
                f"Fail={row.shared_failures:>2} "
                f"({row.both_failure_rate:>6.2%}) | "
                f"Phi={row.phi_correlation:>5.2f}"
            )

        print(
            f"\nDistinct failure clusters: {len(clusters)}"
        )

        print(
            "Clusters containing multiple overlapping failures: "
            f"{int((clusters['failure_entries'] > 1).sum())}"
        )

        print(f"\nPairwise output: {PAIRWISE_OUTPUT}")
        print(f"Combination output: {COMBINATION_OUTPUT}")
        print(f"Score output: {SCORE_OUTPUT}")
        print(f"Cluster output: {CLUSTER_OUTPUT}")
        print(f"Summary output: {SUMMARY_OUTPUT}")
        print("Candidate overlap analysis completed successfully.")

        return 0

    except Exception as exc:
        print(f"\nERROR: {exc}")
        return 1

    finally:
        connection.close()


if __name__ == "__main__":
    sys.exit(main())

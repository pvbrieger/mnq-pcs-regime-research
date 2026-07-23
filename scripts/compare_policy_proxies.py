"""Compare elevated-regime sizing policies using risk-exposure proxies.

This is not an options P&L backtest. Historical option credits, listed-strike
rounding, transaction costs, and the 150-point payoff profile are not available.
The analysis therefore measures how much market exposure, failure exposure,
touch exposure, adverse-move exposure, and clustered-failure exposure would
have been retained under each policy.

Inputs
------
- database/research.duckdb
- config/candidate_regime_score.yaml
- config/policy_proxy.yaml
- pcs.friday_regime_features

Outputs
-------
- results/policy_proxy/policy_proxy_results.csv
- results/policy_proxy/policy_cluster_details.csv
- results/policy_proxy/policy_proxy_summary.txt
- DuckDB tables:
    pcs.policy_proxy_results
    pcs.policy_cluster_details
"""

from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Any

import duckdb
import numpy as np
import pandas as pd
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATABASE_PATH = PROJECT_ROOT / "database" / "research.duckdb"
SCORE_CONFIG_PATH = (
    PROJECT_ROOT / "config" / "candidate_regime_score.yaml"
)
POLICY_CONFIG_PATH = (
    PROJECT_ROOT / "config" / "policy_proxy.yaml"
)

RESULTS_DIR = PROJECT_ROOT / "results" / "policy_proxy"
RESULTS_OUTPUT = RESULTS_DIR / "policy_proxy_results.csv"
CLUSTER_OUTPUT = RESULTS_DIR / "policy_cluster_details.csv"
SUMMARY_OUTPUT = RESULTS_DIR / "policy_proxy_summary.txt"


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
    """Evaluate one configured factor condition."""

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


def build_failure_clusters(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """Assign overlapping failed trades to connected clusters."""

    failures = dataframe.loc[
        dataframe["expiration_failure"]
    ].sort_values(
        ["entry_date", "expiration_proxy_date"]
    ).copy()

    if failures.empty:
        failures["failure_cluster_id"] = pd.Series(
            dtype="int64"
        )
        return failures

    cluster_ids: list[int] = []
    cluster_id = 0
    current_end: pd.Timestamp | None = None

    for row in failures.itertuples(index=False):
        entry_date = pd.Timestamp(row.entry_date)
        expiration_date = pd.Timestamp(
            row.expiration_proxy_date
        )

        if current_end is None or entry_date > current_end:
            cluster_id += 1
            current_end = expiration_date
        else:
            current_end = max(current_end, expiration_date)

        cluster_ids.append(cluster_id)

    failures["failure_cluster_id"] = cluster_ids
    return failures


def make_model_mask(
    dataframe: pd.DataFrame,
    model: dict[str, Any],
    factor_columns: dict[str, str],
) -> pd.Series:
    """Build the elevated-regime mask for one model definition."""

    model_type = str(model["type"])

    if model_type == "score_at_least":
        factor_ids = [
            str(factor_id)
            for factor_id in model["factor_ids"]
        ]

        missing = [
            factor_id
            for factor_id in factor_ids
            if factor_id not in factor_columns
        ]

        if missing:
            raise ValueError(
                f"Model '{model['id']}' references missing factors: "
                + ", ".join(missing)
            )

        score = sum(
            dataframe[factor_columns[factor_id]]
            .astype(int)
            for factor_id in factor_ids
        )

        return score >= int(model["minimum_score"])

    if model_type == "all_factors":
        factor_ids = [
            str(factor_id)
            for factor_id in model["factor_ids"]
        ]

        missing = [
            factor_id
            for factor_id in factor_ids
            if factor_id not in factor_columns
        ]

        if missing:
            raise ValueError(
                f"Model '{model['id']}' references missing factors: "
                + ", ".join(missing)
            )

        mask = pd.Series(True, index=dataframe.index)

        for factor_id in factor_ids:
            mask &= dataframe[
                factor_columns[factor_id]
            ]

        return mask

    raise ValueError(
        f"Unsupported model type: {model_type}"
    )


def negative_magnitude(series: pd.Series) -> pd.Series:
    """Return the magnitude of negative observations, zero otherwise."""

    numeric = pd.to_numeric(
        series,
        errors="coerce",
    ).fillna(0.0)

    return (-numeric.clip(upper=0.0)).astype(float)


def main() -> int:
    """Run the policy risk-exposure comparison."""

    print(
        "MNQ PCS Regime Research — Economic Policy Proxy"
    )
    print("=" * 51)

    required_paths = (
        DATABASE_PATH,
        SCORE_CONFIG_PATH,
        POLICY_CONFIG_PATH,
    )

    for path in required_paths:
        if not path.is_file():
            print(f"ERROR: Missing required file: {path}")
            return 1

    with SCORE_CONFIG_PATH.open(
        "r",
        encoding="utf-8",
    ) as file_handle:
        score_config = yaml.safe_load(file_handle)

    with POLICY_CONFIG_PATH.open(
        "r",
        encoding="utf-8",
    ) as file_handle:
        policy_config = yaml.safe_load(file_handle)

    factors = score_config.get("factors", [])
    models = policy_config.get("models", [])
    policies = policy_config.get("policies", [])
    samples = policy_config.get("samples", [])

    if not factors or not models or not policies or not samples:
        print(
            "ERROR: Factors, models, policies, and samples "
            "must all be configured."
        )
        return 1

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

        for column in (
            "expiration_failure",
            "short_strike_touched",
        ):
            dataframe[column] = (
                dataframe[column]
                .fillna(False)
                .astype(bool)
            )

        factor_columns: dict[str, str] = {}

        for factor in factors:
            factor_id = str(factor["id"])
            source_column = str(factor["column"])
            flag_column = f"flag_{factor_id}"

            if source_column not in dataframe.columns:
                raise ValueError(
                    f"Missing feature column: {source_column}"
                )

            dataframe[flag_column] = evaluate_rule(
                dataframe[source_column],
                str(factor["operator"]),
                factor["value"],
            )

            factor_columns[factor_id] = flag_column

        for model in models:
            model_id = str(model["id"])
            model_column = f"model_{model_id}"

            dataframe[model_column] = make_model_mask(
                dataframe,
                model,
                factor_columns,
            )

        dataframe["downside_return_magnitude"] = (
            negative_magnitude(
                dataframe["expiration_return"]
            )
        )

        dataframe["adverse_move_magnitude"] = (
            negative_magnitude(
                dataframe["maximum_adverse_return"]
            )
        )

        result_rows: list[dict[str, Any]] = []
        cluster_rows: list[dict[str, Any]] = []

        print(
            f"Complete history: {len(dataframe):,} observations | "
            f"{int(dataframe['expiration_failure'].sum())} failures"
        )

        print(
            f"Models: {len(models)} | "
            f"Policies: {len(policies)} | "
            f"Samples: {len(samples)}"
        )

        for sample in samples:
            sample_id = str(sample["id"])
            sample_label = str(sample["label"])

            start_date = pd.Timestamp(
                sample["start_date"]
            )

            end_date = pd.Timestamp(
                sample["end_date"]
            )

            sample_data = dataframe.loc[
                dataframe["entry_date"].between(
                    start_date,
                    end_date,
                )
            ].copy()

            if sample_data.empty:
                raise ValueError(
                    f"Sample '{sample_id}' contains no rows."
                )

            failure_clusters = build_failure_clusters(
                sample_data
            )

            baseline_observations = len(sample_data)
            baseline_failures = int(
                sample_data["expiration_failure"].sum()
            )
            baseline_touches = int(
                sample_data["short_strike_touched"].sum()
            )

            baseline_downside_exposure = float(
                sample_data[
                    "downside_return_magnitude"
                ].sum()
            )

            baseline_adverse_exposure = float(
                sample_data[
                    "adverse_move_magnitude"
                ].sum()
            )

            if failure_clusters.empty:
                baseline_max_cluster_failures = 0.0
            else:
                baseline_max_cluster_failures = float(
                    failure_clusters.groupby(
                        "failure_cluster_id"
                    ).size().max()
                )

            for model in models:
                model_id = str(model["id"])
                model_label = str(model["label"])
                model_column = f"model_{model_id}"

                elevated_mask = sample_data[
                    model_column
                ]

                normal_mask = ~elevated_mask

                elevated_observations = int(
                    elevated_mask.sum()
                )

                elevated_failures = int(
                    sample_data.loc[
                        elevated_mask,
                        "expiration_failure",
                    ].sum()
                )

                elevated_touches = int(
                    sample_data.loc[
                        elevated_mask,
                        "short_strike_touched",
                    ].sum()
                )

                normal_failures = int(
                    sample_data.loc[
                        normal_mask,
                        "expiration_failure",
                    ].sum()
                )

                elevated_failure_rate = safe_divide(
                    elevated_failures,
                    elevated_observations,
                )

                normal_failure_rate = safe_divide(
                    normal_failures,
                    int(normal_mask.sum()),
                )

                for policy in policies:
                    policy_id = str(policy["id"])
                    policy_label = str(policy["label"])
                    multiplier = float(
                        policy["elevated_multiplier"]
                    )

                    if not 0.0 <= multiplier <= 1.0:
                        raise ValueError(
                            f"Policy '{policy_id}' has an "
                            "invalid multiplier."
                        )

                    exposure_multiplier = np.where(
                        elevated_mask.to_numpy(),
                        multiplier,
                        1.0,
                    )

                    exposure_units = float(
                        exposure_multiplier.sum()
                    )

                    weighted_failures = float(
                        (
                            exposure_multiplier
                            * sample_data[
                                "expiration_failure"
                            ].astype(int).to_numpy()
                        ).sum()
                    )

                    weighted_touches = float(
                        (
                            exposure_multiplier
                            * sample_data[
                                "short_strike_touched"
                            ].astype(int).to_numpy()
                        ).sum()
                    )

                    weighted_downside_exposure = float(
                        (
                            exposure_multiplier
                            * sample_data[
                                "downside_return_magnitude"
                            ].to_numpy()
                        ).sum()
                    )

                    weighted_adverse_exposure = float(
                        (
                            exposure_multiplier
                            * sample_data[
                                "adverse_move_magnitude"
                            ].to_numpy()
                        ).sum()
                    )

                    exposure_reduction = (
                        1.0
                        - safe_divide(
                            exposure_units,
                            baseline_observations,
                        )
                    )

                    failure_exposure_reduction = (
                        1.0
                        - safe_divide(
                            weighted_failures,
                            baseline_failures,
                        )
                        if baseline_failures
                        else np.nan
                    )

                    touch_exposure_reduction = (
                        1.0
                        - safe_divide(
                            weighted_touches,
                            baseline_touches,
                        )
                        if baseline_touches
                        else np.nan
                    )

                    downside_exposure_reduction = (
                        1.0
                        - safe_divide(
                            weighted_downside_exposure,
                            baseline_downside_exposure,
                        )
                        if baseline_downside_exposure
                        else np.nan
                    )

                    adverse_exposure_reduction = (
                        1.0
                        - safe_divide(
                            weighted_adverse_exposure,
                            baseline_adverse_exposure,
                        )
                        if baseline_adverse_exposure
                        else np.nan
                    )

                    model_failure_capture = safe_divide(
                        elevated_failures,
                        baseline_failures,
                    )

                    model_touch_capture = safe_divide(
                        elevated_touches,
                        baseline_touches,
                    )

                    cluster_weighted_failures: list[float] = []

                    if not failure_clusters.empty:
                        failure_weight_map = pd.Series(
                            exposure_multiplier,
                            index=sample_data.index,
                        )

                        cluster_detail = (
                            failure_clusters[
                                [
                                    "entry_date",
                                    "expiration_proxy_date",
                                    "failure_cluster_id",
                                ]
                            ]
                            .join(
                                failure_weight_map.rename(
                                    "exposure_multiplier"
                                ),
                                how="left",
                            )
                        )

                        grouped = cluster_detail.groupby(
                            "failure_cluster_id"
                        )

                        for cluster_id, group in grouped:
                            cluster_weight = float(
                                group[
                                    "exposure_multiplier"
                                ].sum()
                            )

                            cluster_weighted_failures.append(
                                cluster_weight
                            )

                            cluster_rows.append(
                                {
                                    "sample_id": sample_id,
                                    "sample_label": sample_label,
                                    "model_id": model_id,
                                    "model_label": model_label,
                                    "policy_id": policy_id,
                                    "policy_label": policy_label,
                                    "elevated_multiplier": multiplier,
                                    "failure_cluster_id": int(
                                        cluster_id
                                    ),
                                    "cluster_start_entry": group[
                                        "entry_date"
                                    ].min(),
                                    "cluster_end_expiration": group[
                                        "expiration_proxy_date"
                                    ].max(),
                                    "baseline_failure_entries": len(
                                        group
                                    ),
                                    "weighted_failure_exposure": (
                                        cluster_weight
                                    ),
                                    "cluster_failure_exposure_reduction": (
                                        1.0
                                        - safe_divide(
                                            cluster_weight,
                                            len(group),
                                        )
                                    ),
                                }
                            )

                    max_cluster_weighted_failures = (
                        max(cluster_weighted_failures)
                        if cluster_weighted_failures
                        else 0.0
                    )

                    max_cluster_reduction = (
                        1.0
                        - safe_divide(
                            max_cluster_weighted_failures,
                            baseline_max_cluster_failures,
                        )
                        if baseline_max_cluster_failures
                        else np.nan
                    )

                    result_rows.append(
                        {
                            "sample_id": sample_id,
                            "sample_label": sample_label,
                            "sample_start": sample_data[
                                "entry_date"
                            ].min(),
                            "sample_end": sample_data[
                                "entry_date"
                            ].max(),
                            "model_id": model_id,
                            "model_label": model_label,
                            "policy_id": policy_id,
                            "policy_label": policy_label,
                            "elevated_multiplier": multiplier,
                            "observations": baseline_observations,
                            "failures": baseline_failures,
                            "touches": baseline_touches,
                            "elevated_observations": (
                                elevated_observations
                            ),
                            "elevated_observation_share": (
                                safe_divide(
                                    elevated_observations,
                                    baseline_observations,
                                )
                            ),
                            "elevated_failures": (
                                elevated_failures
                            ),
                            "elevated_failure_rate": (
                                elevated_failure_rate
                            ),
                            "normal_failures": normal_failures,
                            "normal_failure_rate": (
                                normal_failure_rate
                            ),
                            "model_failure_capture": (
                                model_failure_capture
                            ),
                            "elevated_touches": elevated_touches,
                            "model_touch_capture": (
                                model_touch_capture
                            ),
                            "exposure_units": exposure_units,
                            "exposure_retained": (
                                safe_divide(
                                    exposure_units,
                                    baseline_observations,
                                )
                            ),
                            "exposure_reduction": (
                                exposure_reduction
                            ),
                            "weighted_failure_exposure": (
                                weighted_failures
                            ),
                            "failure_exposure_reduction": (
                                failure_exposure_reduction
                            ),
                            "failure_reduction_efficiency": (
                                safe_divide(
                                    failure_exposure_reduction,
                                    exposure_reduction,
                                )
                                if exposure_reduction > 0
                                else np.nan
                            ),
                            "weighted_touch_exposure": (
                                weighted_touches
                            ),
                            "touch_exposure_reduction": (
                                touch_exposure_reduction
                            ),
                            "touch_reduction_efficiency": (
                                safe_divide(
                                    touch_exposure_reduction,
                                    exposure_reduction,
                                )
                                if exposure_reduction > 0
                                else np.nan
                            ),
                            "weighted_downside_return_exposure": (
                                weighted_downside_exposure
                            ),
                            "downside_return_exposure_reduction": (
                                downside_exposure_reduction
                            ),
                            "weighted_adverse_move_exposure": (
                                weighted_adverse_exposure
                            ),
                            "adverse_move_exposure_reduction": (
                                adverse_exposure_reduction
                            ),
                            "baseline_max_cluster_failures": (
                                baseline_max_cluster_failures
                            ),
                            "max_cluster_weighted_failures": (
                                max_cluster_weighted_failures
                            ),
                            "max_cluster_failure_exposure_reduction": (
                                max_cluster_reduction
                            ),
                        }
                    )

        results = pd.DataFrame(result_rows)

        cluster_details = pd.DataFrame(cluster_rows)

        RESULTS_DIR.mkdir(
            parents=True,
            exist_ok=True,
        )

        results.to_csv(
            RESULTS_OUTPUT,
            index=False,
        )

        cluster_details.to_csv(
            CLUSTER_OUTPUT,
            index=False,
        )

        connection.register(
            "policy_proxy_results_dataframe",
            results,
        )

        connection.execute(
            """
            CREATE OR REPLACE TABLE
                pcs.policy_proxy_results AS
            SELECT *
            FROM policy_proxy_results_dataframe
            ORDER BY sample_id, model_id, elevated_multiplier DESC
            """
        )

        connection.register(
            "policy_cluster_details_dataframe",
            cluster_details,
        )

        connection.execute(
            """
            CREATE OR REPLACE TABLE
                pcs.policy_cluster_details AS
            SELECT *
            FROM policy_cluster_details_dataframe
            ORDER BY
                sample_id,
                model_id,
                elevated_multiplier DESC,
                failure_cluster_id
            """
        )

        forward_results = results.loc[
            results["sample_id"] == "forward_test"
        ].sort_values(
            [
                "model_id",
                "elevated_multiplier",
            ],
            ascending=[True, False],
        )

        print("\nForward-test policy comparison")
        print("-" * 160)

        for row in forward_results.itertuples(index=False):
            print(
                f"{row.model_label:<39} | "
                f"{row.policy_label:<26} | "
                f"Exposure kept={row.exposure_retained:>6.1%} | "
                f"Failure exposure cut={row.failure_exposure_reduction:>6.1%} | "
                f"Touch exposure cut={row.touch_exposure_reduction:>6.1%} | "
                f"Adverse exposure cut={row.adverse_move_exposure_reduction:>6.1%} | "
                f"Max cluster cut={row.max_cluster_failure_exposure_reduction:>6.1%}"
            )

        summary_lines = [
            "MNQ PCS REGIME RESEARCH — ECONOMIC POLICY PROXY",
            "=" * 56,
            "",
            "Methodological limitation",
            "-" * 56,
            (
                "This is not an options P&L backtest. It does not "
                "model historical credits, strike rounding, transaction "
                "costs, or the 150-point spread payoff."
            ),
            (
                "Exposure retained is only a proxy for retained credit "
                "opportunity. Failure, touch, and adverse-move reductions "
                "are exposure-weighted risk proxies."
            ),
            "",
            "Forward-test results",
            "-" * 56,
        ]

        for model in models:
            model_rows = forward_results.loc[
                forward_results["model_id"]
                == str(model["id"])
            ]

            summary_lines.append(
                f"{model['label']}:"
            )

            for row in model_rows.itertuples(index=False):
                summary_lines.append(
                    f"  {row.policy_label} | "
                    f"Exposure retained={row.exposure_retained:.1%} | "
                    f"Failure exposure reduction="
                    f"{row.failure_exposure_reduction:.1%} | "
                    f"Touch exposure reduction="
                    f"{row.touch_exposure_reduction:.1%} | "
                    f"Adverse exposure reduction="
                    f"{row.adverse_move_exposure_reduction:.1%} | "
                    f"Max cluster reduction="
                    f"{row.max_cluster_failure_exposure_reduction:.1%}"
                )

            summary_lines.append("")

        SUMMARY_OUTPUT.write_text(
            "\n".join(summary_lines) + "\n",
            encoding="utf-8",
        )

        print("\nMethodological status")
        print("-" * 88)

        print(
            "Risk-exposure proxy only; no historical option-credit "
            "or spread-payoff modeling."
        )

        print(f"\nResults output: {RESULTS_OUTPUT}")
        print(f"Cluster output: {CLUSTER_OUTPUT}")
        print(f"Summary output: {SUMMARY_OUTPUT}")
        print(
            "DuckDB tables: pcs.policy_proxy_results, "
            "pcs.policy_cluster_details"
        )

        print(
            "Economic policy proxy completed successfully."
        )

        return 0

    except Exception as exc:
        print(f"\nERROR: {exc}")
        return 1

    finally:
        connection.close()


if __name__ == "__main__":
    sys.exit(main())

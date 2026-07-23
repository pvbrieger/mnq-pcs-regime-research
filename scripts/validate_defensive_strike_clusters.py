"""Validate defensive-strike results across failure events and clusters.

This script answers two questions:

1. Which historical failure clusters produced the intrinsic-value savings from
   moving elevated-regime spreads farther below spot?
2. Does the maximum tolerable credit haircut remain meaningful after removing
   each failure cluster one at a time?

This is still a fixed-credit expiration proxy, not a historical options
backtest. It does not include actual option quotes, fills, listed-strike
rounding, slippage, commissions, or portfolio margin.

Inputs
------
- database/research.duckdb
- config/candidate_regime_score.yaml
- config/policy_proxy.yaml
- config/defensive_strike_validation.yaml
- pcs.friday_regime_features

Outputs
-------
- results/defensive_strike_validation/event_attribution.csv
- results/defensive_strike_validation/leave_one_cluster_out.csv
- results/defensive_strike_validation/validation_summary.csv
- results/defensive_strike_validation/defensive_strike_validation_summary.txt
- DuckDB tables:
    pcs.defensive_strike_event_attribution
    pcs.defensive_strike_cluster_validation
    pcs.defensive_strike_validation_summary
"""

from __future__ import annotations

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
VALIDATION_CONFIG_PATH = (
    PROJECT_ROOT / "config" / "defensive_strike_validation.yaml"
)

RESULTS_DIR = (
    PROJECT_ROOT / "results" / "defensive_strike_validation"
)
ATTRIBUTION_OUTPUT = RESULTS_DIR / "event_attribution.csv"
CLUSTER_OUTPUT = RESULTS_DIR / "leave_one_cluster_out.csv"
VALIDATION_OUTPUT = RESULTS_DIR / "validation_summary.csv"
SUMMARY_OUTPUT = (
    RESULTS_DIR / "defensive_strike_validation_summary.txt"
)


def safe_divide(
    numerator: float,
    denominator: float,
) -> float:
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


def first_existing_column(
    dataframe: pd.DataFrame,
    candidates: tuple[str, ...],
) -> str | None:
    """Return the first candidate column present in the dataframe."""

    for column in candidates:
        if column in dataframe.columns:
            return column

    return None


def resolve_numeric_column(
    dataframe: pd.DataFrame,
    candidates: tuple[str, ...],
    description: str,
) -> tuple[pd.Series, str]:
    """Resolve one required numeric column."""

    column = first_existing_column(
        dataframe,
        candidates,
    )

    if column is None:
        raise ValueError(
            f"Unable to resolve {description}. Available columns: "
            + ", ".join(sorted(dataframe.columns))
        )

    values = pd.to_numeric(
        dataframe[column],
        errors="coerce",
    ).astype(float)

    if values.isna().any():
        raise ValueError(
            f"Resolved {description} column '{column}' contains "
            f"{int(values.isna().sum())} missing values."
        )

    return values, column


def make_model_mask(
    dataframe: pd.DataFrame,
    model: dict[str, Any],
    factor_columns: dict[str, str],
) -> pd.Series:
    """Build the elevated-regime mask for one configured model."""

    model_type = str(model["type"])

    if model_type == "score_at_least":
        score = pd.Series(
            0,
            index=dataframe.index,
            dtype="int64",
        )

        for factor_id_value in model["factor_ids"]:
            factor_id = str(factor_id_value)

            if factor_id not in factor_columns:
                raise ValueError(
                    f"Model '{model['id']}' references missing "
                    f"factor '{factor_id}'."
                )

            score += dataframe[
                factor_columns[factor_id]
            ].astype(int)

        return score >= int(model["minimum_score"])

    if model_type == "all_factors":
        mask = pd.Series(
            True,
            index=dataframe.index,
        )

        for factor_id_value in model["factor_ids"]:
            factor_id = str(factor_id_value)

            if factor_id not in factor_columns:
                raise ValueError(
                    f"Model '{model['id']}' references missing "
                    f"factor '{factor_id}'."
                )

            mask &= dataframe[
                factor_columns[factor_id]
            ]

        return mask

    raise ValueError(
        f"Unsupported model type: {model_type}"
    )


def spread_intrinsic(
    short_strike: pd.Series,
    settlement: pd.Series,
    spread_width: float,
) -> pd.Series:
    """Calculate capped put-spread intrinsic value."""

    values = np.clip(
        short_strike.to_numpy(dtype=float)
        - settlement.to_numpy(dtype=float),
        0.0,
        spread_width,
    )

    return pd.Series(
        values,
        index=short_strike.index,
        dtype=float,
    )


def build_failure_clusters(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """Build connected clusters from overlapping failed trades."""

    failures = dataframe.loc[
        dataframe["standard_failure"]
    ].sort_values(
        [
            "entry_date",
            "expiration_proxy_date",
        ]
    )

    if failures.empty:
        return pd.DataFrame(
            columns=[
                "cluster_id",
                "cluster_start",
                "cluster_last_entry",
                "cluster_end",
                "failure_entries",
                "failure_entry_dates",
            ]
        )

    cluster_records: list[dict[str, Any]] = []
    current_indexes: list[Any] = []
    current_end: pd.Timestamp | None = None

    for index, row in failures.iterrows():
        entry_date = pd.Timestamp(
            row["entry_date"]
        )
        expiration_date = pd.Timestamp(
            row["expiration_proxy_date"]
        )

        if (
            current_end is None
            or entry_date > current_end
        ):
            if current_indexes:
                cluster = failures.loc[
                    current_indexes
                ]

                cluster_records.append(
                    {
                        "cluster_id": len(
                            cluster_records
                        ) + 1,
                        "cluster_start": cluster[
                            "entry_date"
                        ].min(),
                        "cluster_last_entry": cluster[
                            "entry_date"
                        ].max(),
                        "cluster_end": cluster[
                            "expiration_proxy_date"
                        ].max(),
                        "failure_entries": len(
                            cluster
                        ),
                        "failure_entry_dates": ", ".join(
                            cluster[
                                "entry_date"
                            ]
                            .dt.strftime("%Y-%m-%d")
                            .tolist()
                        ),
                    }
                )

            current_indexes = [index]
            current_end = expiration_date
        else:
            current_indexes.append(index)
            current_end = max(
                current_end,
                expiration_date,
            )

    if current_indexes:
        cluster = failures.loc[
            current_indexes
        ]

        cluster_records.append(
            {
                "cluster_id": len(
                    cluster_records
                ) + 1,
                "cluster_start": cluster[
                    "entry_date"
                ].min(),
                "cluster_last_entry": cluster[
                    "entry_date"
                ].max(),
                "cluster_end": cluster[
                    "expiration_proxy_date"
                ].max(),
                "failure_entries": len(
                    cluster
                ),
                "failure_entry_dates": ", ".join(
                    cluster[
                        "entry_date"
                    ]
                    .dt.strftime("%Y-%m-%d")
                    .tolist()
                ),
            }
        )

    return pd.DataFrame(
        cluster_records
    )


def assign_failure_cluster_ids(
    dataframe: pd.DataFrame,
    clusters: pd.DataFrame,
) -> pd.Series:
    """Assign each failed row to its connected cluster."""

    cluster_ids = pd.Series(
        pd.NA,
        index=dataframe.index,
        dtype="Int64",
    )

    for row in clusters.itertuples(index=False):
        failure_mask = (
            dataframe["standard_failure"]
            & dataframe["entry_date"].between(
                pd.Timestamp(row.cluster_start),
                pd.Timestamp(row.cluster_last_entry),
            )
        )

        cluster_ids.loc[
            failure_mask
        ] = int(row.cluster_id)

    return cluster_ids


def scenario_metrics(
    dataframe: pd.DataFrame,
    elevated_mask: pd.Series,
    extra_distance: float,
    spread_width: float,
) -> dict[str, Any]:
    """Calculate defensive-strike metrics for one dataset."""

    elevated_mask = (
        elevated_mask
        .reindex(dataframe.index)
        .fillna(False)
        .astype(bool)
    )

    elevated_count = int(
        elevated_mask.sum()
    )

    defensive_short = (
        dataframe["standard_short_strike"]
        - (
            dataframe["entry_spot_proxy"]
            * extra_distance
        )
    )

    defensive_intrinsic = spread_intrinsic(
        defensive_short,
        dataframe["expiration_settlement_proxy"],
        spread_width,
    )

    defensive_failure = (
        dataframe[
            "expiration_settlement_proxy"
        ]
        < defensive_short
    )

    standard_elevated_failures = int(
        dataframe.loc[
            elevated_mask,
            "standard_failure",
        ].sum()
    )

    defensive_elevated_failures = int(
        defensive_failure.loc[
            elevated_mask
        ].sum()
    )

    standard_intrinsic_total = float(
        dataframe.loc[
            elevated_mask,
            "standard_intrinsic_points",
        ].sum()
    )

    defensive_intrinsic_total = float(
        defensive_intrinsic.loc[
            elevated_mask
        ].sum()
    )

    intrinsic_saved = (
        standard_intrinsic_total
        - defensive_intrinsic_total
    )

    max_haircut = safe_divide(
        intrinsic_saved,
        elevated_count,
    )

    return {
        "elevated_observations": elevated_count,
        "standard_elevated_failures": (
            standard_elevated_failures
        ),
        "defensive_elevated_failures": (
            defensive_elevated_failures
        ),
        "failures_avoided": (
            standard_elevated_failures
            - defensive_elevated_failures
        ),
        "standard_elevated_intrinsic_points": (
            standard_intrinsic_total
        ),
        "defensive_elevated_intrinsic_points": (
            defensive_intrinsic_total
        ),
        "intrinsic_points_saved": intrinsic_saved,
        "maximum_tolerable_credit_haircut_points": (
            max_haircut
        ),
        "defensive_short_strike": defensive_short,
        "defensive_intrinsic_points": defensive_intrinsic,
        "defensive_failure": defensive_failure,
    }


def concentration_label(
    top_cluster_share: float,
    minimum_haircut_retention: float,
) -> str:
    """Assign a descriptive concentration label."""

    if (
        not pd.isna(top_cluster_share)
        and top_cluster_share <= 0.50
        and not pd.isna(
            minimum_haircut_retention
        )
        and minimum_haircut_retention >= 0.50
    ):
        return "Broadly distributed"

    if (
        not pd.isna(top_cluster_share)
        and top_cluster_share <= 0.75
        and not pd.isna(
            minimum_haircut_retention
        )
        and minimum_haircut_retention >= 0.25
    ):
        return "Moderately concentrated"

    return "Highly concentrated"


def main() -> int:
    """Run event attribution and leave-one-cluster-out validation."""

    print(
        "MNQ PCS Regime Research — Defensive Strike "
        "Cluster Validation"
    )
    print("=" * 66)

    required_paths = (
        DATABASE_PATH,
        SCORE_CONFIG_PATH,
        POLICY_CONFIG_PATH,
        VALIDATION_CONFIG_PATH,
    )

    for path in required_paths:
        if not path.is_file():
            print(
                f"ERROR: Missing required file: {path}"
            )
            return 1

    with SCORE_CONFIG_PATH.open(
        "r",
        encoding="utf-8",
    ) as file_handle:
        score_config = yaml.safe_load(
            file_handle
        )

    with POLICY_CONFIG_PATH.open(
        "r",
        encoding="utf-8",
    ) as file_handle:
        policy_config = yaml.safe_load(
            file_handle
        )

    with VALIDATION_CONFIG_PATH.open(
        "r",
        encoding="utf-8",
    ) as file_handle:
        validation_config = yaml.safe_load(
            file_handle
        )

    factors = score_config.get(
        "factors",
        [],
    )

    configured_models = {
        str(model["id"]): model
        for model in policy_config.get(
            "models",
            [],
        )
    }

    configured_samples = {
        str(sample["id"]): sample
        for sample in policy_config.get(
            "samples",
            [],
        )
    }

    focus_model_ids = [
        str(model_id)
        for model_id in validation_config[
            "focus_model_ids"
        ]
    ]

    focus_sample_ids = [
        str(sample_id)
        for sample_id in validation_config[
            "focus_sample_ids"
        ]
    ]

    focus_distances = [
        float(value)
        for value in validation_config[
            "focus_additional_distance_fractions"
        ]
    ]

    spread_width = float(
        validation_config[
            "spread_width_points"
        ]
    )

    anchor_credit = float(
        validation_config[
            "anchor_standard_credit_points"
        ]
    )

    missing_models = [
        model_id
        for model_id in focus_model_ids
        if model_id not in configured_models
    ]

    missing_samples = [
        sample_id
        for sample_id in focus_sample_ids
        if sample_id not in configured_samples
    ]

    if missing_models:
        print(
            "ERROR: Missing configured models: "
            + ", ".join(missing_models)
        )
        return 1

    if missing_samples:
        print(
            "ERROR: Missing configured samples: "
            + ", ".join(missing_samples)
        )
        return 1

    connection = duckdb.connect(
        str(DATABASE_PATH)
    )

    try:
        dataframe = connection.execute(
            """
            SELECT *
            FROM pcs.friday_regime_features
            WHERE complete_core_features = TRUE
            ORDER BY entry_date
            """
        ).fetchdf()

        dataframe["entry_date"] = (
            pd.to_datetime(
                dataframe["entry_date"]
            )
        )

        dataframe[
            "expiration_proxy_date"
        ] = pd.to_datetime(
            dataframe[
                "expiration_proxy_date"
            ]
        )

        entry_spot, entry_spot_source = (
            resolve_numeric_column(
                dataframe,
                (
                    "weekly_close",
                    "entry_close",
                    "entry_spot",
                    "entry_price",
                    "ndx_entry_close",
                ),
                "entry spot",
            )
        )

        standard_short, short_source = (
            resolve_numeric_column(
                dataframe,
                (
                    "short_strike_proxy",
                    "proxy_short_strike",
                    "short_strike",
                    "short_strike_level",
                    "short_strike_price",
                ),
                "standard short strike",
            )
        )

        settlement, settlement_source = (
            resolve_numeric_column(
                dataframe,
                (
                    "expiration_close",
                    "expiration_proxy_close",
                    "expiration_settlement_proxy",
                    "expiration_settlement",
                    "expiration_ndx_close",
                    "expiration_price",
                ),
                "expiration settlement",
            )
        )

        dataframe[
            "entry_spot_proxy"
        ] = entry_spot

        dataframe[
            "standard_short_strike"
        ] = standard_short

        dataframe[
            "expiration_settlement_proxy"
        ] = settlement

        dataframe[
            "standard_intrinsic_points"
        ] = spread_intrinsic(
            dataframe[
                "standard_short_strike"
            ],
            dataframe[
                "expiration_settlement_proxy"
            ],
            spread_width,
        )

        dataframe["standard_failure"] = (
            dataframe[
                "expiration_settlement_proxy"
            ]
            < dataframe[
                "standard_short_strike"
            ]
        )

        recorded_failure = (
            dataframe[
                "expiration_failure"
            ]
            .fillna(False)
            .astype(bool)
        )

        mismatch_count = int(
            (
                dataframe[
                    "standard_failure"
                ]
                != recorded_failure
            ).sum()
        )

        if mismatch_count:
            raise ValueError(
                "Resolved standard strike and "
                "settlement disagree with "
                f"expiration_failure on "
                f"{mismatch_count} rows."
            )

        factor_columns: dict[
            str,
            str,
        ] = {}

        for factor in factors:
            factor_id = str(
                factor["id"]
            )
            source_column = str(
                factor["column"]
            )
            flag_column = (
                f"flag_{factor_id}"
            )

            if (
                source_column
                not in dataframe.columns
            ):
                raise ValueError(
                    "Missing feature column: "
                    f"{source_column}"
                )

            dataframe[
                flag_column
            ] = evaluate_rule(
                dataframe[
                    source_column
                ],
                str(
                    factor["operator"]
                ),
                factor["value"],
            )

            factor_columns[
                factor_id
            ] = flag_column

        for model_id in focus_model_ids:
            model = configured_models[
                model_id
            ]

            dataframe[
                f"model_{model_id}"
            ] = make_model_mask(
                dataframe,
                model,
                factor_columns,
            )

        print(
            f"Complete history: "
            f"{len(dataframe):,} observations | "
            f"{int(dataframe['standard_failure'].sum())} "
            "standard-strike failures"
        )

        print(
            f"Entry spot source: "
            f"{entry_spot_source}"
        )

        print(
            f"Short strike source: "
            f"{short_source}"
        )

        print(
            f"Settlement source: "
            f"{settlement_source}"
        )

        print(
            "Focus distances: "
            + ", ".join(
                f"{distance:.1%}"
                for distance in focus_distances
            )
        )

        attribution_rows: list[
            dict[str, Any]
        ] = []

        cluster_validation_rows: list[
            dict[str, Any]
        ] = []

        validation_rows: list[
            dict[str, Any]
        ] = []

        for sample_id in focus_sample_ids:
            sample = configured_samples[
                sample_id
            ]

            sample_data = dataframe.loc[
                dataframe[
                    "entry_date"
                ].between(
                    pd.Timestamp(
                        sample[
                            "start_date"
                        ]
                    ),
                    pd.Timestamp(
                        sample[
                            "end_date"
                        ]
                    ),
                )
            ].copy()

            if sample_data.empty:
                raise ValueError(
                    f"Sample '{sample_id}' "
                    "contains no rows."
                )

            clusters = (
                build_failure_clusters(
                    sample_data
                )
            )

            sample_data[
                "failure_cluster_id"
            ] = assign_failure_cluster_ids(
                sample_data,
                clusters,
            )

            for model_id in focus_model_ids:
                model = configured_models[
                    model_id
                ]
                model_label = str(
                    model["label"]
                )
                elevated_column = (
                    f"model_{model_id}"
                )
                elevated_mask = (
                    sample_data[
                        elevated_column
                    ]
                    .astype(bool)
                )

                for distance in focus_distances:
                    full_metrics = (
                        scenario_metrics(
                            sample_data,
                            elevated_mask,
                            distance,
                            spread_width,
                        )
                    )

                    full_haircut = float(
                        full_metrics[
                            "maximum_tolerable_credit_haircut_points"
                        ]
                    )

                    match_credit = (
                        anchor_credit
                        - full_haircut
                    )

                    saving_series = (
                        sample_data[
                            "standard_intrinsic_points"
                        ]
                        - full_metrics[
                            "defensive_intrinsic_points"
                        ]
                    )

                    saving_series = (
                        saving_series.where(
                            elevated_mask,
                            0.0,
                        )
                    )

                    contributing_cluster_count = 0
                    top_cluster_savings = 0.0

                    for cluster in clusters.itertuples(
                        index=False
                    ):
                        cluster_failure_mask = (
                            sample_data[
                                "standard_failure"
                            ]
                            & (
                                sample_data[
                                    "failure_cluster_id"
                                ]
                                == int(
                                    cluster.cluster_id
                                )
                            )
                        )

                        cluster_elevated_mask = (
                            cluster_failure_mask
                            & elevated_mask
                        )

                        cluster_standard_failures = int(
                            cluster_elevated_mask.sum()
                        )

                        cluster_defensive_failures = int(
                            (
                                cluster_elevated_mask
                                & full_metrics[
                                    "defensive_failure"
                                ]
                            ).sum()
                        )

                        cluster_savings = float(
                            saving_series.loc[
                                cluster_failure_mask
                            ].sum()
                        )

                        if cluster_savings > 0:
                            contributing_cluster_count += 1

                        top_cluster_savings = max(
                            top_cluster_savings,
                            cluster_savings,
                        )

                        attribution_rows.append(
                            {
                                "sample_id": sample_id,
                                "sample_label": str(
                                    sample["label"]
                                ),
                                "model_id": model_id,
                                "model_label": model_label,
                                "additional_distance_fraction": (
                                    distance
                                ),
                                "additional_distance_percent": (
                                    distance * 100.0
                                ),
                                "cluster_id": int(
                                    cluster.cluster_id
                                ),
                                "cluster_start": (
                                    pd.Timestamp(
                                        cluster.cluster_start
                                    )
                                ),
                                "cluster_last_entry": (
                                    pd.Timestamp(
                                        cluster.cluster_last_entry
                                    )
                                ),
                                "cluster_end": (
                                    pd.Timestamp(
                                        cluster.cluster_end
                                    )
                                ),
                                "cluster_failure_entries": int(
                                    cluster.failure_entries
                                ),
                                "elevated_standard_failures_in_cluster": (
                                    cluster_standard_failures
                                ),
                                "elevated_defensive_failures_in_cluster": (
                                    cluster_defensive_failures
                                ),
                                "elevated_failures_avoided_in_cluster": (
                                    cluster_standard_failures
                                    - cluster_defensive_failures
                                ),
                                "intrinsic_points_saved_in_cluster": (
                                    cluster_savings
                                ),
                                "intrinsic_savings_share": (
                                    safe_divide(
                                        cluster_savings,
                                        full_metrics[
                                            "intrinsic_points_saved"
                                        ],
                                    )
                                ),
                            }
                        )

                    loo_haircuts: list[
                        float
                    ] = []

                    loo_failure_differences: list[
                        int
                    ] = []

                    positive_savings_tests = 0
                    nonworsening_failure_tests = 0

                    for cluster in clusters.itertuples(
                        index=False
                    ):
                        overlap_mask = (
                            (
                                sample_data[
                                    "entry_date"
                                ]
                                <= pd.Timestamp(
                                    cluster.cluster_end
                                )
                            )
                            & (
                                sample_data[
                                    "expiration_proxy_date"
                                ]
                                >= pd.Timestamp(
                                    cluster.cluster_start
                                )
                            )
                        )

                        reduced_data = (
                            sample_data.loc[
                                ~overlap_mask
                            ].copy()
                        )

                        reduced_elevated = (
                            elevated_mask.loc[
                                ~overlap_mask
                            ]
                        )

                        reduced_metrics = (
                            scenario_metrics(
                                reduced_data,
                                reduced_elevated,
                                distance,
                                spread_width,
                            )
                        )

                        reduced_haircut = float(
                            reduced_metrics[
                                "maximum_tolerable_credit_haircut_points"
                            ]
                        )

                        failure_difference = (
                            reduced_metrics[
                                "standard_elevated_failures"
                            ]
                            - reduced_metrics[
                                "defensive_elevated_failures"
                            ]
                        )

                        if (
                            reduced_metrics[
                                "intrinsic_points_saved"
                            ] > 0
                        ):
                            positive_savings_tests += 1

                        if failure_difference >= 0:
                            nonworsening_failure_tests += 1

                        loo_haircuts.append(
                            reduced_haircut
                        )

                        loo_failure_differences.append(
                            int(
                                failure_difference
                            )
                        )

                        cluster_validation_rows.append(
                            {
                                "sample_id": sample_id,
                                "sample_label": str(
                                    sample["label"]
                                ),
                                "model_id": model_id,
                                "model_label": model_label,
                                "additional_distance_fraction": (
                                    distance
                                ),
                                "additional_distance_percent": (
                                    distance * 100.0
                                ),
                                "removed_cluster_id": int(
                                    cluster.cluster_id
                                ),
                                "removed_cluster_start": (
                                    pd.Timestamp(
                                        cluster.cluster_start
                                    )
                                ),
                                "removed_cluster_end": (
                                    pd.Timestamp(
                                        cluster.cluster_end
                                    )
                                ),
                                "removed_failure_entries": int(
                                    cluster.failure_entries
                                ),
                                "removed_observations": int(
                                    overlap_mask.sum()
                                ),
                                "remaining_observations": len(
                                    reduced_data
                                ),
                                "remaining_elevated_observations": (
                                    reduced_metrics[
                                        "elevated_observations"
                                    ]
                                ),
                                "remaining_standard_elevated_failures": (
                                    reduced_metrics[
                                        "standard_elevated_failures"
                                    ]
                                ),
                                "remaining_defensive_elevated_failures": (
                                    reduced_metrics[
                                        "defensive_elevated_failures"
                                    ]
                                ),
                                "remaining_failures_avoided": (
                                    failure_difference
                                ),
                                "remaining_intrinsic_points_saved": (
                                    reduced_metrics[
                                        "intrinsic_points_saved"
                                    ]
                                ),
                                "remaining_maximum_tolerable_credit_haircut_points": (
                                    reduced_haircut
                                ),
                                "remaining_match_credit_points": (
                                    anchor_credit
                                    - reduced_haircut
                                ),
                                "haircut_retention_ratio": (
                                    safe_divide(
                                        reduced_haircut,
                                        full_haircut,
                                    )
                                ),
                            }
                        )

                    minimum_haircut = (
                        min(loo_haircuts)
                        if loo_haircuts
                        else np.nan
                    )

                    median_haircut = (
                        float(
                            np.median(
                                loo_haircuts
                            )
                        )
                        if loo_haircuts
                        else np.nan
                    )

                    minimum_haircut_retention = (
                        safe_divide(
                            minimum_haircut,
                            full_haircut,
                        )
                    )

                    top_cluster_share = (
                        safe_divide(
                            top_cluster_savings,
                            full_metrics[
                                "intrinsic_points_saved"
                            ],
                        )
                    )

                    classification = (
                        concentration_label(
                            top_cluster_share,
                            minimum_haircut_retention,
                        )
                    )

                    validation_rows.append(
                        {
                            "sample_id": sample_id,
                            "sample_label": str(
                                sample["label"]
                            ),
                            "model_id": model_id,
                            "model_label": model_label,
                            "additional_distance_fraction": (
                                distance
                            ),
                            "additional_distance_percent": (
                                distance * 100.0
                            ),
                            "anchor_standard_credit_points": (
                                anchor_credit
                            ),
                            "observations": len(
                                sample_data
                            ),
                            "failure_clusters": len(
                                clusters
                            ),
                            "elevated_observations": (
                                full_metrics[
                                    "elevated_observations"
                                ]
                            ),
                            "standard_elevated_failures": (
                                full_metrics[
                                    "standard_elevated_failures"
                                ]
                            ),
                            "defensive_elevated_failures": (
                                full_metrics[
                                    "defensive_elevated_failures"
                                ]
                            ),
                            "failures_avoided": (
                                full_metrics[
                                    "failures_avoided"
                                ]
                            ),
                            "intrinsic_points_saved": (
                                full_metrics[
                                    "intrinsic_points_saved"
                                ]
                            ),
                            "maximum_tolerable_credit_haircut_points": (
                                full_haircut
                            ),
                            "defensive_credit_to_match_standard_points": (
                                match_credit
                            ),
                            "contributing_failure_clusters": (
                                contributing_cluster_count
                            ),
                            "top_cluster_intrinsic_savings_share": (
                                top_cluster_share
                            ),
                            "leave_one_cluster_tests": len(
                                loo_haircuts
                            ),
                            "positive_savings_cluster_removals": (
                                positive_savings_tests
                            ),
                            "nonworsening_failure_cluster_removals": (
                                nonworsening_failure_tests
                            ),
                            "minimum_leave_one_cluster_haircut_points": (
                                minimum_haircut
                            ),
                            "median_leave_one_cluster_haircut_points": (
                                median_haircut
                            ),
                            "minimum_haircut_retention_ratio": (
                                minimum_haircut_retention
                            ),
                            "minimum_failures_avoided_after_cluster_removal": (
                                min(
                                    loo_failure_differences
                                )
                                if loo_failure_differences
                                else np.nan
                            ),
                            "concentration_assessment": (
                                classification
                            ),
                        }
                    )

        attribution = pd.DataFrame(
            attribution_rows
        )

        cluster_validation = pd.DataFrame(
            cluster_validation_rows
        )

        validation = pd.DataFrame(
            validation_rows
        ).sort_values(
            [
                "sample_id",
                "model_id",
                "additional_distance_fraction",
            ]
        ).reset_index(drop=True)

        RESULTS_DIR.mkdir(
            parents=True,
            exist_ok=True,
        )

        attribution.to_csv(
            ATTRIBUTION_OUTPUT,
            index=False,
        )

        cluster_validation.to_csv(
            CLUSTER_OUTPUT,
            index=False,
        )

        validation.to_csv(
            VALIDATION_OUTPUT,
            index=False,
        )

        connection.register(
            "defensive_event_attribution_dataframe",
            attribution,
        )

        connection.register(
            "defensive_cluster_validation_dataframe",
            cluster_validation,
        )

        connection.register(
            "defensive_validation_summary_dataframe",
            validation,
        )

        connection.execute(
            """
            CREATE OR REPLACE TABLE
                pcs.defensive_strike_event_attribution AS
            SELECT *
            FROM defensive_event_attribution_dataframe
            ORDER BY
                sample_id,
                model_id,
                additional_distance_fraction,
                cluster_id
            """
        )

        connection.execute(
            """
            CREATE OR REPLACE TABLE
                pcs.defensive_strike_cluster_validation AS
            SELECT *
            FROM defensive_cluster_validation_dataframe
            ORDER BY
                sample_id,
                model_id,
                additional_distance_fraction,
                removed_cluster_id
            """
        )

        connection.execute(
            """
            CREATE OR REPLACE TABLE
                pcs.defensive_strike_validation_summary AS
            SELECT *
            FROM defensive_validation_summary_dataframe
            ORDER BY
                sample_id,
                model_id,
                additional_distance_fraction
            """
        )

        forward = validation.loc[
            validation[
                "sample_id"
            ] == "forward_test"
        ]

        print(
            "\nForward-test event concentration and "
            "cluster-removal validation"
        )
        print("-" * 188)

        for row in forward.itertuples(
            index=False
        ):
            print(
                f"{row.model_label:<39} | "
                f"Extra={row.additional_distance_percent:>4.1f}% | "
                f"Failures {row.standard_elevated_failures:>2}"
                f"->{row.defensive_elevated_failures:<2} | "
                f"Haircut={row.maximum_tolerable_credit_haircut_points:>5.2f} | "
                f"Top cluster={row.top_cluster_intrinsic_savings_share:>6.1%} | "
                f"Contrib clusters={row.contributing_failure_clusters:>2} | "
                f"LOO min haircut={row.minimum_leave_one_cluster_haircut_points:>5.2f} "
                f"({row.minimum_haircut_retention_ratio:>5.1%}) | "
                f"Positive LOO={row.positive_savings_cluster_removals:>2}/"
                f"{row.leave_one_cluster_tests:<2} | "
                f"{row.concentration_assessment}"
            )

        print(
            "\nLargest forward-test contributing events"
        )
        print("-" * 138)

        forward_attribution = (
            attribution.loc[
                (
                    attribution[
                        "sample_id"
                    ] == "forward_test"
                )
                & (
                    attribution[
                        "intrinsic_points_saved_in_cluster"
                    ] > 0
                )
            ]
            .sort_values(
                [
                    "model_id",
                    "additional_distance_fraction",
                    "intrinsic_points_saved_in_cluster",
                ],
                ascending=[
                    True,
                    True,
                    False,
                ],
            )
        )

        for (
            model_id,
            distance,
        ), group in forward_attribution.groupby(
            [
                "model_id",
                "additional_distance_fraction",
            ],
            sort=False,
        ):
            model_label = str(
                group[
                    "model_label"
                ].iloc[0]
            )

            print(
                f"\n{model_label} | "
                f"Extra={distance:.1%}"
            )

            for row in group.head(
                3
            ).itertuples(index=False):
                print(
                    f"  Cluster {row.cluster_id:>2}: "
                    f"{row.cluster_start.date()} to "
                    f"{row.cluster_end.date()} | "
                    f"Saved={row.intrinsic_points_saved_in_cluster:>6.2f} pts "
                    f"({row.intrinsic_savings_share:>6.1%}) | "
                    f"Failures "
                    f"{row.elevated_standard_failures_in_cluster}"
                    f"->{row.elevated_defensive_failures_in_cluster}"
                )

        summary_lines = [
            (
                "MNQ PCS REGIME RESEARCH — "
                "DEFENSIVE STRIKE CLUSTER VALIDATION"
            ),
            "=" * 68,
            "",
            "Methodological status",
            "-" * 68,
            (
                "Fixed-credit expiration proxy; not a historical "
                "options backtest."
            ),
            (
                "Leave-one-cluster-out tests remove every trade "
                "whose holding period overlaps the selected failure "
                "cluster."
            ),
            (
                "The concentration assessment is descriptive, not "
                "an operational trading rule."
            ),
            f"Entry spot source: {entry_spot_source}",
            f"Short strike source: {short_source}",
            f"Settlement source: {settlement_source}",
            "",
            "Forward-test validation",
            "-" * 68,
        ]

        for row in forward.itertuples(
            index=False
        ):
            summary_lines.append(
                f"{row.model_label} | "
                f"Extra={row.additional_distance_percent:.1f}% | "
                f"Failures={row.standard_elevated_failures}"
                f"->{row.defensive_elevated_failures} | "
                f"Full haircut="
                f"{row.maximum_tolerable_credit_haircut_points:.2f} | "
                f"Top-cluster share="
                f"{row.top_cluster_intrinsic_savings_share:.1%} | "
                f"LOO minimum haircut="
                f"{row.minimum_leave_one_cluster_haircut_points:.2f} | "
                f"Retention="
                f"{row.minimum_haircut_retention_ratio:.1%} | "
                f"Assessment="
                f"{row.concentration_assessment}"
            )

        SUMMARY_OUTPUT.write_text(
            "\n".join(
                summary_lines
            ) + "\n",
            encoding="utf-8",
        )

        print("\nMethodological status")
        print("-" * 92)

        print(
            "Event attribution and leave-one-cluster-out "
            "robustness only; actual defensive premiums remain "
            "unknown."
        )

        print(
            f"\nAttribution output: "
            f"{ATTRIBUTION_OUTPUT}"
        )

        print(
            f"Cluster output: "
            f"{CLUSTER_OUTPUT}"
        )

        print(
            f"Validation output: "
            f"{VALIDATION_OUTPUT}"
        )

        print(
            f"Summary output: "
            f"{SUMMARY_OUTPUT}"
        )

        print(
            "DuckDB tables: "
            "pcs.defensive_strike_event_attribution, "
            "pcs.defensive_strike_cluster_validation, "
            "pcs.defensive_strike_validation_summary"
        )

        print(
            "Defensive strike cluster validation "
            "completed successfully."
        )

        return 0

    except Exception as exc:
        print(
            f"\nERROR: {exc}"
        )
        return 1

    finally:
        connection.close()


if __name__ == "__main__":
    sys.exit(main())

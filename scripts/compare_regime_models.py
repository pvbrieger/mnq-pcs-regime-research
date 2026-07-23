"""Compare the three-factor PCS regime score with simpler alternatives.

The purpose is to determine whether the full score earns its added complexity.
All model definitions are fixed before evaluation. Each chronological test fold
uses only earlier observations to estimate normal and elevated failure
probabilities.

Inputs
------
- database/research.duckdb
- config/candidate_regime_score.yaml
- pcs.friday_regime_features

Outputs
-------
- results/model_comparison/regime_model_comparison.csv
- results/model_comparison/regime_model_fold_results.csv
- results/model_comparison/regime_model_disagreement.csv
- results/model_comparison/regime_model_comparison_summary.txt
- DuckDB tables:
    pcs.regime_model_comparison
    pcs.regime_model_fold_results
    pcs.regime_model_disagreement
"""

from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Any, Callable

import duckdb
import numpy as np
import pandas as pd
import yaml
from scipy.stats import fisher_exact


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATABASE_PATH = PROJECT_ROOT / "database" / "research.duckdb"
CONFIG_PATH = PROJECT_ROOT / "config" / "candidate_regime_score.yaml"

RESULTS_DIR = PROJECT_ROOT / "results" / "model_comparison"
COMPARISON_OUTPUT = RESULTS_DIR / "regime_model_comparison.csv"
FOLD_OUTPUT = RESULTS_DIR / "regime_model_fold_results.csv"
DISAGREEMENT_OUTPUT = RESULTS_DIR / "regime_model_disagreement.csv"
SUMMARY_OUTPUT = RESULTS_DIR / "regime_model_comparison_summary.txt"

TEST_FOLDS = (
    ("2011-2013", "2011-01-01", "2013-12-31"),
    ("2014-2016", "2014-01-01", "2016-12-31"),
    ("2017-2019", "2017-01-01", "2019-12-31"),
    ("2020-2022", "2020-01-01", "2022-12-31"),
    ("2023-2026", "2023-01-01", "2026-12-31"),
)

JEFFREYS_PRIOR_SUCCESS = 0.5
JEFFREYS_PRIOR_TOTAL = 1.0
PROBABILITY_FLOOR = 1e-9


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
    """Evaluate one configured factor rule."""

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


def posterior_probability(
    failures: int,
    observations: int,
) -> float:
    """Estimate a failure probability using a Jeffreys prior."""

    return (
        failures + JEFFREYS_PRIOR_SUCCESS
    ) / (
        observations + JEFFREYS_PRIOR_TOTAL
    )


def brier_score(
    actual: np.ndarray,
    predicted: np.ndarray,
) -> float:
    """Calculate Brier score."""

    return float(
        np.mean(
            (
                predicted.astype(float)
                - actual.astype(float)
            )
            ** 2
        )
    )


def binary_log_loss(
    actual: np.ndarray,
    predicted: np.ndarray,
) -> float:
    """Calculate binary logarithmic loss."""

    probability = np.clip(
        predicted.astype(float),
        PROBABILITY_FLOOR,
        1.0 - PROBABILITY_FLOOR,
    )

    actual_float = actual.astype(float)

    loss = -(
        actual_float * np.log(probability)
        + (1.0 - actual_float) * np.log(1.0 - probability)
    )

    return float(np.mean(loss))


def summarize_rule(
    dataframe: pd.DataFrame,
    elevated_mask: pd.Series,
) -> dict[str, Any]:
    """Summarize outcomes for one binary regime definition."""

    elevated_mask = elevated_mask.fillna(False).astype(bool)
    normal_mask = ~elevated_mask

    elevated = dataframe.loc[elevated_mask]
    normal = dataframe.loc[normal_mask]

    elevated_failures = int(
        elevated["expiration_failure"].sum()
    )

    normal_failures = int(
        normal["expiration_failure"].sum()
    )

    elevated_rate = safe_divide(
        elevated_failures,
        len(elevated),
    )

    normal_rate = safe_divide(
        normal_failures,
        len(normal),
    )

    if (
        not pd.isna(normal_rate)
        and normal_rate > 0
        and not pd.isna(elevated_rate)
    ):
        lift = elevated_rate / normal_rate
    elif (
        normal_rate == 0
        and not pd.isna(elevated_rate)
        and elevated_rate > 0
    ):
        lift = math.inf
    else:
        lift = np.nan

    if len(elevated) and len(normal):
        contingency = [
            [
                elevated_failures,
                len(elevated) - elevated_failures,
            ],
            [
                normal_failures,
                len(normal) - normal_failures,
            ],
        ]

        odds_ratio, fisher_p_value = fisher_exact(
            contingency,
            alternative="two-sided",
        )
    else:
        odds_ratio = np.nan
        fisher_p_value = np.nan

    total_failures = int(
        dataframe["expiration_failure"].sum()
    )

    observation_share = safe_divide(
        len(elevated),
        len(dataframe),
    )

    failure_capture = safe_divide(
        elevated_failures,
        total_failures,
    )

    return {
        "observations": len(dataframe),
        "failures": total_failures,
        "normal_observations": len(normal),
        "normal_failures": normal_failures,
        "normal_failure_rate": normal_rate,
        "elevated_observations": len(elevated),
        "elevated_observation_share": observation_share,
        "elevated_failures": elevated_failures,
        "elevated_failure_rate": elevated_rate,
        "elevated_rate_lift_vs_normal": lift,
        "elevated_failure_capture": failure_capture,
        "failure_capture_efficiency": safe_divide(
            failure_capture,
            observation_share,
        ),
        "odds_ratio": float(odds_ratio),
        "fisher_p_value": float(fisher_p_value),
        "direction_supportive": bool(
            not pd.isna(elevated_rate)
            and not pd.isna(normal_rate)
            and elevated_rate > normal_rate
        ),
    }


def format_rate(value: float) -> str:
    """Format percentage values."""

    if pd.isna(value):
        return "n/a"

    return f"{value:.2%}"


def format_lift(value: float) -> str:
    """Format risk-lift values."""

    if pd.isna(value):
        return "n/a"

    if math.isinf(value):
        return "inf"

    return f"{value:.2f}x"


def main() -> int:
    """Compare full and simplified regime definitions."""

    print("MNQ PCS Regime Research — Model Simplicity Comparison")
    print("=" * 59)

    if not DATABASE_PATH.is_file():
        print(f"ERROR: Missing database: {DATABASE_PATH}")
        return 1

    if not CONFIG_PATH.is_file():
        print(f"ERROR: Missing configuration: {CONFIG_PATH}")
        return 1

    with CONFIG_PATH.open(
        "r",
        encoding="utf-8",
    ) as file_handle:
        config = yaml.safe_load(file_handle)

    factors = config.get("factors", [])

    factor_by_id = {
        str(factor["id"]): factor
        for factor in factors
    }

    required_ids = {
        "rolling_four_week_expansion",
        "below_40_week_sma",
        "weekly_expansion",
    }

    missing_ids = required_ids - set(factor_by_id)

    if missing_ids:
        print(
            "ERROR: Candidate configuration is missing factors: "
            + ", ".join(sorted(missing_ids))
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

        dataframe["expiration_failure"] = (
            dataframe["expiration_failure"]
            .fillna(False)
            .astype(bool)
        )

        for factor_id, factor in factor_by_id.items():
            source_column = str(factor["column"])

            if source_column not in dataframe.columns:
                raise ValueError(
                    f"Missing feature column: {source_column}"
                )

            dataframe[f"flag_{factor_id}"] = evaluate_rule(
                dataframe[source_column],
                str(factor["operator"]),
                factor["value"],
            )

        rolling_flag = dataframe[
            "flag_rolling_four_week_expansion"
        ]

        weekly_flag = dataframe[
            "flag_weekly_expansion"
        ]

        trend_flag = dataframe[
            "flag_below_40_week_sma"
        ]

        full_score = (
            rolling_flag.astype(int)
            + weekly_flag.astype(int)
            + trend_flag.astype(int)
        )

        dataframe["model_full_score"] = full_score >= 2
        dataframe["model_any_one_flag"] = full_score >= 1
        dataframe["model_rolling_atr_only"] = rolling_flag
        dataframe["model_weekly_atr_only"] = weekly_flag
        dataframe["model_trend_only"] = trend_flag
        dataframe["model_atr_any"] = rolling_flag | weekly_flag
        dataframe["model_atr_both"] = rolling_flag & weekly_flag

        model_definitions = [
            {
                "model_id": "full_three_factor_score",
                "model_label": "Full score: at least 2 of 3 flags",
                "complexity": 3,
                "column": "model_full_score",
            },
            {
                "model_id": "any_one_flag",
                "model_label": "Any one of the three flags",
                "complexity": 3,
                "column": "model_any_one_flag",
            },
            {
                "model_id": "rolling_atr_only",
                "model_label": "Rolling four-week ATR expansion only",
                "complexity": 1,
                "column": "model_rolling_atr_only",
            },
            {
                "model_id": "weekly_atr_only",
                "model_label": "Weekly ATR expansion only",
                "complexity": 1,
                "column": "model_weekly_atr_only",
            },
            {
                "model_id": "trend_only",
                "model_label": "Below 40-week SMA only",
                "complexity": 1,
                "column": "model_trend_only",
            },
            {
                "model_id": "atr_any",
                "model_label": "Either ATR expansion flag",
                "complexity": 2,
                "column": "model_atr_any",
            },
            {
                "model_id": "atr_both",
                "model_label": "Both ATR expansion flags",
                "complexity": 2,
                "column": "model_atr_both",
            },
        ]

        print(
            f"Complete history: {len(dataframe):,} observations | "
            f"{int(dataframe['expiration_failure'].sum())} failures"
        )

        print(
            f"Models compared: {len(model_definitions)} | "
            f"Chronological folds: {len(TEST_FOLDS)}"
        )

        fold_rows: list[dict[str, Any]] = []
        pooled_prediction_frames: list[pd.DataFrame] = []

        for fold_number, (
            fold_label,
            test_start_text,
            test_end_text,
        ) in enumerate(TEST_FOLDS, start=1):
            test_start = pd.Timestamp(test_start_text)
            test_end = pd.Timestamp(test_end_text)

            training = dataframe.loc[
                dataframe["entry_date"] < test_start
            ].copy()

            testing = dataframe.loc[
                dataframe["entry_date"].between(
                    test_start,
                    test_end,
                )
            ].copy()

            if training.empty or testing.empty:
                raise ValueError(
                    f"Fold {fold_label} lacks training or test data."
                )

            if (
                training["entry_date"].max()
                >= testing["entry_date"].min()
            ):
                raise ValueError(
                    f"Look-ahead detected in fold {fold_label}."
                )

            baseline_probability = posterior_probability(
                int(training["expiration_failure"].sum()),
                len(training),
            )

            actual = (
                testing["expiration_failure"]
                .astype(int)
                .to_numpy()
            )

            baseline_predictions = np.full(
                len(testing),
                baseline_probability,
                dtype=float,
            )

            baseline_brier = brier_score(
                actual,
                baseline_predictions,
            )

            baseline_log_loss = binary_log_loss(
                actual,
                baseline_predictions,
            )

            for model in model_definitions:
                column = str(model["column"])

                train_summary = summarize_rule(
                    training,
                    training[column],
                )

                test_summary = summarize_rule(
                    testing,
                    testing[column],
                )

                train_normal_probability = (
                    posterior_probability(
                        train_summary["normal_failures"],
                        train_summary["normal_observations"],
                    )
                )

                train_elevated_probability = (
                    posterior_probability(
                        train_summary["elevated_failures"],
                        train_summary["elevated_observations"],
                    )
                )

                model_predictions = np.where(
                    testing[column].to_numpy(),
                    train_elevated_probability,
                    train_normal_probability,
                )

                model_brier = brier_score(
                    actual,
                    model_predictions,
                )

                model_log_loss = binary_log_loss(
                    actual,
                    model_predictions,
                )

                brier_skill = (
                    1.0 - model_brier / baseline_brier
                    if baseline_brier > 0
                    else np.nan
                )

                fold_rows.append(
                    {
                        "fold_number": fold_number,
                        "fold_label": fold_label,
                        "model_id": model["model_id"],
                        "model_label": model["model_label"],
                        "complexity": model["complexity"],
                        "training_start": training[
                            "entry_date"
                        ].min(),
                        "training_end": training[
                            "entry_date"
                        ].max(),
                        "test_start": testing[
                            "entry_date"
                        ].min(),
                        "test_end": testing[
                            "entry_date"
                        ].max(),
                        **{
                            f"train_{key}": value
                            for key, value in train_summary.items()
                        },
                        **{
                            f"test_{key}": value
                            for key, value in test_summary.items()
                        },
                        "train_normal_probability": (
                            train_normal_probability
                        ),
                        "train_elevated_probability": (
                            train_elevated_probability
                        ),
                        "train_baseline_probability": (
                            baseline_probability
                        ),
                        "test_model_brier_score": model_brier,
                        "test_baseline_brier_score": (
                            baseline_brier
                        ),
                        "test_brier_skill_vs_baseline": (
                            brier_skill
                        ),
                        "test_model_log_loss": model_log_loss,
                        "test_baseline_log_loss": (
                            baseline_log_loss
                        ),
                        "test_log_loss_improvement": (
                            baseline_log_loss - model_log_loss
                        ),
                        "test_informative": bool(
                            test_summary["failures"] > 0
                        ),
                    }
                )

                prediction_frame = pd.DataFrame(
                    {
                        "entry_date": testing["entry_date"],
                        "expiration_failure": actual,
                        "fold_number": fold_number,
                        "fold_label": fold_label,
                        "model_id": model["model_id"],
                        "model_label": model["model_label"],
                        "complexity": model["complexity"],
                        "elevated_regime": testing[
                            column
                        ].to_numpy(),
                        "model_predicted_probability": (
                            model_predictions
                        ),
                        "baseline_predicted_probability": (
                            baseline_predictions
                        ),
                    }
                )

                pooled_prediction_frames.append(
                    prediction_frame
                )

        fold_results = pd.DataFrame(fold_rows)

        pooled_predictions = pd.concat(
            pooled_prediction_frames,
            ignore_index=True,
        )

        comparison_rows: list[dict[str, Any]] = []

        for model in model_definitions:
            model_predictions = pooled_predictions.loc[
                pooled_predictions["model_id"]
                == model["model_id"]
            ].sort_values("entry_date")

            model_test_data = dataframe.loc[
                dataframe["entry_date"].isin(
                    model_predictions["entry_date"]
                )
            ].sort_values("entry_date")

            if len(model_test_data) != len(model_predictions):
                raise ValueError(
                    f"Pooled row mismatch for {model['model_id']}."
                )

            column = str(model["column"])

            pooled_summary = summarize_rule(
                model_test_data,
                model_test_data[column],
            )

            actual = (
                model_predictions["expiration_failure"]
                .astype(int)
                .to_numpy()
            )

            predicted = model_predictions[
                "model_predicted_probability"
            ].to_numpy()

            baseline_predicted = model_predictions[
                "baseline_predicted_probability"
            ].to_numpy()

            model_brier = brier_score(
                actual,
                predicted,
            )

            baseline_brier = brier_score(
                actual,
                baseline_predicted,
            )

            model_log_loss = binary_log_loss(
                actual,
                predicted,
            )

            baseline_log_loss = binary_log_loss(
                actual,
                baseline_predicted,
            )

            model_fold_rows = fold_results.loc[
                fold_results["model_id"]
                == model["model_id"]
            ]

            informative_folds = model_fold_rows.loc[
                model_fold_rows["test_informative"]
            ]

            supportive_test_folds = int(
                model_fold_rows[
                    "test_direction_supportive"
                ].sum()
            )

            supportive_informative_folds = int(
                informative_folds[
                    "test_direction_supportive"
                ].sum()
            )

            positive_brier_folds = int(
                (
                    model_fold_rows[
                        "test_brier_skill_vs_baseline"
                    ] > 0
                ).sum()
            )

            comparison_rows.append(
                {
                    "model_id": model["model_id"],
                    "model_label": model["model_label"],
                    "complexity": model["complexity"],
                    **pooled_summary,
                    "pooled_model_brier_score": model_brier,
                    "pooled_baseline_brier_score": (
                        baseline_brier
                    ),
                    "pooled_brier_skill_vs_baseline": (
                        1.0 - model_brier / baseline_brier
                        if baseline_brier > 0
                        else np.nan
                    ),
                    "pooled_model_log_loss": model_log_loss,
                    "pooled_baseline_log_loss": (
                        baseline_log_loss
                    ),
                    "pooled_log_loss_improvement": (
                        baseline_log_loss - model_log_loss
                    ),
                    "supportive_test_folds": (
                        supportive_test_folds
                    ),
                    "informative_test_folds": (
                        len(informative_folds)
                    ),
                    "supportive_informative_folds": (
                        supportive_informative_folds
                    ),
                    "positive_brier_folds": (
                        positive_brier_folds
                    ),
                }
            )

        comparison = pd.DataFrame(comparison_rows)

        full_row = comparison.loc[
            comparison["model_id"]
            == "full_three_factor_score"
        ].iloc[0]

        comparison["brier_difference_vs_full"] = (
            comparison["pooled_model_brier_score"]
            - full_row["pooled_model_brier_score"]
        )

        comparison["log_loss_difference_vs_full"] = (
            comparison["pooled_model_log_loss"]
            - full_row["pooled_model_log_loss"]
        )

        comparison["failure_rate_lift_difference_vs_full"] = (
            comparison["elevated_rate_lift_vs_normal"]
            - full_row["elevated_rate_lift_vs_normal"]
        )

        comparison["capture_efficiency_difference_vs_full"] = (
            comparison["failure_capture_efficiency"]
            - full_row["failure_capture_efficiency"]
        )

        comparison = comparison.sort_values(
            [
                "pooled_model_brier_score",
                "pooled_model_log_loss",
                "complexity",
            ],
            ascending=[True, True, True],
        ).reset_index(drop=True)

        comparison.insert(
            0,
            "predictive_rank",
            np.arange(1, len(comparison) + 1),
        )

        disagreement_rows: list[dict[str, Any]] = []

        test_start = pd.Timestamp(TEST_FOLDS[0][1])
        test_end = pd.Timestamp(TEST_FOLDS[-1][2])

        pooled_test_data = dataframe.loc[
            dataframe["entry_date"].between(
                test_start,
                test_end,
            )
        ].copy()

        full_mask = pooled_test_data[
            "model_full_score"
        ]

        for model in model_definitions:
            model_mask = pooled_test_data[
                str(model["column"])
            ]

            both = full_mask & model_mask
            full_only = full_mask & ~model_mask
            simple_only = ~full_mask & model_mask
            neither = ~full_mask & ~model_mask

            disagreement_rows.append(
                {
                    "model_id": model["model_id"],
                    "model_label": model["model_label"],
                    "both_elevated_observations": int(
                        both.sum()
                    ),
                    "both_elevated_failures": int(
                        pooled_test_data.loc[
                            both,
                            "expiration_failure",
                        ].sum()
                    ),
                    "full_only_observations": int(
                        full_only.sum()
                    ),
                    "full_only_failures": int(
                        pooled_test_data.loc[
                            full_only,
                            "expiration_failure",
                        ].sum()
                    ),
                    "model_only_observations": int(
                        simple_only.sum()
                    ),
                    "model_only_failures": int(
                        pooled_test_data.loc[
                            simple_only,
                            "expiration_failure",
                        ].sum()
                    ),
                    "neither_observations": int(
                        neither.sum()
                    ),
                    "neither_failures": int(
                        pooled_test_data.loc[
                            neither,
                            "expiration_failure",
                        ].sum()
                    ),
                    "agreement_rate": float(
                        (full_mask == model_mask).mean()
                    ),
                    "jaccard_elevated_overlap": safe_divide(
                        int(both.sum()),
                        int((full_mask | model_mask).sum()),
                    ),
                }
            )

        disagreement = pd.DataFrame(disagreement_rows)

        RESULTS_DIR.mkdir(
            parents=True,
            exist_ok=True,
        )

        comparison.to_csv(
            COMPARISON_OUTPUT,
            index=False,
        )

        fold_results.to_csv(
            FOLD_OUTPUT,
            index=False,
        )

        disagreement.to_csv(
            DISAGREEMENT_OUTPUT,
            index=False,
        )

        connection.register(
            "regime_model_comparison_dataframe",
            comparison,
        )

        connection.register(
            "regime_model_fold_results_dataframe",
            fold_results,
        )

        connection.register(
            "regime_model_disagreement_dataframe",
            disagreement,
        )

        connection.execute(
            """
            CREATE OR REPLACE TABLE
                pcs.regime_model_comparison AS
            SELECT *
            FROM regime_model_comparison_dataframe
            ORDER BY predictive_rank
            """
        )

        connection.execute(
            """
            CREATE OR REPLACE TABLE
                pcs.regime_model_fold_results AS
            SELECT *
            FROM regime_model_fold_results_dataframe
            ORDER BY model_id, fold_number
            """
        )

        connection.execute(
            """
            CREATE OR REPLACE TABLE
                pcs.regime_model_disagreement AS
            SELECT *
            FROM regime_model_disagreement_dataframe
            ORDER BY model_id
            """
        )

        best_row = comparison.iloc[0]

        full_rank = int(
            comparison.loc[
                comparison["model_id"]
                == "full_three_factor_score",
                "predictive_rank",
            ].iloc[0]
        )

        best_simple = comparison.loc[
            comparison["model_id"]
            != "full_three_factor_score"
        ].iloc[0]

        if full_rank == 1:
            complexity_assessment = (
                "The full three-factor score produced the best "
                "forward probability performance."
            )
        elif (
            full_row["pooled_model_brier_score"]
            - best_simple["pooled_model_brier_score"]
        ) <= 0.00005:
            complexity_assessment = (
                "A simpler rule matched the full score closely; "
                "simplicity may deserve preference."
            )
        else:
            complexity_assessment = (
                "A simpler rule materially outperformed the full "
                "score; the added complexity is not currently earned."
            )

        summary_lines = [
            "MNQ PCS REGIME RESEARCH — MODEL SIMPLICITY COMPARISON",
            "=" * 61,
            "",
            "Purpose",
            "-" * 61,
            (
                "Determine whether the full three-factor score "
                "earns its added complexity versus simpler rules."
            ),
            "",
            "Pooled chronological test set",
            "-" * 61,
            (
                f"Observations: "
                f"{int(full_row['observations']):,}"
            ),
            (
                f"Failures: "
                f"{int(full_row['failures'])}"
            ),
            "",
            "Predictive ranking",
            "-" * 61,
        ]

        for row in comparison.itertuples(index=False):
            summary_lines.append(
                f"{row.predictive_rank}. {row.model_label} | "
                f"Elevated N={row.elevated_observations} "
                f"({row.elevated_observation_share:.1%}) | "
                f"Failures={row.elevated_failures} "
                f"({row.elevated_failure_rate:.2%}) | "
                f"Lift={format_lift(row.elevated_rate_lift_vs_normal)} | "
                f"Captured={row.elevated_failure_capture:.1%} | "
                f"Brier={row.pooled_model_brier_score:.6f} | "
                f"Skill={row.pooled_brier_skill_vs_baseline:.2%}"
            )

        summary_lines.extend(
            [
                "",
                "Complexity assessment",
                "-" * 61,
                complexity_assessment,
                "",
                (
                    f"Best predictive model: "
                    f"{best_row['model_label']}"
                ),
                (
                    f"Full-score predictive rank: "
                    f"{full_rank}/{len(comparison)}"
                ),
                (
                    f"Best simpler model: "
                    f"{best_simple['model_label']}"
                ),
            ]
        )

        SUMMARY_OUTPUT.write_text(
            "\n".join(summary_lines) + "\n",
            encoding="utf-8",
        )

        print("\nPooled forward-test comparison")
        print("-" * 152)

        for row in comparison.itertuples(index=False):
            print(
                f"{row.predictive_rank:>2}. "
                f"{row.model_label:<40} | "
                f"N={row.elevated_observations:>3} "
                f"({row.elevated_observation_share:>5.1%}) | "
                f"Fail={row.elevated_failures:>2} "
                f"({row.elevated_failure_rate:>6.2%}) | "
                f"Lift={format_lift(row.elevated_rate_lift_vs_normal):>6} | "
                f"Capture={row.elevated_failure_capture:>5.1%} | "
                f"Eff={row.failure_capture_efficiency:>4.2f}x | "
                f"Brier={row.pooled_model_brier_score:.6f} | "
                f"Skill={row.pooled_brier_skill_vs_baseline:>6.2%}"
            )

        print("\nComplexity assessment")
        print("-" * 92)

        print(complexity_assessment)

        print(
            f"Best model: {best_row['model_label']} | "
            f"Full-score rank: {full_rank}/{len(comparison)} | "
            f"Best simpler model: {best_simple['model_label']}"
        )

        print(f"\nComparison output: {COMPARISON_OUTPUT}")
        print(f"Fold output: {FOLD_OUTPUT}")
        print(f"Disagreement output: {DISAGREEMENT_OUTPUT}")
        print(f"Summary output: {SUMMARY_OUTPUT}")
        print(
            "DuckDB tables: pcs.regime_model_comparison, "
            "pcs.regime_model_fold_results, "
            "pcs.regime_model_disagreement"
        )

        print(
            "Model simplicity comparison completed successfully."
        )

        return 0

    except Exception as exc:
        print(f"\nERROR: {exc}")
        return 1

    finally:
        connection.close()


if __name__ == "__main__":
    sys.exit(main())

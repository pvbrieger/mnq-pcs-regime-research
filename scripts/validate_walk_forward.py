"""Run forward-chaining validation of the fixed Friday PCS regime score.

Important methodological note
-----------------------------
The candidate score was discovered using the full historical dataset. Therefore,
this is a chronological stability and pseudo-out-of-sample diagnostic, not a
pristine untouched holdout test. The score definition is frozen during this
script, and each test fold uses only earlier observations to estimate failure
probabilities.

Inputs
------
- database/research.duckdb
- config/candidate_regime_score.yaml
- pcs.friday_regime_features

Outputs
-------
- results/walk_forward_validation/walk_forward_folds.csv
- results/walk_forward_validation/walk_forward_yearly.csv
- results/walk_forward_validation/walk_forward_summary.txt
- DuckDB tables:
    pcs.regime_walk_forward_folds
    pcs.regime_walk_forward_yearly
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
from scipy.stats import fisher_exact
from statsmodels.stats.proportion import proportion_confint


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATABASE_PATH = PROJECT_ROOT / "database" / "research.duckdb"
CONFIG_PATH = PROJECT_ROOT / "config" / "candidate_regime_score.yaml"

RESULTS_DIR = PROJECT_ROOT / "results" / "walk_forward_validation"
FOLD_OUTPUT = RESULTS_DIR / "walk_forward_folds.csv"
YEARLY_OUTPUT = RESULTS_DIR / "walk_forward_yearly.csv"
SUMMARY_OUTPUT = RESULTS_DIR / "walk_forward_summary.txt"

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


def wilson_interval(
    successes: int,
    observations: int,
) -> tuple[float, float]:
    """Return a 95% Wilson confidence interval."""

    if observations == 0:
        return np.nan, np.nan

    lower, upper = proportion_confint(
        count=successes,
        nobs=observations,
        alpha=0.05,
        method="wilson",
    )

    return float(lower), float(upper)


def posterior_failure_probability(
    failures: int,
    observations: int,
) -> float:
    """Estimate failure probability using a Jeffreys prior."""

    return (
        failures + JEFFREYS_PRIOR_SUCCESS
    ) / (
        observations + JEFFREYS_PRIOR_TOTAL
    )


def binary_log_loss(
    actual: np.ndarray,
    predicted: np.ndarray,
) -> float:
    """Calculate binary logarithmic loss."""

    probabilities = np.clip(
        predicted.astype(float),
        PROBABILITY_FLOOR,
        1.0 - PROBABILITY_FLOOR,
    )

    actual_values = actual.astype(float)

    losses = -(
        actual_values * np.log(probabilities)
        + (1.0 - actual_values) * np.log(1.0 - probabilities)
    )

    return float(np.mean(losses))


def brier_score(
    actual: np.ndarray,
    predicted: np.ndarray,
) -> float:
    """Calculate the Brier score."""

    return float(
        np.mean(
            (
                predicted.astype(float)
                - actual.astype(float)
            )
            ** 2
        )
    )


def summarize_regimes(
    dataframe: pd.DataFrame,
) -> dict[str, Any]:
    """Summarize normal and elevated outcome rates."""

    normal = dataframe.loc[~dataframe["elevated_regime"]]
    elevated = dataframe.loc[dataframe["elevated_regime"]]

    normal_failures = int(normal["expiration_failure"].sum())
    elevated_failures = int(
        elevated["expiration_failure"].sum()
    )

    normal_rate = safe_divide(normal_failures, len(normal))
    elevated_rate = safe_divide(
        elevated_failures,
        len(elevated),
    )

    normal_ci_lower, normal_ci_upper = wilson_interval(
        normal_failures,
        len(normal),
    )

    elevated_ci_lower, elevated_ci_upper = wilson_interval(
        elevated_failures,
        len(elevated),
    )

    if len(normal) and len(elevated):
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

    if (
        not pd.isna(normal_rate)
        and normal_rate > 0
        and not pd.isna(elevated_rate)
    ):
        rate_lift = elevated_rate / normal_rate
    elif (
        normal_rate == 0
        and not pd.isna(elevated_rate)
        and elevated_rate > 0
    ):
        rate_lift = math.inf
    else:
        rate_lift = np.nan

    total_failures = int(
        dataframe["expiration_failure"].sum()
    )

    return {
        "observations": len(dataframe),
        "failures": total_failures,
        "overall_failure_rate": safe_divide(
            total_failures,
            len(dataframe),
        ),
        "normal_observations": len(normal),
        "normal_failures": normal_failures,
        "normal_failure_rate": normal_rate,
        "normal_ci_lower": normal_ci_lower,
        "normal_ci_upper": normal_ci_upper,
        "elevated_observations": len(elevated),
        "elevated_observation_share": safe_divide(
            len(elevated),
            len(dataframe),
        ),
        "elevated_failures": elevated_failures,
        "elevated_failure_rate": elevated_rate,
        "elevated_ci_lower": elevated_ci_lower,
        "elevated_ci_upper": elevated_ci_upper,
        "elevated_rate_lift_vs_normal": rate_lift,
        "elevated_failure_capture": safe_divide(
            elevated_failures,
            total_failures,
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
    """Format a rate for terminal output."""

    if pd.isna(value):
        return "n/a"

    return f"{value:.2%}"


def format_lift(value: float) -> str:
    """Format a relative-risk lift."""

    if pd.isna(value):
        return "n/a"

    if math.isinf(value):
        return "inf"

    return f"{value:.2f}x"


def main() -> int:
    """Run forward-chaining validation."""

    print(
        "MNQ PCS Regime Research — Forward-Chaining Validation"
    )
    print("=" * 58)

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
    regimes = config.get("regimes", {})

    if not factors:
        print("ERROR: No candidate-score factors were configured.")
        return 1

    elevated_minimum = int(
        regimes["elevated"]["minimum_score"]
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

            factor_columns.append(flag_column)

        dataframe["candidate_score"] = (
            dataframe[factor_columns]
            .astype(int)
            .sum(axis=1)
        )

        dataframe["elevated_regime"] = (
            dataframe["candidate_score"]
            >= elevated_minimum
        )

        if dataframe["entry_date"].duplicated().any():
            raise ValueError(
                "Duplicate entry dates detected."
            )

        print(
            f"Fixed score factors: {len(factors)} | "
            f"Elevated cutoff: score >= {elevated_minimum}"
        )

        print(
            f"Complete history: {len(dataframe):,} observations | "
            f"{int(dataframe['expiration_failure'].sum())} failures"
        )

        fold_rows: list[dict[str, Any]] = []
        prediction_frames: list[pd.DataFrame] = []

        print("\nChronological test folds")
        print("-" * 142)

        previous_test_end: pd.Timestamp | None = None

        for fold_number, (
            fold_label,
            test_start_text,
            test_end_text,
        ) in enumerate(TEST_FOLDS, start=1):
            test_start = pd.Timestamp(test_start_text)
            test_end = pd.Timestamp(test_end_text)

            if (
                previous_test_end is not None
                and test_start <= previous_test_end
            ):
                raise ValueError(
                    "Configured test folds overlap."
                )

            training = dataframe.loc[
                dataframe["entry_date"] < test_start
            ].copy()

            testing = dataframe.loc[
                dataframe["entry_date"].between(
                    test_start,
                    test_end,
                )
            ].copy()

            if training.empty:
                raise ValueError(
                    f"Fold {fold_label} has no training observations."
                )

            if testing.empty:
                raise ValueError(
                    f"Fold {fold_label} has no test observations."
                )

            if (
                training["entry_date"].max()
                >= testing["entry_date"].min()
            ):
                raise ValueError(
                    f"Look-ahead detected in fold {fold_label}."
                )

            previous_test_end = test_end

            training_summary = summarize_regimes(training)
            test_summary = summarize_regimes(testing)

            train_normal_probability = (
                posterior_failure_probability(
                    training_summary["normal_failures"],
                    training_summary["normal_observations"],
                )
            )

            train_elevated_probability = (
                posterior_failure_probability(
                    training_summary["elevated_failures"],
                    training_summary["elevated_observations"],
                )
            )

            train_baseline_probability = (
                posterior_failure_probability(
                    training_summary["failures"],
                    training_summary["observations"],
                )
            )

            testing["fold_number"] = fold_number
            testing["fold_label"] = fold_label

            testing["regime_predicted_probability"] = np.where(
                testing["elevated_regime"],
                train_elevated_probability,
                train_normal_probability,
            )

            testing["baseline_predicted_probability"] = (
                train_baseline_probability
            )

            actual = (
                testing["expiration_failure"]
                .astype(int)
                .to_numpy()
            )

            regime_predictions = testing[
                "regime_predicted_probability"
            ].to_numpy()

            baseline_predictions = testing[
                "baseline_predicted_probability"
            ].to_numpy()

            regime_brier = brier_score(
                actual,
                regime_predictions,
            )

            baseline_brier = brier_score(
                actual,
                baseline_predictions,
            )

            brier_skill = (
                1.0 - regime_brier / baseline_brier
                if baseline_brier > 0
                else np.nan
            )

            regime_log_loss = binary_log_loss(
                actual,
                regime_predictions,
            )

            baseline_log_loss = binary_log_loss(
                actual,
                baseline_predictions,
            )

            fold_row = {
                "fold_number": fold_number,
                "fold_label": fold_label,
                "training_start": training[
                    "entry_date"
                ].min(),
                "training_end": training[
                    "entry_date"
                ].max(),
                "test_start": testing["entry_date"].min(),
                "test_end": testing["entry_date"].max(),
                **{
                    f"train_{key}": value
                    for key, value in training_summary.items()
                },
                **{
                    f"test_{key}": value
                    for key, value in test_summary.items()
                },
                "train_normal_posterior_probability": (
                    train_normal_probability
                ),
                "train_elevated_posterior_probability": (
                    train_elevated_probability
                ),
                "train_baseline_posterior_probability": (
                    train_baseline_probability
                ),
                "test_regime_brier_score": regime_brier,
                "test_baseline_brier_score": baseline_brier,
                "test_brier_skill_vs_baseline": brier_skill,
                "test_regime_log_loss": regime_log_loss,
                "test_baseline_log_loss": baseline_log_loss,
                "test_log_loss_improvement": (
                    baseline_log_loss - regime_log_loss
                ),
                "test_informative": bool(
                    test_summary["failures"] > 0
                ),
            }

            fold_rows.append(fold_row)

            prediction_frames.append(
                testing[
                    [
                        "entry_date",
                        "expiration_proxy_date",
                        "fold_number",
                        "fold_label",
                        "candidate_score",
                        "elevated_regime",
                        "expiration_failure",
                        "regime_predicted_probability",
                        "baseline_predicted_probability",
                    ]
                ]
            )

            print(
                f"{fold_label} | "
                f"Train through {training['entry_date'].max().date()} | "
                f"Test N={test_summary['observations']:>3}, "
                f"Failures={test_summary['failures']:>2} | "
                f"Normal={format_rate(test_summary['normal_failure_rate']):>7} | "
                f"Elevated={format_rate(test_summary['elevated_failure_rate']):>7} | "
                f"Lift={format_lift(test_summary['elevated_rate_lift_vs_normal']):>6} | "
                f"Captured={format_rate(test_summary['elevated_failure_capture']):>7} | "
                f"Brier skill={brier_skill:>7.2%}"
            )

        folds = pd.DataFrame(fold_rows)

        out_of_fold = pd.concat(
            prediction_frames,
            ignore_index=True,
        ).sort_values("entry_date")

        if out_of_fold["entry_date"].duplicated().any():
            raise ValueError(
                "A test observation appeared in more than one fold."
            )

        pooled_summary = summarize_regimes(out_of_fold)

        pooled_actual = (
            out_of_fold["expiration_failure"]
            .astype(int)
            .to_numpy()
        )

        pooled_regime_predictions = out_of_fold[
            "regime_predicted_probability"
        ].to_numpy()

        pooled_baseline_predictions = out_of_fold[
            "baseline_predicted_probability"
        ].to_numpy()

        pooled_regime_brier = brier_score(
            pooled_actual,
            pooled_regime_predictions,
        )

        pooled_baseline_brier = brier_score(
            pooled_actual,
            pooled_baseline_predictions,
        )

        pooled_brier_skill = (
            1.0 - pooled_regime_brier / pooled_baseline_brier
            if pooled_baseline_brier > 0
            else np.nan
        )

        pooled_regime_log_loss = binary_log_loss(
            pooled_actual,
            pooled_regime_predictions,
        )

        pooled_baseline_log_loss = binary_log_loss(
            pooled_actual,
            pooled_baseline_predictions,
        )

        yearly_rows: list[dict[str, Any]] = []

        for calendar_year, group in out_of_fold.groupby(
            out_of_fold["entry_date"].dt.year
        ):
            year_summary = summarize_regimes(group)

            yearly_rows.append(
                {
                    "calendar_year": int(calendar_year),
                    **year_summary,
                }
            )

        yearly = pd.DataFrame(yearly_rows)

        informative_folds = folds.loc[
            folds["test_informative"]
        ]

        supportive_folds = int(
            folds["test_direction_supportive"].sum()
        )

        supportive_informative_folds = int(
            informative_folds[
                "test_direction_supportive"
            ].sum()
        )

        positive_brier_folds = int(
            (folds["test_brier_skill_vs_baseline"] > 0).sum()
        )

        training_supportive_folds = int(
            folds["train_direction_supportive"].sum()
        )

        if (
            pooled_summary["elevated_rate_lift_vs_normal"] >= 2.0
            and pooled_summary["fisher_p_value"] <= 0.05
            and supportive_informative_folds
            >= max(1, len(informative_folds) - 1)
            and pooled_brier_skill > 0
        ):
            validation_assessment = (
                "Strong chronological support"
            )
        elif (
            pooled_summary["elevated_rate_lift_vs_normal"] >= 1.5
            and supportive_informative_folds
            >= math.ceil(len(informative_folds) / 2)
        ):
            validation_assessment = (
                "Moderate chronological support"
            )
        else:
            validation_assessment = (
                "Insufficient chronological support"
            )

        RESULTS_DIR.mkdir(
            parents=True,
            exist_ok=True,
        )

        folds.to_csv(
            FOLD_OUTPUT,
            index=False,
        )

        yearly.to_csv(
            YEARLY_OUTPUT,
            index=False,
        )

        connection.register(
            "walk_forward_folds_dataframe",
            folds,
        )

        connection.register(
            "walk_forward_yearly_dataframe",
            yearly,
        )

        connection.execute(
            """
            CREATE OR REPLACE TABLE
                pcs.regime_walk_forward_folds AS
            SELECT *
            FROM walk_forward_folds_dataframe
            ORDER BY fold_number
            """
        )

        connection.execute(
            """
            CREATE OR REPLACE TABLE
                pcs.regime_walk_forward_yearly AS
            SELECT *
            FROM walk_forward_yearly_dataframe
            ORDER BY calendar_year
            """
        )

        summary_lines = [
            (
                "MNQ PCS REGIME RESEARCH — "
                "FORWARD-CHAINING VALIDATION"
            ),
            "=" * 62,
            "",
            "Methodological status",
            "-" * 62,
            (
                "This is a chronological stability and "
                "pseudo-out-of-sample diagnostic."
            ),
            (
                "The score was discovered using the full history, "
                "so the historical test folds are not a pristine "
                "untouched holdout."
            ),
            (
                "Within each fold, failure probabilities are "
                "estimated only from earlier observations."
            ),
            "",
            "Fixed definition",
            "-" * 62,
            f"Factors: {len(factors)}",
            f"Elevated cutoff: score >= {elevated_minimum}",
            f"Test folds: {len(folds)}",
            "",
            "Pooled non-overlapping test observations",
            "-" * 62,
            (
                f"Observations: "
                f"{pooled_summary['observations']:,}"
            ),
            (
                f"Failures: {pooled_summary['failures']} "
                f"({pooled_summary['overall_failure_rate']:.2%})"
            ),
            (
                f"Normal: N={pooled_summary['normal_observations']}, "
                f"failures={pooled_summary['normal_failures']} "
                f"({pooled_summary['normal_failure_rate']:.2%})"
            ),
            (
                f"Elevated: "
                f"N={pooled_summary['elevated_observations']}, "
                f"failures={pooled_summary['elevated_failures']} "
                f"({pooled_summary['elevated_failure_rate']:.2%})"
            ),
            (
                f"Elevated lift versus normal: "
                f"{format_lift(pooled_summary['elevated_rate_lift_vs_normal'])}"
            ),
            (
                f"Elevated failure capture: "
                f"{pooled_summary['elevated_failure_capture']:.1%}"
            ),
            (
                f"Fisher p-value: "
                f"{pooled_summary['fisher_p_value']:.4f}"
            ),
            "",
            "Forward probability scoring",
            "-" * 62,
            (
                f"Regime Brier score: "
                f"{pooled_regime_brier:.6f}"
            ),
            (
                f"Baseline Brier score: "
                f"{pooled_baseline_brier:.6f}"
            ),
            (
                f"Brier skill versus baseline: "
                f"{pooled_brier_skill:.2%}"
            ),
            (
                f"Regime log loss: "
                f"{pooled_regime_log_loss:.6f}"
            ),
            (
                f"Baseline log loss: "
                f"{pooled_baseline_log_loss:.6f}"
            ),
            (
                f"Log-loss improvement: "
                f"{pooled_baseline_log_loss - pooled_regime_log_loss:.6f}"
            ),
            "",
            "Fold consistency",
            "-" * 62,
            (
                f"Training folds with elevated risk above normal: "
                f"{training_supportive_folds}/{len(folds)}"
            ),
            (
                f"Test folds with elevated risk above normal: "
                f"{supportive_folds}/{len(folds)}"
            ),
            (
                f"Informative test folds: "
                f"{len(informative_folds)}/{len(folds)}"
            ),
            (
                f"Supportive informative test folds: "
                f"{supportive_informative_folds}/"
                f"{len(informative_folds)}"
            ),
            (
                f"Test folds with positive Brier skill: "
                f"{positive_brier_folds}/{len(folds)}"
            ),
            "",
            f"Assessment: {validation_assessment}",
        ]

        SUMMARY_OUTPUT.write_text(
            "\n".join(summary_lines) + "\n",
            encoding="utf-8",
        )

        print("\nPooled non-overlapping test results")
        print("-" * 94)

        print(
            f"Test observations: "
            f"{pooled_summary['observations']:,} | "
            f"Failures: {pooled_summary['failures']} "
            f"({pooled_summary['overall_failure_rate']:.2%})"
        )

        print(
            f"Normal: N={pooled_summary['normal_observations']} | "
            f"Failures={pooled_summary['normal_failures']} "
            f"({pooled_summary['normal_failure_rate']:.2%})"
        )

        print(
            f"Elevated: "
            f"N={pooled_summary['elevated_observations']} | "
            f"Failures={pooled_summary['elevated_failures']} "
            f"({pooled_summary['elevated_failure_rate']:.2%}) | "
            f"Lift={format_lift(pooled_summary['elevated_rate_lift_vs_normal'])} | "
            f"Captured={pooled_summary['elevated_failure_capture']:.1%} | "
            f"Fisher p={pooled_summary['fisher_p_value']:.4f}"
        )

        print("\nForward probability scoring")
        print("-" * 94)

        print(
            f"Regime Brier={pooled_regime_brier:.6f} | "
            f"Baseline Brier={pooled_baseline_brier:.6f} | "
            f"Skill={pooled_brier_skill:.2%}"
        )

        print(
            f"Regime log loss={pooled_regime_log_loss:.6f} | "
            f"Baseline log loss={pooled_baseline_log_loss:.6f} | "
            f"Improvement="
            f"{pooled_baseline_log_loss - pooled_regime_log_loss:.6f}"
        )

        print("\nFold consistency")
        print("-" * 94)

        print(
            f"Training supportive: "
            f"{training_supportive_folds}/{len(folds)} | "
            f"Test supportive: {supportive_folds}/{len(folds)} | "
            f"Supportive informative folds: "
            f"{supportive_informative_folds}/"
            f"{len(informative_folds)} | "
            f"Positive Brier skill: "
            f"{positive_brier_folds}/{len(folds)}"
        )

        print(
            f"\nAssessment: {validation_assessment}"
        )

        print(f"\nFold output: {FOLD_OUTPUT}")
        print(f"Yearly output: {YEARLY_OUTPUT}")
        print(f"Summary output: {SUMMARY_OUTPUT}")
        print(
            "DuckDB tables: pcs.regime_walk_forward_folds, "
            "pcs.regime_walk_forward_yearly"
        )

        print(
            "Forward-chaining validation completed successfully."
        )

        return 0

    except Exception as exc:
        print(f"\nERROR: {exc}")
        return 1

    finally:
        connection.close()


if __name__ == "__main__":
    sys.exit(main())

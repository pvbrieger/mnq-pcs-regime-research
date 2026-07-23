"""Analyze defensive short-strike scenarios for elevated PCS regimes.

This is a fixed-credit expiration proxy, not a historical options backtest.
For elevated-regime entries only, the script moves the short strike farther
below Friday spot by a configured percentage-point distance and moves the long
strike by the same amount, preserving the 150-point spread width.

The key output is the maximum credit haircut the defensive strike could absorb
before its total P&L falls below the standard-strike policy.

Inputs
------
- database/research.duckdb
- config/candidate_regime_score.yaml
- config/policy_proxy.yaml
- config/defensive_strike_scenarios.yaml
- pcs.friday_regime_features

Outputs
-------
- results/defensive_strikes/defensive_strike_results.csv
- results/defensive_strikes/defensive_strike_summary.txt
- DuckDB tables:
    pcs.defensive_strike_results
    pcs.defensive_strike_trade_scenarios
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
DEFENSIVE_CONFIG_PATH = (
    PROJECT_ROOT / "config" / "defensive_strike_scenarios.yaml"
)

RESULTS_DIR = PROJECT_ROOT / "results" / "defensive_strikes"
RESULTS_OUTPUT = RESULTS_DIR / "defensive_strike_results.csv"
SUMMARY_OUTPUT = RESULTS_DIR / "defensive_strike_summary.txt"


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


def first_existing_column(
    dataframe: pd.DataFrame,
    candidates: tuple[str, ...],
) -> str | None:
    """Return the first available candidate column."""

    for column in candidates:
        if column in dataframe.columns:
            return column

    return None


def resolve_entry_spot(
    dataframe: pd.DataFrame,
) -> tuple[pd.Series, str]:
    """Resolve the Friday entry-price proxy."""

    column = first_existing_column(
        dataframe,
        (
            "weekly_close",
            "entry_close",
            "entry_spot",
            "entry_price",
            "ndx_entry_close",
        ),
    )

    if column is None:
        raise ValueError(
            "Unable to resolve Friday entry spot. Available columns: "
            + ", ".join(sorted(dataframe.columns))
        )

    values = pd.to_numeric(
        dataframe[column],
        errors="coerce",
    ).astype(float)

    return values, column


def resolve_short_strike(
    dataframe: pd.DataFrame,
) -> tuple[pd.Series, str]:
    """Resolve the standard proxy short strike."""

    column = first_existing_column(
        dataframe,
        (
            "short_strike_proxy",
            "proxy_short_strike",
            "short_strike",
            "short_strike_level",
            "short_strike_price",
        ),
    )

    if column is None:
        raise ValueError(
            "Unable to resolve the proxy short strike. "
            "Available columns: "
            + ", ".join(sorted(dataframe.columns))
        )

    values = pd.to_numeric(
        dataframe[column],
        errors="coerce",
    ).astype(float)

    return values, column


def resolve_expiration_settlement(
    dataframe: pd.DataFrame,
) -> tuple[pd.Series, str]:
    """Resolve the expiration settlement proxy."""

    column = first_existing_column(
        dataframe,
        (
            "expiration_close",
            "expiration_proxy_close",
            "expiration_settlement_proxy",
            "expiration_settlement",
            "expiration_ndx_close",
            "expiration_price",
        ),
    )

    if column is None:
        raise ValueError(
            "Unable to resolve expiration settlement. "
            "Available columns: "
            + ", ".join(sorted(dataframe.columns))
        )

    values = pd.to_numeric(
        dataframe[column],
        errors="coerce",
    ).astype(float)

    return values, column


def make_model_mask(
    dataframe: pd.DataFrame,
    model: dict[str, Any],
    factor_columns: dict[str, str],
) -> pd.Series:
    """Build the elevated-regime mask for one model."""

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
                    f"Model '{model['id']}' references "
                    f"missing factor '{factor_id}'."
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
                    f"Model '{model['id']}' references "
                    f"missing factor '{factor_id}'."
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
    """Calculate capped put-spread intrinsic value in points."""

    return pd.Series(
        np.clip(
            short_strike.to_numpy(dtype=float)
            - settlement.to_numpy(dtype=float),
            0.0,
            spread_width,
        ),
        index=short_strike.index,
        dtype=float,
    )


def maximum_drawdown(
    pnl_values: pd.Series,
) -> float:
    """Return maximum cumulative-P&L drawdown as a positive number."""

    values = pd.to_numeric(
        pnl_values,
        errors="coerce",
    ).fillna(0.0).to_numpy(dtype=float)

    equity_curve = np.concatenate(
        ([0.0], np.cumsum(values))
    )

    running_peak = np.maximum.accumulate(
        equity_curve
    )

    return float(
        (running_peak - equity_curve).max()
    )


def worst_consecutive_sum(
    pnl_values: pd.Series,
    window: int,
) -> float:
    """Return the worst sum across consecutive trades."""

    values = pd.to_numeric(
        pnl_values,
        errors="coerce",
    ).fillna(0.0)

    if len(values) < window:
        return float(values.sum())

    return float(
        values.rolling(window).sum().min()
    )


def main() -> int:
    """Run defensive-strike scenario analysis."""

    print(
        "MNQ PCS Regime Research — Defensive Strike Analysis"
    )
    print("=" * 55)

    required_paths = (
        DATABASE_PATH,
        SCORE_CONFIG_PATH,
        POLICY_CONFIG_PATH,
        DEFENSIVE_CONFIG_PATH,
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

    with DEFENSIVE_CONFIG_PATH.open(
        "r",
        encoding="utf-8",
    ) as file_handle:
        defensive_config = yaml.safe_load(file_handle)

    factors = score_config.get("factors", [])
    models = policy_config.get("models", [])
    samples = policy_config.get("samples", [])

    spread_width = float(
        defensive_config["spread_width_points"]
    )

    contract_multiplier = float(
        defensive_config[
            "contract_multiplier_dollars_per_point"
        ]
    )

    standard_credits = [
        float(value)
        for value in defensive_config[
            "standard_credit_scenarios_points"
        ]
    ]

    extra_distances = [
        float(value)
        for value in defensive_config[
            "additional_distance_fractions"
        ]
    ]

    rolling_window = int(
        defensive_config.get(
            "consecutive_trade_window",
            4,
        )
    )

    if spread_width <= 0:
        print("ERROR: Spread width must be positive.")
        return 1

    if contract_multiplier <= 0:
        print("ERROR: Contract multiplier must be positive.")
        return 1

    for credit in standard_credits:
        if not 0 < credit < spread_width:
            print(
                f"ERROR: Standard credit {credit} must be "
                "between zero and spread width."
            )
            return 1

    for distance in extra_distances:
        if not 0 < distance < 0.10:
            print(
                f"ERROR: Additional distance {distance} "
                "must be between 0 and 10%."
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

        entry_spot, entry_spot_source = (
            resolve_entry_spot(dataframe)
        )

        standard_short, short_strike_source = (
            resolve_short_strike(dataframe)
        )

        settlement, settlement_source = (
            resolve_expiration_settlement(dataframe)
        )

        dataframe["entry_spot_proxy"] = entry_spot
        dataframe["standard_short_strike"] = standard_short
        dataframe["expiration_settlement_proxy"] = settlement

        missing_values = dataframe[
            [
                "entry_spot_proxy",
                "standard_short_strike",
                "expiration_settlement_proxy",
            ]
        ].isna().any(axis=1)

        if missing_values.any():
            raise ValueError(
                "Required market values are missing on "
                f"{int(missing_values.sum())} rows."
            )

        dataframe["standard_intrinsic_points"] = (
            spread_intrinsic(
                dataframe["standard_short_strike"],
                dataframe["expiration_settlement_proxy"],
                spread_width,
            )
        )

        derived_failure = (
            dataframe["expiration_settlement_proxy"]
            < dataframe["standard_short_strike"]
        )

        recorded_failure = (
            dataframe["expiration_failure"]
            .fillna(False)
            .astype(bool)
        )

        mismatch_count = int(
            (derived_failure != recorded_failure).sum()
        )

        if mismatch_count:
            raise ValueError(
                "Resolved standard strike and settlement disagree "
                f"with expiration_failure on {mismatch_count} rows."
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

            dataframe[f"model_{model_id}"] = (
                make_model_mask(
                    dataframe,
                    model,
                    factor_columns,
                )
            )

        print(
            f"Complete history: {len(dataframe):,} observations | "
            f"{int(recorded_failure.sum())} standard-strike failures"
        )

        print(
            f"Entry spot source: {entry_spot_source}"
        )

        print(
            f"Short strike source: {short_strike_source}"
        )

        print(
            f"Expiration settlement source: "
            f"{settlement_source}"
        )

        print(
            "Additional distances: "
            + ", ".join(
                f"{distance:.1%}"
                for distance in extra_distances
            )
        )

        result_rows: list[dict[str, Any]] = []
        trade_frames: list[pd.DataFrame] = []

        for sample in samples:
            sample_id = str(sample["id"])
            sample_label = str(sample["label"])

            sample_data = dataframe.loc[
                dataframe["entry_date"].between(
                    pd.Timestamp(sample["start_date"]),
                    pd.Timestamp(sample["end_date"]),
                )
            ].copy()

            if sample_data.empty:
                raise ValueError(
                    f"Sample '{sample_id}' contains no rows."
                )

            sample_data = sample_data.sort_values(
                [
                    "expiration_proxy_date",
                    "entry_date",
                ]
            )

            for model in models:
                model_id = str(model["id"])
                model_label = str(model["label"])
                elevated_column = f"model_{model_id}"

                elevated_mask = sample_data[
                    elevated_column
                ].astype(bool)

                elevated_count = int(
                    elevated_mask.sum()
                )

                if elevated_count == 0:
                    raise ValueError(
                        f"Model '{model_id}' has no elevated "
                        f"observations in sample '{sample_id}'."
                    )

                standard_elevated_intrinsic = (
                    sample_data.loc[
                        elevated_mask,
                        "standard_intrinsic_points",
                    ]
                )

                standard_elevated_failures = int(
                    (
                        sample_data.loc[
                            elevated_mask,
                            "expiration_settlement_proxy",
                        ]
                        < sample_data.loc[
                            elevated_mask,
                            "standard_short_strike",
                        ]
                    ).sum()
                )

                for extra_distance in extra_distances:
                    defensive_short = (
                        sample_data["standard_short_strike"]
                        - (
                            sample_data["entry_spot_proxy"]
                            * extra_distance
                        )
                    )

                    defensive_intrinsic = spread_intrinsic(
                        defensive_short,
                        sample_data[
                            "expiration_settlement_proxy"
                        ],
                        spread_width,
                    )

                    defensive_failures_mask = (
                        sample_data[
                            "expiration_settlement_proxy"
                        ]
                        < defensive_short
                    )

                    defensive_elevated_failures = int(
                        defensive_failures_mask.loc[
                            elevated_mask
                        ].sum()
                    )

                    failures_avoided = (
                        standard_elevated_failures
                        - defensive_elevated_failures
                    )

                    standard_intrinsic_total = float(
                        standard_elevated_intrinsic.sum()
                    )

                    defensive_intrinsic_total = float(
                        defensive_intrinsic.loc[
                            elevated_mask
                        ].sum()
                    )

                    intrinsic_avoided = (
                        standard_intrinsic_total
                        - defensive_intrinsic_total
                    )

                    max_credit_haircut = safe_divide(
                        intrinsic_avoided,
                        elevated_count,
                    )

                    standard_elevated_break_even = (
                        safe_divide(
                            standard_intrinsic_total,
                            elevated_count,
                        )
                    )

                    defensive_elevated_break_even = (
                        safe_divide(
                            defensive_intrinsic_total,
                            elevated_count,
                        )
                    )

                    for standard_credit in standard_credits:
                        theoretical_match_credit = (
                            standard_credit
                            - max_credit_haircut
                        )

                        feasible_match_credit = max(
                            0.0,
                            theoretical_match_credit,
                        )

                        retained_credit_ratio = (
                            safe_divide(
                                feasible_match_credit,
                                standard_credit,
                            )
                        )

                        standard_pnl = (
                            standard_credit
                            - sample_data[
                                "standard_intrinsic_points"
                            ]
                        )

                        same_credit_defensive_pnl = (
                            standard_credit
                            - defensive_intrinsic
                        )

                        portfolio_same_credit_pnl = (
                            standard_pnl.copy()
                        )

                        portfolio_same_credit_pnl.loc[
                            elevated_mask
                        ] = same_credit_defensive_pnl.loc[
                            elevated_mask
                        ]

                        match_credit_defensive_pnl = (
                            feasible_match_credit
                            - defensive_intrinsic
                        )

                        portfolio_match_credit_pnl = (
                            standard_pnl.copy()
                        )

                        portfolio_match_credit_pnl.loc[
                            elevated_mask
                        ] = match_credit_defensive_pnl.loc[
                            elevated_mask
                        ]

                        standard_total_points = float(
                            standard_pnl.sum()
                        )

                        same_credit_total_points = float(
                            portfolio_same_credit_pnl.sum()
                        )

                        match_credit_total_points = float(
                            portfolio_match_credit_pnl.sum()
                        )

                        standard_drawdown = maximum_drawdown(
                            standard_pnl
                        )

                        same_credit_drawdown = maximum_drawdown(
                            portfolio_same_credit_pnl
                        )

                        match_credit_drawdown = maximum_drawdown(
                            portfolio_match_credit_pnl
                        )

                        standard_worst_window = (
                            worst_consecutive_sum(
                                standard_pnl,
                                rolling_window,
                            )
                        )

                        same_credit_worst_window = (
                            worst_consecutive_sum(
                                portfolio_same_credit_pnl,
                                rolling_window,
                            )
                        )

                        match_credit_worst_window = (
                            worst_consecutive_sum(
                                portfolio_match_credit_pnl,
                                rolling_window,
                            )
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
                                "scheduled_entries": len(
                                    sample_data
                                ),
                                "elevated_entries": elevated_count,
                                "elevated_entry_share": (
                                    safe_divide(
                                        elevated_count,
                                        len(sample_data),
                                    )
                                ),
                                "spread_width_points": (
                                    spread_width
                                ),
                                "contract_multiplier": (
                                    contract_multiplier
                                ),
                                "standard_credit_points": (
                                    standard_credit
                                ),
                                "additional_distance_fraction": (
                                    extra_distance
                                ),
                                "additional_distance_percent": (
                                    extra_distance * 100.0
                                ),
                                "average_additional_distance_points": float(
                                    (
                                        sample_data.loc[
                                            elevated_mask,
                                            "entry_spot_proxy",
                                        ]
                                        * extra_distance
                                    ).mean()
                                ),
                                "standard_elevated_failures": (
                                    standard_elevated_failures
                                ),
                                "defensive_elevated_failures": (
                                    defensive_elevated_failures
                                ),
                                "failures_avoided": (
                                    failures_avoided
                                ),
                                "failure_reduction": (
                                    safe_divide(
                                        failures_avoided,
                                        standard_elevated_failures,
                                    )
                                ),
                                "standard_elevated_intrinsic_points": (
                                    standard_intrinsic_total
                                ),
                                "defensive_elevated_intrinsic_points": (
                                    defensive_intrinsic_total
                                ),
                                "intrinsic_points_avoided": (
                                    intrinsic_avoided
                                ),
                                "intrinsic_dollars_avoided": (
                                    intrinsic_avoided
                                    * contract_multiplier
                                ),
                                "standard_elevated_break_even_credit": (
                                    standard_elevated_break_even
                                ),
                                "defensive_elevated_break_even_credit": (
                                    defensive_elevated_break_even
                                ),
                                "maximum_tolerable_credit_haircut_points": (
                                    max_credit_haircut
                                ),
                                "theoretical_defensive_credit_to_match_standard": (
                                    theoretical_match_credit
                                ),
                                "feasible_defensive_credit_to_match_standard": (
                                    feasible_match_credit
                                ),
                                "minimum_retained_credit_ratio_to_match_standard": (
                                    retained_credit_ratio
                                ),
                                "standard_total_pnl_points": (
                                    standard_total_points
                                ),
                                "standard_total_pnl_dollars": (
                                    standard_total_points
                                    * contract_multiplier
                                ),
                                "same_credit_defensive_total_pnl_points": (
                                    same_credit_total_points
                                ),
                                "same_credit_defensive_total_pnl_dollars": (
                                    same_credit_total_points
                                    * contract_multiplier
                                ),
                                "same_credit_pnl_improvement_points": (
                                    same_credit_total_points
                                    - standard_total_points
                                ),
                                "same_credit_pnl_improvement_dollars": (
                                    (
                                        same_credit_total_points
                                        - standard_total_points
                                    )
                                    * contract_multiplier
                                ),
                                "match_credit_total_pnl_points": (
                                    match_credit_total_points
                                ),
                                "match_credit_total_pnl_dollars": (
                                    match_credit_total_points
                                    * contract_multiplier
                                ),
                                "match_credit_pnl_difference_points": (
                                    match_credit_total_points
                                    - standard_total_points
                                ),
                                "standard_maximum_drawdown_points": (
                                    standard_drawdown
                                ),
                                "same_credit_maximum_drawdown_points": (
                                    same_credit_drawdown
                                ),
                                "same_credit_drawdown_reduction_points": (
                                    standard_drawdown
                                    - same_credit_drawdown
                                ),
                                "match_credit_maximum_drawdown_points": (
                                    match_credit_drawdown
                                ),
                                "match_credit_drawdown_reduction_points": (
                                    standard_drawdown
                                    - match_credit_drawdown
                                ),
                                f"standard_worst_{rolling_window}_trade_sum_points": (
                                    standard_worst_window
                                ),
                                f"same_credit_worst_{rolling_window}_trade_sum_points": (
                                    same_credit_worst_window
                                ),
                                f"match_credit_worst_{rolling_window}_trade_sum_points": (
                                    match_credit_worst_window
                                ),
                            }
                        )

                        trade_frames.append(
                            pd.DataFrame(
                                {
                                    "sample_id": sample_id,
                                    "model_id": model_id,
                                    "standard_credit_points": (
                                        standard_credit
                                    ),
                                    "additional_distance_fraction": (
                                        extra_distance
                                    ),
                                    "entry_date": sample_data[
                                        "entry_date"
                                    ].to_numpy(),
                                    "expiration_proxy_date": (
                                        sample_data[
                                            "expiration_proxy_date"
                                        ].to_numpy()
                                    ),
                                    "elevated_regime": (
                                        elevated_mask.to_numpy()
                                    ),
                                    "entry_spot": sample_data[
                                        "entry_spot_proxy"
                                    ].to_numpy(),
                                    "standard_short_strike": (
                                        sample_data[
                                            "standard_short_strike"
                                        ].to_numpy()
                                    ),
                                    "defensive_short_strike": (
                                        defensive_short.to_numpy()
                                    ),
                                    "expiration_settlement": (
                                        sample_data[
                                            "expiration_settlement_proxy"
                                        ].to_numpy()
                                    ),
                                    "standard_intrinsic_points": (
                                        sample_data[
                                            "standard_intrinsic_points"
                                        ].to_numpy()
                                    ),
                                    "defensive_intrinsic_points": (
                                        defensive_intrinsic.to_numpy()
                                    ),
                                    "standard_pnl_points": (
                                        standard_pnl.to_numpy()
                                    ),
                                    "same_credit_portfolio_pnl_points": (
                                        portfolio_same_credit_pnl.to_numpy()
                                    ),
                                    "match_credit_portfolio_pnl_points": (
                                        portfolio_match_credit_pnl.to_numpy()
                                    ),
                                    "defensive_credit_to_match_standard": (
                                        feasible_match_credit
                                    ),
                                }
                            )
                        )

        results = pd.DataFrame(result_rows)

        trades = pd.concat(
            trade_frames,
            ignore_index=True,
        )

        results = results.sort_values(
            [
                "sample_id",
                "standard_credit_points",
                "model_id",
                "additional_distance_fraction",
            ]
        ).reset_index(drop=True)

        RESULTS_DIR.mkdir(
            parents=True,
            exist_ok=True,
        )

        results.to_csv(
            RESULTS_OUTPUT,
            index=False,
        )

        connection.register(
            "defensive_strike_results_dataframe",
            results,
        )

        connection.register(
            "defensive_strike_trades_dataframe",
            trades,
        )

        connection.execute(
            """
            CREATE OR REPLACE TABLE
                pcs.defensive_strike_results AS
            SELECT *
            FROM defensive_strike_results_dataframe
            ORDER BY
                sample_id,
                standard_credit_points,
                model_id,
                additional_distance_fraction
            """
        )

        connection.execute(
            """
            CREATE OR REPLACE TABLE
                pcs.defensive_strike_trade_scenarios AS
            SELECT *
            FROM defensive_strike_trades_dataframe
            ORDER BY
                sample_id,
                model_id,
                standard_credit_points,
                additional_distance_fraction,
                expiration_proxy_date,
                entry_date
            """
        )

        anchor_credit = min(standard_credits)

        forward = results.loc[
            (
                results["sample_id"] == "forward_test"
            )
            & (
                results["standard_credit_points"]
                == anchor_credit
            )
        ].sort_values(
            [
                "model_id",
                "additional_distance_fraction",
            ]
        )

        print(
            f"\nForward-test defensive strikes at "
            f"{anchor_credit:.2f}-point standard credit"
        )
        print("-" * 184)

        for row in forward.itertuples(index=False):
            print(
                f"{row.model_label:<39} | "
                f"Extra={row.additional_distance_percent:>4.1f}% "
                f"(~{row.average_additional_distance_points:>6.0f} pts) | "
                f"Failures {row.standard_elevated_failures:>2}"
                f"->{row.defensive_elevated_failures:<2} | "
                f"Intrinsic saved=${row.intrinsic_dollars_avoided:>7,.0f} | "
                f"Max haircut={row.maximum_tolerable_credit_haircut_points:>5.2f} pts | "
                f"Match credit={row.feasible_defensive_credit_to_match_standard:>5.2f} "
                f"({row.minimum_retained_credit_ratio_to_match_standard:>5.1%}) | "
                f"Match-credit DD change="
                f"${row.match_credit_drawdown_reduction_points * contract_multiplier:>7,.0f}"
            )

        print("\nForward-test elevated-trade break-even credits")
        print("-" * 130)

        for row in forward.itertuples(index=False):
            print(
                f"{row.model_label:<39} | "
                f"Extra={row.additional_distance_percent:>4.1f}% | "
                f"Standard BE={row.standard_elevated_break_even_credit:>5.2f} | "
                f"Defensive BE={row.defensive_elevated_break_even_credit:>5.2f} | "
                f"Reduction={row.maximum_tolerable_credit_haircut_points:>5.2f} points"
            )

        summary_lines = [
            "MNQ PCS REGIME RESEARCH — DEFENSIVE STRIKE ANALYSIS",
            "=" * 59,
            "",
            "Methodological status",
            "-" * 59,
            (
                "Fixed-credit expiration proxy; not a historical "
                "options backtest."
            ),
            (
                "Defensive strikes are applied only to elevated-regime "
                "entries and preserve the 150-point spread width."
            ),
            (
                "The maximum tolerable credit haircut is the average "
                "intrinsic-value reduction per elevated entry."
            ),
            f"Entry spot source: {entry_spot_source}",
            f"Short strike source: {short_strike_source}",
            f"Settlement source: {settlement_source}",
            "",
            (
                f"Forward-test results at "
                f"{anchor_credit:.2f}-point standard credit"
            ),
            "-" * 59,
        ]

        for row in forward.itertuples(index=False):
            summary_lines.append(
                f"{row.model_label} | "
                f"Extra distance={row.additional_distance_percent:.1f}% | "
                f"Failures={row.standard_elevated_failures}"
                f"->{row.defensive_elevated_failures} | "
                f"Max haircut="
                f"{row.maximum_tolerable_credit_haircut_points:.2f} | "
                f"Match credit="
                f"{row.feasible_defensive_credit_to_match_standard:.2f} | "
                f"Retained credit="
                f"{row.minimum_retained_credit_ratio_to_match_standard:.1%} | "
                f"Match-credit drawdown change="
                f"${row.match_credit_drawdown_reduction_points * contract_multiplier:,.0f}"
            )

        SUMMARY_OUTPUT.write_text(
            "\n".join(summary_lines) + "\n",
            encoding="utf-8",
        )

        print("\nMethodological status")
        print("-" * 92)

        print(
            "Fixed-credit expiration proxy only; actual defensive "
            "strike premiums are not available."
        )

        print(f"\nResults output: {RESULTS_OUTPUT}")
        print(f"Summary output: {SUMMARY_OUTPUT}")
        print(
            "DuckDB tables: pcs.defensive_strike_results, "
            "pcs.defensive_strike_trade_scenarios"
        )

        print(
            "Defensive strike analysis completed successfully."
        )

        return 0

    except Exception as exc:
        print(f"\nERROR: {exc}")
        return 1

    finally:
        connection.close()


if __name__ == "__main__":
    sys.exit(main())

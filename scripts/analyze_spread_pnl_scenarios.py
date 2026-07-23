"""Run normalized put-spread P&L scenario analysis.

This is a proxy analysis, not a historical options backtest. It applies fixed
credit assumptions to the existing NDX-based proxy short strikes and four-week
expiration settlements.

Limitations include:
- NDX proxy rather than actual MNQ futures settlement
- four-week proxy expiration rather than exact listed 26–35 DTE contract
- no historical option quotes, fills, commissions, or slippage
- no listed-strike rounding
- fractional policy multipliers represent expected exposure, not contract
  rounding in a live account
- cumulative P&L is per one-contract baseline and does not model compounding

Inputs
------
- database/research.duckdb
- config/candidate_regime_score.yaml
- config/policy_proxy.yaml
- config/spread_pnl_scenarios.yaml
- pcs.friday_regime_features

Outputs
-------
- results/spread_pnl_scenarios/spread_pnl_policy_results.csv
- results/spread_pnl_scenarios/spread_pnl_summary.txt
- DuckDB tables:
    pcs.spread_pnl_policy_results
    pcs.spread_pnl_trade_scenarios
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
PNL_CONFIG_PATH = (
    PROJECT_ROOT / "config" / "spread_pnl_scenarios.yaml"
)

RESULTS_DIR = PROJECT_ROOT / "results" / "spread_pnl_scenarios"
RESULTS_OUTPUT = RESULTS_DIR / "spread_pnl_policy_results.csv"
SUMMARY_OUTPUT = RESULTS_DIR / "spread_pnl_summary.txt"


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
    """Return the first candidate column present in the dataframe."""

    for column in candidates:
        if column in dataframe.columns:
            return column

    return None


def normalize_fraction(series: pd.Series) -> pd.Series:
    """Normalize decimal or percentage-point values to decimal form."""

    values = pd.to_numeric(
        series,
        errors="coerce",
    ).astype(float)

    median_absolute = float(
        values.abs().dropna().median()
    ) if values.notna().any() else np.nan

    if not pd.isna(median_absolute) and median_absolute > 1.0:
        values = values / 100.0

    return values


def resolve_short_strike(
    dataframe: pd.DataFrame,
) -> tuple[pd.Series, str]:
    """Resolve or derive the proxy short strike."""

    direct_candidates = (
        "short_strike_proxy",
        "proxy_short_strike",
        "short_strike",
        "short_strike_level",
        "short_strike_price",
    )

    direct_column = first_existing_column(
        dataframe,
        direct_candidates,
    )

    if direct_column is not None:
        values = pd.to_numeric(
            dataframe[direct_column],
            errors="coerce",
        ).astype(float)

        return values, direct_column

    entry_column = first_existing_column(
        dataframe,
        (
            "weekly_close",
            "entry_close",
            "entry_spot",
            "entry_price",
            "ndx_entry_close",
        ),
    )

    distance_column = first_existing_column(
        dataframe,
        (
            "short_strike_distance_pct",
            "strike_distance_pct",
            "distance_below_spot_pct",
            "vix_ladder_distance",
            "short_strike_distance",
            "strike_distance",
        ),
    )

    if entry_column is None or distance_column is None:
        raise ValueError(
            "Unable to resolve the short strike. Available columns: "
            + ", ".join(sorted(dataframe.columns))
        )

    entry_price = pd.to_numeric(
        dataframe[entry_column],
        errors="coerce",
    ).astype(float)

    raw_distance = pd.to_numeric(
        dataframe[distance_column],
        errors="coerce",
    ).astype(float)

    median_absolute = float(
        raw_distance.abs().dropna().median()
    ) if raw_distance.notna().any() else np.nan

    if pd.isna(median_absolute):
        raise ValueError(
            f"Distance column '{distance_column}' contains no values."
        )

    if median_absolute <= 0.50:
        distance_fraction = raw_distance
        short_strike = entry_price * (
            1.0 - distance_fraction
        )
        method = (
            f"derived from {entry_column} and decimal "
            f"{distance_column}"
        )
    elif median_absolute <= 50.0:
        distance_fraction = raw_distance / 100.0
        short_strike = entry_price * (
            1.0 - distance_fraction
        )
        method = (
            f"derived from {entry_column} and percentage "
            f"{distance_column}"
        )
    else:
        short_strike = entry_price - raw_distance
        method = (
            f"derived from {entry_column} and point "
            f"{distance_column}"
        )

    return short_strike, method


def resolve_expiration_settlement(
    dataframe: pd.DataFrame,
) -> tuple[pd.Series, str]:
    """Resolve or derive the expiration proxy settlement."""

    direct_candidates = (
        "expiration_proxy_close",
        "expiration_close",
        "expiration_settlement_proxy",
        "expiration_settlement",
        "expiration_ndx_close",
        "expiration_price",
    )

    direct_column = first_existing_column(
        dataframe,
        direct_candidates,
    )

    if direct_column is not None:
        values = pd.to_numeric(
            dataframe[direct_column],
            errors="coerce",
        ).astype(float)

        return values, direct_column

    entry_column = first_existing_column(
        dataframe,
        (
            "weekly_close",
            "entry_close",
            "entry_spot",
            "entry_price",
            "ndx_entry_close",
        ),
    )

    if (
        entry_column is not None
        and "expiration_return" in dataframe.columns
    ):
        entry_price = pd.to_numeric(
            dataframe[entry_column],
            errors="coerce",
        ).astype(float)

        expiration_return = normalize_fraction(
            dataframe["expiration_return"]
        )

        settlement = entry_price * (
            1.0 + expiration_return
        )

        return (
            settlement,
            (
                f"derived from {entry_column} and "
                "expiration_return"
            ),
        )

    raise ValueError(
        "Unable to resolve the expiration settlement. "
        "Available columns: "
        + ", ".join(sorted(dataframe.columns))
    )


def make_model_mask(
    dataframe: pd.DataFrame,
    model: dict[str, Any],
    factor_columns: dict[str, str],
) -> pd.Series:
    """Build the elevated-regime mask for one model."""

    model_type = str(model["type"])

    if model_type == "score_at_least":
        factor_ids = [
            str(factor_id)
            for factor_id in model["factor_ids"]
        ]

        score = pd.Series(
            0,
            index=dataframe.index,
            dtype="int64",
        )

        for factor_id in factor_ids:
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

    drawdowns = running_peak - equity_curve

    return float(drawdowns.max())


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
    """Run fixed-credit expiration P&L scenarios."""

    print(
        "MNQ PCS Regime Research — Spread P&L Scenarios"
    )
    print("=" * 51)

    required_paths = (
        DATABASE_PATH,
        SCORE_CONFIG_PATH,
        POLICY_CONFIG_PATH,
        PNL_CONFIG_PATH,
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

    with PNL_CONFIG_PATH.open(
        "r",
        encoding="utf-8",
    ) as file_handle:
        pnl_config = yaml.safe_load(file_handle)

    factors = score_config.get("factors", [])
    models = policy_config.get("models", [])
    policies = policy_config.get("policies", [])
    samples = policy_config.get("samples", [])

    spread_width = float(
        pnl_config["spread_width_points"]
    )

    contract_multiplier = float(
        pnl_config["contract_multiplier_dollars_per_point"]
    )

    credit_scenarios = [
        float(value)
        for value in pnl_config["credit_scenarios_points"]
    ]

    rolling_window = int(
        pnl_config.get(
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

    for credit in credit_scenarios:
        if not 0 < credit < spread_width:
            print(
                f"ERROR: Credit {credit} must be between "
                "zero and the spread width."
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

        dataframe["expiration_failure"] = (
            dataframe["expiration_failure"]
            .fillna(False)
            .astype(bool)
        )

        short_strike, short_strike_source = (
            resolve_short_strike(dataframe)
        )

        expiration_settlement, settlement_source = (
            resolve_expiration_settlement(dataframe)
        )

        dataframe["scenario_short_strike"] = (
            short_strike
        )

        dataframe["scenario_expiration_settlement"] = (
            expiration_settlement
        )

        invalid_market_rows = (
            dataframe[
                [
                    "scenario_short_strike",
                    "scenario_expiration_settlement",
                ]
            ]
            .isna()
            .any(axis=1)
        )

        if invalid_market_rows.any():
            raise ValueError(
                "Missing short-strike or expiration-settlement "
                f"values on {int(invalid_market_rows.sum())} rows."
            )

        if (
            dataframe["scenario_short_strike"] <= 0
        ).any():
            raise ValueError(
                "Resolved short strikes contain nonpositive values."
            )

        dataframe["scenario_long_strike"] = (
            dataframe["scenario_short_strike"]
            - spread_width
        )

        dataframe["spread_intrinsic_points"] = np.clip(
            (
                dataframe["scenario_short_strike"]
                - dataframe[
                    "scenario_expiration_settlement"
                ]
            ),
            0.0,
            spread_width,
        )

        derived_failure = (
            dataframe[
                "scenario_expiration_settlement"
            ]
            < dataframe["scenario_short_strike"]
        )

        mismatch_count = int(
            (
                derived_failure
                != dataframe["expiration_failure"]
            ).sum()
        )

        if mismatch_count:
            raise ValueError(
                "Resolved strike and settlement disagree with "
                f"expiration_failure on {mismatch_count} rows."
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
            f"{int(dataframe['expiration_failure'].sum())} "
            "strike failures"
        )

        print(
            f"Short strike source: {short_strike_source}"
        )

        print(
            f"Expiration settlement source: "
            f"{settlement_source}"
        )

        print(
            f"Spread width: {spread_width:.2f} points | "
            f"Multiplier: ${contract_multiplier:.2f}/point | "
            f"Credits: {', '.join(f'{value:.2f}' for value in credit_scenarios)}"
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

                elevated_intrinsic = sample_data.loc[
                    elevated_mask,
                    "spread_intrinsic_points",
                ]

                break_even_credit = float(
                    elevated_intrinsic.mean()
                ) if len(elevated_intrinsic) else np.nan

                elevated_observations = int(
                    elevated_mask.sum()
                )

                elevated_failures = int(
                    sample_data.loc[
                        elevated_mask,
                        "expiration_failure",
                    ].sum()
                )

                for credit in credit_scenarios:
                    base_pnl_points = (
                        credit
                        - sample_data[
                            "spread_intrinsic_points"
                        ]
                    )

                    max_risk_points = (
                        spread_width - credit
                    )

                    for policy in policies:
                        policy_id = str(policy["id"])
                        policy_label = str(policy["label"])
                        elevated_multiplier = float(
                            policy["elevated_multiplier"]
                        )

                        if not 0.0 <= elevated_multiplier <= 1.0:
                            raise ValueError(
                                f"Policy '{policy_id}' has an "
                                "invalid elevated multiplier."
                            )

                        position_multiplier = np.where(
                            elevated_mask.to_numpy(),
                            elevated_multiplier,
                            1.0,
                        )

                        weighted_pnl_points = (
                            base_pnl_points.to_numpy()
                            * position_multiplier
                        )

                        weighted_pnl_dollars = (
                            weighted_pnl_points
                            * contract_multiplier
                        )

                        credit_points_received = float(
                            (
                                position_multiplier
                                * credit
                            ).sum()
                        )

                        intrinsic_points_paid = float(
                            (
                                position_multiplier
                                * sample_data[
                                    "spread_intrinsic_points"
                                ].to_numpy()
                            ).sum()
                        )

                        exposure_units = float(
                            position_multiplier.sum()
                        )

                        risk_points_deployed = float(
                            (
                                position_multiplier
                                * max_risk_points
                            ).sum()
                        )

                        total_pnl_points = float(
                            weighted_pnl_points.sum()
                        )

                        total_pnl_dollars = float(
                            weighted_pnl_dollars.sum()
                        )

                        weighted_series_points = pd.Series(
                            weighted_pnl_points,
                            index=sample_data.index,
                        )

                        drawdown_points = maximum_drawdown(
                            weighted_series_points
                        )

                        worst_window_points = (
                            worst_consecutive_sum(
                                weighted_series_points,
                                rolling_window,
                            )
                        )

                        full_size_elevated_pnl = float(
                            base_pnl_points.loc[
                                elevated_mask
                            ].sum()
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
                                "elevated_multiplier": (
                                    elevated_multiplier
                                ),
                                "spread_width_points": (
                                    spread_width
                                ),
                                "contract_multiplier": (
                                    contract_multiplier
                                ),
                                "credit_points": credit,
                                "scheduled_entries": len(
                                    sample_data
                                ),
                                "elevated_entries": (
                                    elevated_observations
                                ),
                                "elevated_entry_share": (
                                    safe_divide(
                                        elevated_observations,
                                        len(sample_data),
                                    )
                                ),
                                "elevated_failures": (
                                    elevated_failures
                                ),
                                "break_even_credit_points": (
                                    break_even_credit
                                ),
                                "elevated_full_size_pnl_points": (
                                    full_size_elevated_pnl
                                ),
                                "position_equivalent_entries": (
                                    exposure_units
                                ),
                                "exposure_retained": (
                                    safe_divide(
                                        exposure_units,
                                        len(sample_data),
                                    )
                                ),
                                "risk_points_deployed": (
                                    risk_points_deployed
                                ),
                                "credit_points_received": (
                                    credit_points_received
                                ),
                                "intrinsic_points_paid": (
                                    intrinsic_points_paid
                                ),
                                "total_pnl_points": (
                                    total_pnl_points
                                ),
                                "total_pnl_dollars": (
                                    total_pnl_dollars
                                ),
                                "average_pnl_points_per_scheduled_entry": (
                                    safe_divide(
                                        total_pnl_points,
                                        len(sample_data),
                                    )
                                ),
                                "average_pnl_points_per_position_equivalent": (
                                    safe_divide(
                                        total_pnl_points,
                                        exposure_units,
                                    )
                                ),
                                "pnl_to_total_max_risk": (
                                    safe_divide(
                                        total_pnl_points,
                                        risk_points_deployed,
                                    )
                                ),
                                "losing_trade_count": int(
                                    (
                                        (
                                            base_pnl_points < 0
                                        )
                                        & (
                                            position_multiplier > 0
                                        )
                                    ).sum()
                                ),
                                "weighted_losing_exposure": float(
                                    (
                                        position_multiplier
                                        * (
                                            base_pnl_points
                                            < 0
                                        ).astype(int).to_numpy()
                                    ).sum()
                                ),
                                "worst_trade_points": float(
                                    weighted_pnl_points.min()
                                ),
                                "maximum_drawdown_points": (
                                    drawdown_points
                                ),
                                "maximum_drawdown_dollars": (
                                    drawdown_points
                                    * contract_multiplier
                                ),
                                f"worst_{rolling_window}_trade_sum_points": (
                                    worst_window_points
                                ),
                                f"worst_{rolling_window}_trade_sum_dollars": (
                                    worst_window_points
                                    * contract_multiplier
                                ),
                            }
                        )

                        trade_frames.append(
                            pd.DataFrame(
                                {
                                    "sample_id": sample_id,
                                    "model_id": model_id,
                                    "policy_id": policy_id,
                                    "credit_points": credit,
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
                                    "position_multiplier": (
                                        position_multiplier
                                    ),
                                    "short_strike": (
                                        sample_data[
                                            "scenario_short_strike"
                                        ].to_numpy()
                                    ),
                                    "long_strike": (
                                        sample_data[
                                            "scenario_long_strike"
                                        ].to_numpy()
                                    ),
                                    "expiration_settlement": (
                                        sample_data[
                                            "scenario_expiration_settlement"
                                        ].to_numpy()
                                    ),
                                    "spread_intrinsic_points": (
                                        sample_data[
                                            "spread_intrinsic_points"
                                        ].to_numpy()
                                    ),
                                    "pnl_points_before_policy": (
                                        base_pnl_points.to_numpy()
                                    ),
                                    "weighted_pnl_points": (
                                        weighted_pnl_points
                                    ),
                                    "weighted_pnl_dollars": (
                                        weighted_pnl_dollars
                                    ),
                                }
                            )
                        )

        results = pd.DataFrame(result_rows)

        full_size = results.loc[
            results["policy_id"] == "full_size",
            [
                "sample_id",
                "model_id",
                "credit_points",
                "total_pnl_points",
                "total_pnl_dollars",
                "credit_points_received",
                "intrinsic_points_paid",
                "risk_points_deployed",
                "maximum_drawdown_points",
                f"worst_{rolling_window}_trade_sum_points",
            ],
        ].rename(
            columns={
                "total_pnl_points": (
                    "full_size_total_pnl_points"
                ),
                "total_pnl_dollars": (
                    "full_size_total_pnl_dollars"
                ),
                "credit_points_received": (
                    "full_size_credit_points_received"
                ),
                "intrinsic_points_paid": (
                    "full_size_intrinsic_points_paid"
                ),
                "risk_points_deployed": (
                    "full_size_risk_points_deployed"
                ),
                "maximum_drawdown_points": (
                    "full_size_maximum_drawdown_points"
                ),
                f"worst_{rolling_window}_trade_sum_points": (
                    f"full_size_worst_{rolling_window}_trade_sum_points"
                ),
            }
        )

        results = results.merge(
            full_size,
            on=[
                "sample_id",
                "model_id",
                "credit_points",
            ],
            how="left",
            validate="many_to_one",
        )

        results["pnl_difference_vs_full_size_points"] = (
            results["total_pnl_points"]
            - results["full_size_total_pnl_points"]
        )

        results["pnl_difference_vs_full_size_dollars"] = (
            results["total_pnl_dollars"]
            - results["full_size_total_pnl_dollars"]
        )

        results["credit_points_foregone"] = (
            results["full_size_credit_points_received"]
            - results["credit_points_received"]
        )

        results["intrinsic_points_avoided"] = (
            results["full_size_intrinsic_points_paid"]
            - results["intrinsic_points_paid"]
        )

        results["risk_exposure_reduction"] = (
            1.0
            - results["risk_points_deployed"]
            / results["full_size_risk_points_deployed"]
        )

        results["drawdown_reduction_points"] = (
            results["full_size_maximum_drawdown_points"]
            - results["maximum_drawdown_points"]
        )

        results["drawdown_reduction_percent"] = (
            results["drawdown_reduction_points"]
            / results["full_size_maximum_drawdown_points"]
        )

        results[
            f"worst_{rolling_window}_trade_improvement_points"
        ] = (
            results[
                f"worst_{rolling_window}_trade_sum_points"
            ]
            - results[
                f"full_size_worst_{rolling_window}_trade_sum_points"
            ]
        )

        results = results.sort_values(
            [
                "sample_id",
                "credit_points",
                "model_id",
                "elevated_multiplier",
            ],
            ascending=[True, True, True, False],
        ).reset_index(drop=True)

        trades = pd.concat(
            trade_frames,
            ignore_index=True,
        )

        RESULTS_DIR.mkdir(
            parents=True,
            exist_ok=True,
        )

        results.to_csv(
            RESULTS_OUTPUT,
            index=False,
        )

        connection.register(
            "spread_pnl_results_dataframe",
            results,
        )

        connection.register(
            "spread_pnl_trades_dataframe",
            trades,
        )

        connection.execute(
            """
            CREATE OR REPLACE TABLE
                pcs.spread_pnl_policy_results AS
            SELECT *
            FROM spread_pnl_results_dataframe
            ORDER BY
                sample_id,
                credit_points,
                model_id,
                elevated_multiplier DESC
            """
        )

        connection.execute(
            """
            CREATE OR REPLACE TABLE
                pcs.spread_pnl_trade_scenarios AS
            SELECT *
            FROM spread_pnl_trades_dataframe
            ORDER BY
                sample_id,
                model_id,
                policy_id,
                credit_points,
                expiration_proxy_date,
                entry_date
            """
        )

        forward = results.loc[
            results["sample_id"] == "forward_test"
        ]

        anchor_credit = min(credit_scenarios)

        anchor_rows = forward.loc[
            forward["credit_points"] == anchor_credit
        ].sort_values(
            [
                "model_id",
                "elevated_multiplier",
            ],
            ascending=[True, False],
        )

        print(
            f"\nForward-test policy results at "
            f"{anchor_credit:.2f}-point credit"
        )
        print("-" * 168)

        for row in anchor_rows.itertuples(index=False):
            print(
                f"{row.model_label:<39} | "
                f"{row.policy_label:<26} | "
                f"Exposure={row.exposure_retained:>6.1%} | "
                f"P&L=${row.total_pnl_dollars:>9,.0f} | "
                f"vs full=${row.pnl_difference_vs_full_size_dollars:>8,.0f} | "
                f"Max DD=${row.maximum_drawdown_dollars:>8,.0f} | "
                f"DD cut={row.drawdown_reduction_percent:>6.1%} | "
                f"Worst {rolling_window}=${getattr(row, f'worst_{rolling_window}_trade_sum_dollars'):>8,.0f}"
            )

        print("\nForward-test elevated-regime break-even credits")
        print("-" * 92)

        break_even_rows = (
            forward[
                [
                    "model_id",
                    "model_label",
                    "break_even_credit_points",
                    "elevated_entries",
                    "elevated_failures",
                ]
            ]
            .drop_duplicates()
            .sort_values("model_id")
        )

        for row in break_even_rows.itertuples(index=False):
            print(
                f"{row.model_label:<39} | "
                f"Elevated N={row.elevated_entries:>3} | "
                f"Failures={row.elevated_failures:>2} | "
                f"Break-even average credit="
                f"{row.break_even_credit_points:>6.2f} points"
            )

        print("\nForward-test P&L difference versus full size")
        print("-" * 128)

        focus_policies = {
            "reduce_to_50_percent",
            "skip_elevated",
        }

        focus_rows = forward.loc[
            forward["policy_id"].isin(
                focus_policies
            )
        ].sort_values(
            [
                "model_id",
                "policy_id",
                "credit_points",
            ]
        )

        for row in focus_rows.itertuples(index=False):
            print(
                f"{row.model_label:<39} | "
                f"{row.policy_label:<26} | "
                f"Credit={row.credit_points:>5.2f} | "
                f"P&L difference=${row.pnl_difference_vs_full_size_dollars:>8,.0f} | "
                f"DD cut={row.drawdown_reduction_percent:>6.1%} | "
                f"Worst-{rolling_window} improvement="
                f"${getattr(row, f'worst_{rolling_window}_trade_improvement_points') * contract_multiplier:>8,.0f}"
            )

        summary_lines = [
            "MNQ PCS REGIME RESEARCH — SPREAD P&L SCENARIOS",
            "=" * 55,
            "",
            "Methodological status",
            "-" * 55,
            (
                "Fixed-credit expiration proxy; not a historical "
                "options backtest."
            ),
            f"Short strike source: {short_strike_source}",
            f"Expiration settlement source: {settlement_source}",
            f"Spread width: {spread_width:.2f} points",
            (
                f"Contract multiplier: "
                f"${contract_multiplier:.2f} per point"
            ),
            (
                "Credit scenarios: "
                + ", ".join(
                    f"{value:.2f}"
                    for value in credit_scenarios
                )
            ),
            "",
            "Forward-test elevated-regime break-even credits",
            "-" * 55,
        ]

        for row in break_even_rows.itertuples(index=False):
            summary_lines.append(
                f"{row.model_label} | "
                f"N={row.elevated_entries} | "
                f"Failures={row.elevated_failures} | "
                f"Break-even credit="
                f"{row.break_even_credit_points:.2f}"
            )

        summary_lines.extend(
            [
                "",
                (
                    f"Forward-test policies at "
                    f"{anchor_credit:.2f}-point credit"
                ),
                "-" * 55,
            ]
        )

        for row in anchor_rows.itertuples(index=False):
            summary_lines.append(
                f"{row.model_label} | "
                f"{row.policy_label} | "
                f"Exposure retained={row.exposure_retained:.1%} | "
                f"Total P&L=${row.total_pnl_dollars:,.0f} | "
                f"P&L difference vs full="
                f"${row.pnl_difference_vs_full_size_dollars:,.0f} | "
                f"Max drawdown=${row.maximum_drawdown_dollars:,.0f} | "
                f"Drawdown reduction="
                f"{row.drawdown_reduction_percent:.1%}"
            )

        SUMMARY_OUTPUT.write_text(
            "\n".join(summary_lines) + "\n",
            encoding="utf-8",
        )

        print("\nMethodological status")
        print("-" * 92)

        print(
            "Fixed-credit expiration proxy only; no historical "
            "option quotes, fills, strike rounding, or compounding."
        )

        print(f"\nResults output: {RESULTS_OUTPUT}")
        print(f"Summary output: {SUMMARY_OUTPUT}")
        print(
            "DuckDB tables: pcs.spread_pnl_policy_results, "
            "pcs.spread_pnl_trade_scenarios"
        )

        print(
            "Spread P&L scenario analysis completed successfully."
        )

        return 0

    except Exception as exc:
        print(f"\nERROR: {exc}")
        return 1

    finally:
        connection.close()


if __name__ == "__main__":
    sys.exit(main())

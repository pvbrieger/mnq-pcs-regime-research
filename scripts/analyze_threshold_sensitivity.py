"""Test whether the candidate MNQ PCS regime survives nearby thresholds."""
from __future__ import annotations

import sys
from itertools import product
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from scipy.stats import fisher_exact

ROOT = Path(__file__).resolve().parents[1]
DB = ROOT / "database" / "research.duckdb"
OUT = ROOT / "results" / "threshold_sensitivity"
GRID_CSV = OUT / "threshold_sensitivity_grid.csv"
NEIGHBOR_CSV = OUT / "threshold_sensitivity_anchor_neighbors.csv"
SUMMARY_TXT = OUT / "threshold_sensitivity_summary.txt"

ROLLING = (1.10, 1.15, 1.20, 1.25, 1.30)
WEEKLY = (1.10, 1.15, 1.20, 1.25, 1.30)
SMAS = (30, 35, 40, 45, 50)
CUTOFFS = (1, 2, 3)
ANCHOR = (1.20, 1.20, 40, 2)
PERIODS = (
    ("2005_2012", "2005-01-01", "2012-12-31"),
    ("2013_2019", "2013-01-01", "2019-12-31"),
    ("2020_2026", "2020-01-01", "2026-12-31"),
)


def ratio(a: float, b: float) -> float:
    return np.nan if b == 0 or pd.isna(b) else a / b


def summarize(df: pd.DataFrame, elevated: pd.Series) -> dict:
    elevated = elevated.fillna(False).astype(bool)
    normal = ~elevated
    en, nn = int(elevated.sum()), int(normal.sum())
    ef = int(df.loc[elevated, "expiration_failure"].sum())
    nf = int(df.loc[normal, "expiration_failure"].sum())
    er, nr = ratio(ef, en), ratio(nf, nn)
    lift = ratio(er, nr)
    total_failures = int(df["expiration_failure"].sum())
    if en and nn:
        odds, p_value = fisher_exact(
            [[ef, en - ef], [nf, nn - nf]], alternative="two-sided"
        )
    else:
        odds, p_value = np.nan, np.nan
    return {
        "normal_observations": nn,
        "normal_failures": nf,
        "normal_failure_rate": nr,
        "elevated_observations": en,
        "elevated_share": ratio(en, len(df)),
        "elevated_failures": ef,
        "elevated_failure_rate": er,
        "elevated_rate_lift_vs_normal": lift,
        "elevated_failure_capture": ratio(ef, total_failures),
        "odds_ratio": float(odds) if np.isfinite(odds) else np.nan,
        "fisher_p_value": float(p_value),
    }


def failure_clusters(failures: pd.DataFrame) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    rows = failures.sort_values(["entry_date", "expiration_proxy_date"])
    clusters: list[tuple[pd.Timestamp, pd.Timestamp]] = []
    start = end = None
    for row in rows.itertuples(index=False):
        entry = pd.Timestamp(row.entry_date)
        expiry = pd.Timestamp(row.expiration_proxy_date)
        if end is None or entry > end:
            if start is not None:
                clusters.append((start, end))
            start, end = entry, expiry
        else:
            end = max(end, expiry)
    if start is not None:
        clusters.append((start, end))
    return clusters


def classification(row: dict) -> str:
    if row["elevated_observations"] < 40 or row["elevated_failures"] < 3:
        return "Too sparse"
    if (
        row["elevated_observations"] >= 100
        and row["elevated_failures"] >= 6
        and row["elevated_rate_lift_vs_normal"] >= 2.0
        and row["higher_risk_subperiods"] == 3
        and row["minimum_cluster_removal_lift"] >= 1.5
    ):
        return "Robust"
    if (
        row["elevated_observations"] >= 75
        and row["elevated_failures"] >= 4
        and row["elevated_rate_lift_vs_normal"] >= 1.5
        and row["higher_risk_subperiods"] >= 2
        and row["minimum_cluster_removal_lift"] > 1.0
    ):
        return "Supportive"
    return "Weak or unstable"


def main() -> int:
    print("MNQ PCS Regime Research — Threshold Sensitivity")
    print("=" * 51)
    if not DB.is_file():
        print(f"ERROR: Missing database: {DB}")
        return 1

    con = duckdb.connect(str(DB))
    try:
        df = con.execute(
            """
            SELECT entry_date, expiration_proxy_date, weekly_close,
                   weekly_range_atr14_ratio, rolling4w_range_atr14_ratio,
                   below_sma40, expiration_failure, complete_core_features
            FROM pcs.friday_regime_features
            ORDER BY entry_date
            """
        ).fetchdf()
        df["entry_date"] = pd.to_datetime(df["entry_date"])
        df["expiration_proxy_date"] = pd.to_datetime(df["expiration_proxy_date"])
        df["expiration_failure"] = df["expiration_failure"].fillna(False).astype(bool)
        df["below_sma40"] = df["below_sma40"].fillna(False).astype(bool)

        for length in SMAS:
            average = df["weekly_close"].rolling(length).mean()
            df[f"below_sma_{length}"] = df["weekly_close"] < average

        data = df.loc[df["complete_core_features"].fillna(False)].copy()
        mismatches = int((data["below_sma40"] != data["below_sma_40"]).sum())
        if mismatches:
            raise ValueError(f"40-week SMA reconstruction mismatched {mismatches} rows.")

        clusters = failure_clusters(data.loc[data["expiration_failure"]])
        grid_size = len(ROLLING) * len(WEEKLY) * len(SMAS) * len(CUTOFFS)
        print(
            f"Analysis observations: {len(data):,} | "
            f"Failures: {int(data['expiration_failure'].sum())}"
        )
        print(f"Grid combinations: {grid_size:,} | Failure clusters: {len(clusters)}")

        results = []
        for rolling_threshold, weekly_threshold, sma_length, cutoff in product(
            ROLLING, WEEKLY, SMAS, CUTOFFS
        ):
            score = (
                (data["rolling4w_range_atr14_ratio"] >= rolling_threshold).astype(int)
                + (data["weekly_range_atr14_ratio"] >= weekly_threshold).astype(int)
                + data[f"below_sma_{sma_length}"].astype(int)
            )
            elevated = score >= cutoff
            row = summarize(data, elevated)

            higher_periods = 0
            period_lifts = []
            for label, start, end in PERIODS:
                mask = data["entry_date"].between(pd.Timestamp(start), pd.Timestamp(end))
                period_result = summarize(data.loc[mask], elevated.loc[mask])
                row[f"{label}_elevated_observations"] = period_result["elevated_observations"]
                row[f"{label}_elevated_failures"] = period_result["elevated_failures"]
                row[f"{label}_elevated_failure_rate"] = period_result["elevated_failure_rate"]
                row[f"{label}_normal_failure_rate"] = period_result["normal_failure_rate"]
                row[f"{label}_lift"] = period_result["elevated_rate_lift_vs_normal"]
                if (
                    not pd.isna(period_result["elevated_failure_rate"])
                    and not pd.isna(period_result["normal_failure_rate"])
                    and period_result["elevated_failure_rate"]
                    > period_result["normal_failure_rate"]
                ):
                    higher_periods += 1
                if not pd.isna(period_result["elevated_rate_lift_vs_normal"]):
                    period_lifts.append(period_result["elevated_rate_lift_vs_normal"])

            cluster_lifts = []
            higher_clusters = 0
            for start, end in clusters:
                remove = (data["entry_date"] <= end) & (
                    data["expiration_proxy_date"] >= start
                )
                reduced = summarize(data.loc[~remove], elevated.loc[~remove])
                lift = reduced["elevated_rate_lift_vs_normal"]
                if not pd.isna(lift):
                    cluster_lifts.append(lift)
                if (
                    not pd.isna(reduced["elevated_failure_rate"])
                    and not pd.isna(reduced["normal_failure_rate"])
                    and reduced["elevated_failure_rate"]
                    > reduced["normal_failure_rate"]
                ):
                    higher_clusters += 1

            row.update(
                {
                    "rolling4w_atr_threshold": rolling_threshold,
                    "weekly_atr_threshold": weekly_threshold,
                    "sma_length_weeks": sma_length,
                    "elevated_score_cutoff": cutoff,
                    "higher_risk_subperiods": higher_periods,
                    "minimum_subperiod_lift": min(period_lifts) if period_lifts else np.nan,
                    "median_subperiod_lift": float(np.median(period_lifts))
                    if period_lifts
                    else np.nan,
                    "cluster_tests_higher": higher_clusters,
                    "cluster_test_count": len(clusters),
                    "minimum_cluster_removal_lift": min(cluster_lifts)
                    if cluster_lifts
                    else np.nan,
                    "median_cluster_removal_lift": float(np.median(cluster_lifts))
                    if cluster_lifts
                    else np.nan,
                    "is_anchor_definition": (
                        rolling_threshold,
                        weekly_threshold,
                        sma_length,
                        cutoff,
                    )
                    == ANCHOR,
                }
            )
            row["classification"] = classification(row)
            results.append(row)

        result = pd.DataFrame(results)
        class_rank = {
            "Robust": 0,
            "Supportive": 1,
            "Weak or unstable": 2,
            "Too sparse": 3,
        }
        result["_rank"] = result["classification"].map(class_rank)
        result = result.sort_values(
            [
                "_rank",
                "higher_risk_subperiods",
                "minimum_cluster_removal_lift",
                "elevated_rate_lift_vs_normal",
                "elevated_observations",
            ],
            ascending=[True, False, False, False, False],
        ).drop(columns="_rank").reset_index(drop=True)
        result.insert(0, "robustness_rank", np.arange(1, len(result) + 1))

        anchor = result.loc[result["is_anchor_definition"]]
        if len(anchor) != 1:
            raise ValueError("Anchor definition was not uniquely identified.")
        anchor = anchor.iloc[0]

        neighbors = result.loc[
            result["rolling4w_atr_threshold"].isin((1.15, 1.20, 1.25))
            & result["weekly_atr_threshold"].isin((1.15, 1.20, 1.25))
            & result["sma_length_weeks"].isin((35, 40, 45))
            & (result["elevated_score_cutoff"] == 2)
        ].sort_values(
            ["rolling4w_atr_threshold", "weekly_atr_threshold", "sma_length_weeks"]
        )

        OUT.mkdir(parents=True, exist_ok=True)
        result.to_csv(GRID_CSV, index=False)
        neighbors.to_csv(NEIGHBOR_CSV, index=False)

        con.register("threshold_sensitivity_dataframe", result)
        con.execute(
            """
            CREATE OR REPLACE TABLE pcs.threshold_sensitivity_results AS
            SELECT * FROM threshold_sensitivity_dataframe
            ORDER BY robustness_rank
            """
        )

        robust = int((result["classification"] == "Robust").sum())
        supportive = int((result["classification"] == "Supportive").sum())
        all_periods = int((result["higher_risk_subperiods"] == 3).sum())
        all_clusters = int(
            (result["cluster_tests_higher"] == result["cluster_test_count"]).sum()
        )
        neighbor_robust = int((neighbors["classification"] == "Robust").sum())
        neighbor_good = int(
            neighbors["classification"].isin(("Robust", "Supportive")).sum()
        )

        summary = [
            "MNQ PCS REGIME RESEARCH — THRESHOLD SENSITIVITY",
            "=" * 55,
            "",
            f"Analysis observations: {len(data):,}",
            f"Expiration failures: {int(data['expiration_failure'].sum())}",
            f"Threshold combinations tested: {len(result):,}",
            f"Failure clusters tested: {len(clusters)}",
            "",
            "Anchor definition",
            "-" * 55,
            "Rolling four-week ATR threshold: 1.20",
            "Weekly ATR threshold: 1.20",
            "SMA length: 40 weeks",
            "Elevated score cutoff: 2 active flags",
            f"Anchor elevated observations: {int(anchor['elevated_observations'])}",
            f"Anchor elevated failures: {int(anchor['elevated_failures'])}",
            f"Anchor normal failure rate: {anchor['normal_failure_rate']:.2%}",
            f"Anchor elevated failure rate: {anchor['elevated_failure_rate']:.2%}",
            f"Anchor lift versus normal: {anchor['elevated_rate_lift_vs_normal']:.2f}x",
            f"Anchor minimum cluster-removal lift: {anchor['minimum_cluster_removal_lift']:.2f}x",
            f"Anchor classification: {anchor['classification']}",
            "",
            "Grid robustness",
            "-" * 55,
            f"Robust definitions: {robust}/{len(result)}",
            f"Supportive definitions: {supportive}/{len(result)}",
            f"Higher elevated risk in all three subperiods: {all_periods}/{len(result)}",
            f"Higher elevated risk after every cluster removal: {all_clusters}/{len(result)}",
            "",
            "Local anchor neighborhood",
            "-" * 55,
            f"Neighboring definitions tested: {len(neighbors)}",
            f"Robust neighbors: {neighbor_robust}/{len(neighbors)}",
            f"Supportive or robust neighbors: {neighbor_good}/{len(neighbors)}",
        ]
        SUMMARY_TXT.write_text("\n".join(summary) + "\n", encoding="utf-8")

        print("\nAnchor definition")
        print("-" * 82)
        print("Rolling ATR >= 1.20 | Weekly ATR >= 1.20 | Below 40-week SMA | Score >= 2")
        print(
            f"Normal: N={int(anchor['normal_observations'])} | "
            f"Failures={int(anchor['normal_failures'])} "
            f"({anchor['normal_failure_rate']:.2%})"
        )
        print(
            f"Elevated: N={int(anchor['elevated_observations'])} | "
            f"Failures={int(anchor['elevated_failures'])} "
            f"({anchor['elevated_failure_rate']:.2%}) | "
            f"Lift={anchor['elevated_rate_lift_vs_normal']:.2f}x"
        )
        print(
            f"Subperiods higher: {int(anchor['higher_risk_subperiods'])}/3 | "
            f"Cluster removals higher: {int(anchor['cluster_tests_higher'])}/"
            f"{int(anchor['cluster_test_count'])} | "
            f"Minimum cluster lift: {anchor['minimum_cluster_removal_lift']:.2f}x"
        )
        print(f"Anchor classification: {anchor['classification']}")

        print("\nGrid robustness")
        print("-" * 82)
        print(f"Robust definitions: {robust}/{len(result)}")
        print(f"Supportive definitions: {supportive}/{len(result)}")
        print(f"Higher risk in all three subperiods: {all_periods}/{len(result)}")
        print(f"Higher risk after every cluster removal: {all_clusters}/{len(result)}")

        print("\nLocal anchor neighborhood")
        print("-" * 82)
        print(
            f"Definitions tested: {len(neighbors)} | Robust: {neighbor_robust} | "
            f"Supportive or robust: {neighbor_good}"
        )

        print("\nTop 10 robustness-ranked definitions")
        print("-" * 132)
        for row in result.head(10).itertuples(index=False):
            print(
                f"{row.robustness_rank:>3}. "
                f"R4W>={row.rolling4w_atr_threshold:.2f} | "
                f"W1>={row.weekly_atr_threshold:.2f} | "
                f"SMA={row.sma_length_weeks:>2} | "
                f"Score>={row.elevated_score_cutoff} | "
                f"N={row.elevated_observations:>3} | "
                f"Fail={row.elevated_failures:>2} ({row.elevated_failure_rate:>6.2%}) | "
                f"Lift={row.elevated_rate_lift_vs_normal:>5.2f}x | "
                f"MinCluster={row.minimum_cluster_removal_lift:>5.2f}x | "
                f"{row.classification}"
            )

        print(f"\nGrid output: {GRID_CSV}")
        print(f"Anchor-neighbor output: {NEIGHBOR_CSV}")
        print(f"Summary output: {SUMMARY_TXT}")
        print("DuckDB table: pcs.threshold_sensitivity_results")
        print("Threshold sensitivity completed successfully.")
        return 0

    except Exception as exc:
        print(f"\nERROR: {exc}")
        return 1
    finally:
        con.close()


if __name__ == "__main__":
    sys.exit(main())

"""Business threshold selection and customer-level retention policy logic."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


def evaluate_thresholds(
    y_true: pd.Series | np.ndarray,
    probabilities: np.ndarray,
    *,
    coupon_cost: float,
    churn_cost: float,
) -> pd.DataFrame:
    """Evaluate net profit for thresholds from 0.01 to 0.99.

    Business value convention:
    - True positive: churn is targeted, so avoided churn value minus coupon cost.
    - False positive: customer was not going to churn, so coupon is wasted.
    - False negative: churn is missed, so churn value is lost.
    - True negative: no intervention and no churn loss.
    """

    y_array = np.asarray(y_true).astype(int)
    probability_array = np.asarray(probabilities, dtype=float)
    rows: list[dict[str, float | int]] = []

    for threshold in np.round(np.arange(0.01, 1.00, 0.01), 2):
        targeted = probability_array >= threshold
        true_positive = int(((targeted == 1) & (y_array == 1)).sum())
        false_positive = int(((targeted == 1) & (y_array == 0)).sum())
        false_negative = int(((targeted == 0) & (y_array == 1)).sum())
        true_negative = int(((targeted == 0) & (y_array == 0)).sum())

        total_profit = (
            true_positive * (churn_cost - coupon_cost)
            - false_positive * coupon_cost
            - false_negative * churn_cost
        )

        rows.append(
            {
                "threshold": threshold,
                "true_positive": true_positive,
                "false_positive": false_positive,
                "false_negative": false_negative,
                "true_negative": true_negative,
                "customers_targeted": int(targeted.sum()),
                "total_profit": float(total_profit),
                "profit_per_customer": float(total_profit / len(y_array)) if len(y_array) else 0.0,
            }
        )

    return pd.DataFrame(rows)


def select_best_threshold(threshold_frame: pd.DataFrame) -> float:
    """Select the threshold with maximum total expected profit."""

    if threshold_frame.empty:
        raise ValueError("Threshold analysis is empty; cannot select a decision threshold.")
    return float(threshold_frame.sort_values(["total_profit", "threshold"], ascending=[False, True]).iloc[0]["threshold"])


def save_selected_threshold(path: Path, *, threshold: float, coupon_cost: float, churn_cost: float) -> None:
    """Save selected threshold and assumptions as JSON."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "selected_threshold": threshold,
                "coupon_cost": coupon_cost,
                "churn_cost": churn_cost,
                "selection_rule": "threshold with maximum total expected profit on evaluation set",
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def evaluate_thresholds_constrained(
    y_true: pd.Series | np.ndarray,
    probabilities: np.ndarray,
    *,
    coupon_cost: float,
    churn_cost: float,
    max_target_rate: float,
    minimum_precision: float,
    budget_cap: float,
) -> pd.DataFrame:
    """Evaluate threshold profit with operational feasibility constraints."""

    frame = evaluate_thresholds(
        y_true,
        probabilities,
        coupon_cost=coupon_cost,
        churn_cost=churn_cost,
    )
    y_array = np.asarray(y_true).astype(int)
    n_rows = max(len(y_array), 1)

    precision_values: list[float] = []
    for _, row in frame.iterrows():
        predicted_positive = row["true_positive"] + row["false_positive"]
        precision = row["true_positive"] / predicted_positive if predicted_positive else 0.0
        precision_values.append(float(precision))

    frame["precision"] = precision_values
    frame["target_rate"] = frame["customers_targeted"] / n_rows
    frame["estimated_coupon_budget"] = frame["customers_targeted"] * coupon_cost
    frame["meets_max_target_rate"] = frame["target_rate"] <= max_target_rate
    frame["meets_minimum_precision"] = frame["precision"] >= minimum_precision
    frame["meets_budget_cap"] = frame["estimated_coupon_budget"] <= budget_cap
    frame["constraints_satisfied"] = (
        frame["meets_max_target_rate"]
        & frame["meets_minimum_precision"]
        & frame["meets_budget_cap"]
    )

    budget_denominator = max(float(budget_cap), 1.0)
    frame["constraint_violation_score"] = (
        np.maximum(0.0, frame["target_rate"] - max_target_rate)
        + np.maximum(0.0, minimum_precision - frame["precision"])
        + np.maximum(0.0, frame["estimated_coupon_budget"] - budget_cap) / budget_denominator
    )
    return frame


def select_constrained_threshold(threshold_frame: pd.DataFrame) -> tuple[float, str | None]:
    """Select best feasible threshold, or least-bad threshold with warning."""

    feasible = threshold_frame[threshold_frame["constraints_satisfied"] == True]  # noqa: E712
    if not feasible.empty:
        selected = feasible.sort_values(["total_profit", "threshold"], ascending=[False, True]).iloc[0]
        return float(selected["threshold"]), None

    selected = threshold_frame.sort_values(
        ["constraint_violation_score", "total_profit", "threshold"],
        ascending=[True, False, True],
    ).iloc[0]
    warning = (
        "No threshold satisfied all operational constraints. Selected the least-bad "
        "threshold by minimum constraint violation, then maximum expected profit."
    )
    return float(selected["threshold"]), warning


def save_selected_threshold_constrained(
    path: Path,
    *,
    threshold: float,
    coupon_cost: float,
    churn_cost: float,
    max_target_rate: float,
    minimum_precision: float,
    budget_cap: float,
    warning: str | None,
) -> None:
    """Save constrained threshold and assumptions as JSON."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "selected_threshold": threshold,
                "coupon_cost": coupon_cost,
                "churn_cost": churn_cost,
                "max_target_rate": max_target_rate,
                "minimum_precision": minimum_precision,
                "budget_cap": budget_cap,
                "selection_rule": "maximum expected profit subject to operational constraints",
                "warning": warning,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def assign_retention_policy(
    customers: pd.DataFrame,
    *,
    customer_id_column: str,
    probability_column: str,
    ltv_column: str,
    selected_threshold: float,
    ltv_threshold: float,
    shap_threshold: float,
) -> pd.DataFrame:
    """Assign risk segments, coupon budgets, and retention actions.

    The project requirement uses the phrase "SHAP score > 0.7". Here, SHAP
    score is interpreted as a normalized individual-level positive SHAP
    contribution toward churn risk. It ranges from 0 to 1, where 1 means the
    customer's features had the strongest positive churn-risk contribution in
    the scored population.
    """

    scored = customers.copy()
    scored["estimated_ltv"] = pd.to_numeric(scored[ltv_column], errors="coerce").fillna(0.0)
    scored[probability_column] = pd.to_numeric(scored[probability_column], errors="coerce").fillna(0.0)
    if "shap_score" not in scored.columns:
        scored["shap_score"] = 0.0

    high_cutoff = selected_threshold
    medium_cutoff = 0.5 * selected_threshold
    scored["risk_segment"] = np.select(
        [
            scored[probability_column] >= high_cutoff,
            scored[probability_column] >= medium_cutoff,
        ],
        ["High risk", "Medium risk"],
        default="Low risk",
    )

    scored["recommended_coupon"] = 0.0
    high_value = scored["estimated_ltv"] >= ltv_threshold
    scored.loc[(scored["risk_segment"] == "High risk") & high_value, "recommended_coupon"] = 15.0
    scored.loc[(scored["risk_segment"] == "Medium risk") & high_value, "recommended_coupon"] = 5.0

    explicit_shap_rule = (scored["shap_score"] > shap_threshold) & (scored["estimated_ltv"] > ltv_threshold)
    scored.loc[explicit_shap_rule, "recommended_coupon"] = 15.0

    scored["retention_action"] = np.select(
        [
            scored["recommended_coupon"] >= 15,
            scored["recommended_coupon"] > 0,
            scored["risk_segment"] == "Medium risk",
        ],
        [
            "Send $15 retention coupon",
            "Send $5 light-touch retention coupon",
            "Monitor; no coupon because LTV is below threshold",
        ],
        default="No coupon; standard lifecycle messaging",
    )

    output_columns = [
        customer_id_column,
        probability_column,
        "risk_segment",
        "estimated_ltv",
        "shap_score",
        "recommended_coupon",
        "retention_action",
    ]
    return scored[output_columns].rename(columns={customer_id_column: "customer_id"})


def assign_retention_policy_constrained(
    customers: pd.DataFrame,
    *,
    customer_id_column: str,
    probability_column: str,
    ltv_column: str,
    selected_threshold: float,
    ltv_threshold: float,
    budget_cap: float | None = None,
) -> pd.DataFrame:
    """Assign an operationally constrained retention policy.

    High-risk customers must clear the constrained threshold, meet the LTV
    threshold, and fall in the top 20% of model risk scores. This prevents the
    default cost matrix from targeting nearly every customer when probabilities
    are compressed or model signal is weak.
    """

    scored = customers.copy()
    scored["estimated_ltv"] = pd.to_numeric(scored[ltv_column], errors="coerce").fillna(0.0)
    scored[probability_column] = pd.to_numeric(scored[probability_column], errors="coerce").fillna(0.0)
    risk_top_20_cutoff = scored[probability_column].quantile(0.80)
    high_value = scored["estimated_ltv"] >= ltv_threshold

    high_risk = (
        (scored[probability_column] >= selected_threshold)
        & high_value
        & (scored[probability_column] >= risk_top_20_cutoff)
    )
    medium_risk = (
        ~high_risk
        & (scored[probability_column] >= 0.5 * selected_threshold)
        & high_value
    )

    scored["risk_segment"] = np.select(
        [high_risk, medium_risk],
        ["High risk", "Medium risk"],
        default="Low risk",
    )
    scored["recommended_coupon"] = np.select(
        [high_risk, medium_risk],
        [15.0, 5.0],
        default=0.0,
    )

    if budget_cap is not None:
        priority = np.select([high_risk, medium_risk], [2, 1], default=0)
        scored["_coupon_priority"] = priority
        scored = scored.sort_values(
            ["_coupon_priority", probability_column, "estimated_ltv"],
            ascending=[False, False, False],
        )
        cumulative_budget = scored["recommended_coupon"].cumsum()
        over_budget = cumulative_budget > budget_cap
        scored.loc[over_budget, "recommended_coupon"] = 0.0
        scored = scored.drop(columns=["_coupon_priority"]).sort_index()
    scored["retention_action"] = np.select(
        [scored["recommended_coupon"] >= 15, scored["recommended_coupon"] > 0, high_risk | medium_risk],
        [
            "Send $15 retention coupon",
            "Send $5 light-touch retention coupon",
            "No coupon; eligible but held out by budget/targeting constraints",
        ],
        default="No coupon; standard lifecycle messaging",
    )

    output_columns = [
        customer_id_column,
        probability_column,
        "risk_segment",
        "estimated_ltv",
        "recommended_coupon",
        "retention_action",
    ]
    if "shap_score" in scored.columns:
        output_columns.insert(4, "shap_score")
    return scored[output_columns].rename(columns={customer_id_column: "customer_id"})


def summarize_retention_budget(scored_customers: pd.DataFrame) -> pd.DataFrame:
    """Aggregate coupon budget by risk segment."""

    return (
        scored_customers.groupby("risk_segment", observed=True)
        .agg(
            number_of_customers=("customer_id", "count"),
            average_churn_probability=("predicted_churn_probability", "mean"),
            average_ltv=("estimated_ltv", "mean"),
            total_recommended_coupon_budget=("recommended_coupon", "sum"),
        )
        .reset_index()
        .sort_values("risk_segment")
    )

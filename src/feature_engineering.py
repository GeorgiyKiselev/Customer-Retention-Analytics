"""Customer-level churn feature engineering."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

LOGGER = logging.getLogger(__name__)


def build_churn_modeling_table(
    transactions: pd.DataFrame,
    observation_days: int = 90,
    prediction_days: int = 30,
) -> pd.DataFrame:
    """Builds a snapshot whose prediction window ends at the latest purchase date.

    The observation interval is inclusive and contains exactly ``observation_days``
    calendar dates. The immediately following interval contains exactly
    ``prediction_days`` calendar dates.
    """
    if observation_days <= 0 or prediction_days <= 0:
        raise ValueError("Observation and prediction windows must both be positive integers.")

    required = {"customer_id", "order_id", "order_date", "amount"}
    missing = sorted(required - set(transactions.columns))
    if missing:
        raise ValueError(
            f"Canonical transaction data is missing columns {missing}. "
            f"Available columns: {sorted(transactions.columns)}"
        )

    latest_date = transactions["order_date"].max().normalize()
    earliest_date = transactions["order_date"].min().normalize()
    prediction_end = latest_date
    observation_end = prediction_end - pd.Timedelta(days=prediction_days)
    observation_start = observation_end - pd.Timedelta(days=observation_days - 1)
    prediction_start = observation_end + pd.Timedelta(days=1)

    LOGGER.info(
        "Observation window: %s through %s (%d days, inclusive)",
        observation_start.date(), observation_end.date(), observation_days,
    )
    LOGGER.info(
        "Prediction window: %s through %s (%d days, inclusive)",
        prediction_start.date(), prediction_end.date(), prediction_days,
    )

    history = transactions.loc[
        transactions["order_date"].between(observation_start, observation_end, inclusive="both")
    ].copy()
    future = transactions.loc[
        transactions["order_date"].between(prediction_start, prediction_end, inclusive="both")
    ]
    if history.empty:
        raise ValueError(
            "No transactions fall in the observation window "
            f"{observation_start.date()} through {observation_end.date()}. "
            "Check the mapped order-date column and window arguments."
        )

    # Source data may be one row per order line. Aggregate to one row per order
    # before counting orders and calculating average order value.
    order_aggregations: dict[str, tuple[str, object]] = {
        "order_date": ("order_date", "min"),
        "order_amount": ("amount", "sum"),
    }
    orders = (
        history.groupby(["customer_id", "order_id"], as_index=False, dropna=False)
        .agg(**order_aggregations)
    )

    grouped = orders.groupby("customer_id", sort=False)
    features = grouped.agg(
        last_purchase_date=("order_date", "max"),
        first_purchase_date=("order_date", "min"),
        frequency=("order_date", "nunique"),
        total_orders=("order_id", "nunique"),
        monetary_value=("order_amount", "sum"),
        average_order_value=("order_amount", "mean"),
    )
    features["recency"] = (
        observation_end - features["last_purchase_date"]
    ).dt.days.astype("int64")
    features["tenure"] = (
        observation_end - features["first_purchase_date"]
    ).dt.days.astype("int64")
    sorted_orders = orders.sort_values(["customer_id", "order_date", "order_id"])
    order_gaps = (
        sorted_orders.groupby("customer_id", sort=False)["order_date"]
        .diff()
        .dt.total_seconds()
        .div(86_400)
    )
    gap_grouped = order_gaps.groupby(sorted_orders["customer_id"])
    interpurchase = gap_grouped.mean()
    interpurchase = interpurchase.reindex(features.index)
    features["avg_interpurchase_missing"] = interpurchase.isna().astype("int8")
    features["average_interpurchase_time"] = interpurchase.fillna(float(observation_days))
    features["purchase_gap_std"] = gap_grouped.std().reindex(features.index).fillna(0.0)
    features["max_purchase_gap"] = gap_grouped.max().reindex(features.index).fillna(0.0)
    features["min_purchase_gap"] = gap_grouped.min().reindex(features.index).fillna(0.0)
    features["purchase_gap_cv"] = (
        features["purchase_gap_std"] / features["average_interpurchase_time"].replace(0, np.nan)
    ).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    features["log_monetary_value"] = np.log1p(features["monetary_value"].clip(lower=0))
    features["days_since_first_purchase"] = features["tenure"]
    features["days_since_last_purchase"] = features["recency"]
    features["active_purchase_span_days"] = (
        features["last_purchase_date"] - features["first_purchase_date"]
    ).dt.days.clip(lower=0)
    features["unique_purchase_days"] = features["frequency"]
    features["orders_per_active_day"] = (
        features["total_orders"] / (features["active_purchase_span_days"] + 1)
    ).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    features["avg_basket_value"] = features["average_order_value"]
    features["basket_value_std"] = grouped["order_amount"].std().reindex(features.index).fillna(0.0)
    features["basket_value_cv"] = (
        features["basket_value_std"] / features["average_order_value"].replace(0, np.nan)
    ).replace([np.inf, -np.inf], np.nan).fillna(0.0)

    def _window_order_features(start: pd.Timestamp, end: pd.Timestamp, prefix: str) -> None:
        window_orders = orders.loc[orders["order_date"].between(start, end, inclusive="both")]
        order_counts = window_orders.groupby("customer_id")["order_id"].nunique()
        monetary = window_orders.groupby("customer_id")["order_amount"].sum()
        features[f"orders_{prefix}"] = order_counts.reindex(features.index).fillna(0).astype("int64")
        features[f"monetary_{prefix}"] = monetary.reindex(features.index).fillna(0.0)

    _window_order_features(observation_end - pd.Timedelta(days=6), observation_end, "last_7d")
    _window_order_features(observation_end - pd.Timedelta(days=13), observation_end, "last_14d")
    _window_order_features(observation_end - pd.Timedelta(days=29), observation_end, "last_30d")
    _window_order_features(observation_end - pd.Timedelta(days=59), observation_end, "last_60d")
    _window_order_features(observation_start, observation_start + pd.Timedelta(days=29), "first_30d")

    prior_30_start = observation_end - pd.Timedelta(days=59)
    prior_30_end = observation_end - pd.Timedelta(days=30)
    prior_30 = orders.loc[orders["order_date"].between(prior_30_start, prior_30_end, inclusive="both")]
    prior_orders = prior_30.groupby("customer_id")["order_id"].nunique().reindex(features.index).fillna(0.0)
    prior_monetary = prior_30.groupby("customer_id")["order_amount"].sum().reindex(features.index).fillna(0.0)
    features["monetary_trend_30d_vs_prior"] = (
        (features["monetary_last_30d"] - prior_monetary) / (prior_monetary.abs() + 1.0)
    )
    features["frequency_trend_30d_vs_prior"] = (
        (features["orders_last_30d"] - prior_orders) / (prior_orders.abs() + 1.0)
    )
    features["purchase_acceleration"] = (
        features["frequency_trend_30d_vs_prior"] / (features["average_interpurchase_time"] + 1.0)
    )

    future_customers = set(future["customer_id"].dropna().astype(str))
    features["churn_30d"] = (
        ~features.index.astype(str).isin(future_customers)
    ).astype("int8")

    if "category" in history.columns:
        category_values = history.dropna(subset=["category"])
        category_counts = (
            category_values.groupby(["customer_id", "category"], sort=False)
            .size()
            .rename("count")
            .reset_index()
            .sort_values(["customer_id", "count", "category"], ascending=[True, False, True])
        )
        favorite_category = category_counts.drop_duplicates("customer_id").set_index(
            "customer_id"
        )["category"]
        features["favorite_category"] = favorite_category.reindex(features.index)
        features["unique_categories"] = (
            category_values.groupby("customer_id")["category"].nunique().reindex(features.index)
        )
        total_category_rows = category_values.groupby("customer_id").size().reindex(features.index).fillna(0)
        top_category_rows = category_counts.drop_duplicates("customer_id").set_index("customer_id")[
            "count"
        ].reindex(features.index).fillna(0)
        features["dominant_category_share"] = (
            top_category_rows / total_category_rows.replace(0, np.nan)
        ).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        repeated_category_customers = (
            category_counts.loc[category_counts["count"] > 1]
            .groupby("customer_id")["category"]
            .nunique()
        )
        features["repeat_category_rate"] = (
            repeated_category_customers.reindex(features.index).fillna(0)
            / features["unique_categories"].replace(0, np.nan)
        ).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    if "product" in history.columns:
        features["unique_products"] = (
            history.groupby("customer_id")["product"].nunique().reindex(features.index)
        )

    features["observation_start"] = observation_start.date().isoformat()
    features["observation_end"] = observation_end.date().isoformat()
    features["prediction_end"] = prediction_end.date().isoformat()
    features["source_min_transaction_date"] = earliest_date.date().isoformat()
    features["source_max_transaction_date"] = latest_date.date().isoformat()
    features = features.reset_index()

    ordered_columns = [
        "customer_id", "recency", "frequency", "monetary_value", "tenure",
        "average_interpurchase_time", "avg_interpurchase_missing", "total_orders",
        "average_order_value", "log_monetary_value",
        "orders_last_7d", "orders_last_14d", "orders_last_30d", "orders_last_60d",
        "orders_first_30d", "monetary_last_30d", "monetary_first_30d",
        "monetary_trend_30d_vs_prior", "frequency_trend_30d_vs_prior",
        "days_since_first_purchase", "days_since_last_purchase",
        "purchase_acceleration", "purchase_gap_std", "purchase_gap_cv",
        "max_purchase_gap", "min_purchase_gap", "active_purchase_span_days",
        "unique_purchase_days", "orders_per_active_day", "avg_basket_value",
        "basket_value_std", "basket_value_cv",
    ]
    optional_columns = [
        column for column in (
            "favorite_category", "unique_categories", "unique_products",
            "repeat_category_rate", "dominant_category_share",
        )
        if column in features.columns
    ]
    final_columns = ordered_columns + optional_columns + [
        "churn_30d", "observation_start", "observation_end", "prediction_end",
        "source_min_transaction_date", "source_max_transaction_date",
    ]
    result = features[final_columns].sort_values("customer_id").reset_index(drop=True)
    LOGGER.info(
        "Built %d customer rows; churn rate is %.2f%%.",
        len(result), result["churn_30d"].mean() * 100,
    )
    return result

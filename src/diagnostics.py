"""Diagnostics for churn labels and processed customer behavior."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from analysis_plots import save_current_figure, set_plot_style


def _first_existing(columns: list[str], candidates: list[str]) -> str | None:
    lowered = {column.lower(): column for column in columns}
    for candidate in candidates:
        if candidate.lower() in lowered:
            return lowered[candidate.lower()]
    return None


def _date_column(frame: pd.DataFrame, name: str) -> pd.Series | None:
    if name not in frame.columns:
        return None
    return pd.to_datetime(frame[name], errors="coerce")


def create_churn_label_diagnostics(
    frame: pd.DataFrame,
    *,
    target_column: str,
    customer_id_column: str,
    tables_dir: Path,
    plots_dir: Path,
) -> pd.DataFrame:
    """Save churn-label diagnostics and label/behavior distribution plots."""

    tables_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)
    set_plot_style()

    columns = frame.columns.tolist()
    order_count_col = _first_existing(columns, ["total_orders", "order_count", "orders"])
    recency_col = _first_existing(columns, ["days_since_last_purchase", "recency_days", "recency"])
    gap_col = _first_existing(columns, ["average_interpurchase_time", "avg_interpurchase_days"])

    observation_start = _date_column(frame, "observation_start")
    observation_end = _date_column(frame, "observation_end")
    prediction_end = _date_column(frame, "prediction_end")
    source_min_date = _date_column(frame, "source_min_transaction_date")
    source_max_date = _date_column(frame, "source_max_transaction_date")

    if observation_start is not None and observation_end is not None:
        observation_days = int((observation_end.max() - observation_start.min()).days + 1)
    else:
        observation_days = pd.NA

    if observation_end is not None and prediction_end is not None:
        prediction_days = int((prediction_end.max() - observation_end.max()).days)
        enough_future_time = prediction_days >= 30
    else:
        prediction_days = pd.NA
        enough_future_time = False

    if source_min_date is not None and source_min_date.notna().any():
        min_transaction_date = source_min_date.min().date().isoformat()
    elif observation_start is not None:
        min_transaction_date = observation_start.min().date().isoformat()
    else:
        min_transaction_date = ""

    if source_max_date is not None and source_max_date.notna().any():
        max_transaction_date = source_max_date.max().date().isoformat()
    elif prediction_end is not None:
        max_transaction_date = prediction_end.max().date().isoformat()
    else:
        max_transaction_date = ""

    if order_count_col:
        order_counts = pd.to_numeric(frame[order_count_col], errors="coerce").fillna(0)
        one_order_customers = int((order_counts <= 1).sum())
        multi_order_customers = int((order_counts >= 2).sum())
    else:
        order_counts = pd.Series(dtype="float64")
        one_order_customers = 0
        multi_order_customers = 0

    if recency_col and pd.notna(prediction_days):
        recency = pd.to_numeric(frame[recency_col], errors="coerce")
        unstable = (recency <= float(prediction_days)) & (frame[target_column].astype(int) == 1)
        pct_unstable = float(unstable.mean())
    else:
        pct_unstable = float("nan")

    diagnostics = pd.DataFrame(
        [
            {
                "number_of_customers": int(frame[customer_id_column].nunique()),
                "churn_rate": float(frame[target_column].mean()),
                "customers_with_only_one_order": one_order_customers,
                "customers_with_two_or_more_orders": multi_order_customers,
                "observation_window_length_days": observation_days,
                "prediction_window_length_days": prediction_days,
                "minimum_transaction_date": min_transaction_date,
                "maximum_transaction_date": max_transaction_date,
                "has_enough_future_time_after_observation_window": bool(enough_future_time),
                "percent_labels_potentially_unstable_near_dataset_end": pct_unstable,
            }
        ]
    )
    diagnostics.to_csv(tables_dir / "churn_label_diagnostics.csv", index=False)

    plt.figure(figsize=(7, 5))
    ax = sns.countplot(data=frame, x=target_column)
    ax.set_title("Churn Label Distribution")
    ax.set_xlabel("churn_30d")
    ax.set_ylabel("Customers")
    save_current_figure(plots_dir / "churn_label_distribution.png")

    if gap_col:
        plt.figure(figsize=(8, 5))
        gap_values = pd.to_numeric(frame[gap_col], errors="coerce").dropna()
        sns.histplot(gap_values, bins=40)
        plt.title("Purchase Gap Distribution")
        plt.xlabel("Average days between purchases")
        plt.ylabel("Customers")
        save_current_figure(plots_dir / "purchase_gap_distribution.png")

    if order_count_col:
        plt.figure(figsize=(8, 5))
        clipped_orders = order_counts.clip(upper=order_counts.quantile(0.99))
        sns.histplot(clipped_orders, bins=30)
        plt.title("Customer Order Count Distribution")
        plt.xlabel("Orders in observation window")
        plt.ylabel("Customers")
        save_current_figure(plots_dir / "customer_order_count_distribution.png")

    return diagnostics

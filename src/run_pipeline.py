"""Command-line entry point for the churn data pipeline."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from data_loading import (
    discover_csv_schemas,
    load_transactions,
    resolve_dataset_mapping,
)
from feature_engineering import build_churn_modeling_table

LOGGER = logging.getLogger("churn_pipeline")
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a customer churn modeling table from local E-Commerce CSV files."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        required=True,
        help="Folder containing the unzipped raw CSV files (subfolders are searched too).",
    )
    parser.add_argument(
        "--observation-days",
        type=int,
        default=90,
        help="Length of transactional feature history in days (default: 90).",
    )
    parser.add_argument(
        "--prediction-days",
        type=int,
        default=30,
        help="Length of no-purchase churn horizon in days (default: 30).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Processed output folder (default: {DEFAULT_OUTPUT_DIR}).",
    )
    return parser.parse_args()


def write_data_dictionary(path: Path, observation_days: int, prediction_days: int) -> None:
    content = f"""# Churn Modeling Table Data Dictionary

One row represents a customer with at least one order during the {observation_days}-day
observation window. The label uses the immediately following {prediction_days}-day window.

| Column | Description |
|---|---|
| `customer_id` | Customer identifier from the source transaction table. |
| `recency` | Days from the customer's most recent observed purchase to the observation end. |
| `frequency` | Number of distinct purchase dates in the observation window. |
| `monetary_value` | Sum of order value in the observation window. Line items are summed to order level first. |
| `tenure` | Days from the customer's first purchase in the observation window to the observation end. |
| `average_interpurchase_time` | Mean days between observed orders. Set to the observation-window length for one-order customers. |
| `avg_interpurchase_missing` | 1 when inter-purchase time is unavailable because the customer has one order; otherwise 0. |
| `total_orders` | Number of distinct orders in the observation window. |
| `average_order_value` | Mean order value after line-item aggregation. |
| `log_monetary_value` | `log(1 + max(monetary_value, 0))`, reducing right skew. |
| `orders_last_7d`, `orders_last_14d`, `orders_last_30d`, `orders_last_60d` | Order counts in recent subwindows inside the observation window only. |
| `orders_first_30d` | Order count in the first 30 days of the observation window. |
| `monetary_last_30d`, `monetary_first_30d` | Monetary value in the last/first 30 days of the observation window. |
| `monetary_trend_30d_vs_prior` | Last-30-day monetary value compared with the prior 30 days. |
| `frequency_trend_30d_vs_prior` | Last-30-day order count compared with the prior 30 days. |
| `days_since_first_purchase`, `days_since_last_purchase` | Explicit aliases for tenure and recency. |
| `purchase_acceleration` | Frequency trend scaled by average inter-purchase time. |
| `purchase_gap_std`, `purchase_gap_cv`, `max_purchase_gap`, `min_purchase_gap` | Variability and range of days between observed orders. |
| `active_purchase_span_days` | Days between the first and last observed purchases. |
| `unique_purchase_days` | Number of distinct purchase dates in the observation window. |
| `orders_per_active_day` | Order density over the customer's active span. |
| `avg_basket_value`, `basket_value_std`, `basket_value_cv` | Mean and variability of order-level basket value. |
| `favorite_category` | Most frequent observed order category, when a category field is available. |
| `unique_categories` | Number of distinct observed categories, when available. |
| `unique_products` | Number of distinct observed products, when available. |
| `repeat_category_rate` | Share of observed categories purchased more than once, when category is available. |
| `dominant_category_share` | Share of order lines belonging to the favorite category, when category is available. |
| `churn_30d` | 1 if no purchase occurs in the prediction window; 0 otherwise. The historical name is retained even when `--prediction-days` differs from 30. |
| `observation_start` | Inclusive observation-window start date. |
| `observation_end` | Inclusive observation-window end date. |
| `prediction_end` | Inclusive prediction-window end date. |
| `source_min_transaction_date` | Earliest valid transaction date found in the source transaction table. |
| `source_max_transaction_date` | Latest valid transaction date found in the source transaction table. |
"""
    path.write_text(content, encoding="utf-8")


def validate_output(output_path: Path) -> None:
    """Lightweight post-write smoke validation."""
    import pandas as pd

    if not output_path.is_file():
        raise RuntimeError(f"Processed modeling table was not created: {output_path}")
    output = pd.read_csv(output_path)
    required = {
        "customer_id", "recency", "frequency", "monetary_value", "tenure",
        "average_interpurchase_time", "total_orders", "average_order_value",
        "log_monetary_value", "churn_30d",
    }
    missing = sorted(required - set(output.columns))
    if missing:
        raise RuntimeError(f"Processed table is missing required columns: {missing}")
    if output.empty:
        raise RuntimeError("Processed modeling table contains no feature rows.")
    if not output["churn_30d"].isin([0, 1]).all():
        raise RuntimeError("churn_30d contains values other than 0 and 1.")
    LOGGER.info("Smoke validation passed: %d rows and all required fields exist.", len(output))


def main() -> int:
    args = parse_args()
    data_dir = args.data_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()

    if not data_dir.exists():
        LOGGER.error(
            "Data folder does not exist: %s. Download and unzip the Kaggle dataset, "
            "then pass its folder with --data-dir.",
            data_dir,
        )
        return 2
    if not data_dir.is_dir():
        LOGGER.error("--data-dir must point to a folder, not a file: %s", data_dir)
        return 2

    try:
        schemas = discover_csv_schemas(data_dir)
        mapping = resolve_dataset_mapping(schemas, data_dir)
        transactions = load_transactions(mapping)
        modeling_table = build_churn_modeling_table(
            transactions,
            observation_days=args.observation_days,
            prediction_days=args.prediction_days,
        )

        output_dir.mkdir(parents=True, exist_ok=True)
        table_path = output_dir / "churn_modeling_table.csv"
        enhanced_table_path = output_dir / "churn_modeling_table_enhanced.csv"
        dictionary_path = output_dir / "data_dictionary.md"
        modeling_table.to_csv(table_path, index=False)
        modeling_table.to_csv(enhanced_table_path, index=False)
        write_data_dictionary(
            dictionary_path,
            observation_days=args.observation_days,
            prediction_days=args.prediction_days,
        )
        validate_output(table_path)
        validate_output(enhanced_table_path)
        LOGGER.info("Wrote modeling table: %s", table_path)
        LOGGER.info("Wrote enhanced modeling table: %s", enhanced_table_path)
        LOGGER.info("Wrote data dictionary: %s", dictionary_path)
        return 0
    except (FileNotFoundError, NotADirectoryError, ValueError, RuntimeError) as exc:
        LOGGER.error("%s", exc)
        return 1


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    sys.exit(main())

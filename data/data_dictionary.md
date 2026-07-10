# Churn Modeling Table Data Dictionary

One row represents a customer with at least one order during the 90-day
observation window. The label uses the immediately following 30-day window.

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

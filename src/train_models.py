"""Command-line workflow for churn modeling, survival analysis, and retention policy."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from analysis_plots import (
    create_segments,
    generate_shap_outputs,
    plot_baseline_vs_ml_auc,
    plot_eda,
    plot_feature_importance,
    plot_model_evaluation,
    plot_survival_curves,
    plot_threshold_profit,
)
from diagnostics import create_churn_label_diagnostics
from modeling import (
    COLUMN_ALIASES,
    PreparedData,
    choose_best_classifier,
    confusion_matrix_frame,
    evaluate_probabilities,
    get_logistic_coefficients,
    get_xgboost_gain_importance,
    load_processed_data,
    predict_cox_probability,
    predict_xgboost_probability,
    prepare_modeling_data,
    save_model_artifact,
    split_train_test,
    split_train_test_temporal,
    train_single_feature_baseline,
    train_cox_model,
    train_logistic_regression,
    train_xgboost_model,
)
from retention_policy import (
    assign_retention_policy_constrained,
    assign_retention_policy,
    evaluate_thresholds_constrained,
    evaluate_thresholds,
    save_selected_threshold_constrained,
    save_selected_threshold,
    select_constrained_threshold,
    select_best_threshold,
    summarize_retention_budget,
)

LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description="Train churn models, generate survival/EDA plots, and create retention recommendations."
    )
    parser.add_argument("--input", required=True, type=Path, help="Path to data/processed/churn_modeling_table.csv")
    parser.add_argument("--coupon-cost", type=float, default=15.0, help="Cost of one high-value retention coupon.")
    parser.add_argument("--churn-cost", type=float, default=100.0, help="Estimated business cost of a churned customer.")
    parser.add_argument("--ltv-threshold", type=float, default=500.0, help="Minimum LTV for coupon eligibility.")
    parser.add_argument(
        "--shap-threshold",
        type=float,
        default=0.7,
        help="Normalized positive SHAP contribution threshold for explicit report rule.",
    )
    parser.add_argument("--test-size", type=float, default=0.25, help="Holdout test-set fraction.")
    parser.add_argument("--random-state", type=int, default=42, help="Fixed random seed.")
    parser.add_argument(
        "--split",
        choices=["random", "temporal"],
        default="random",
        help="Validation split strategy. Temporal is preferred when multiple observation cutoffs exist.",
    )
    parser.add_argument("--max-target-rate", type=float, default=0.20, help="Maximum share of customers to target.")
    parser.add_argument("--minimum-precision", type=float, default=0.10, help="Minimum precision required.")
    parser.add_argument("--budget-cap", type=float, default=10000.0, help="Maximum coupon budget on evaluation set.")
    parser.add_argument("--outputs-dir", type=Path, default=Path("outputs"), help="Directory for plots and tables.")
    parser.add_argument("--models-dir", type=Path, default=Path("models"), help="Directory for model artifacts.")
    return parser.parse_args()


def configure_logging() -> None:
    """Configure concise console logging."""

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )


def _find_alias(columns: list[str], aliases: list[str]) -> str | None:
    lowered = {column.lower(): column for column in columns}
    for alias in aliases:
        if alias.lower() in lowered:
            return lowered[alias.lower()]
    return None


def _make_output_dirs(outputs_dir: Path, models_dir: Path) -> tuple[Path, Path, Path]:
    plots_dir = outputs_dir / "plots"
    shap_dir = outputs_dir / "shap"
    tables_dir = outputs_dir / "tables"
    for directory in [outputs_dir, plots_dir, shap_dir, tables_dir, models_dir]:
        directory.mkdir(parents=True, exist_ok=True)
    return plots_dir, shap_dir, tables_dir


def _probability_from_model(model: object, x: pd.DataFrame) -> np.ndarray:
    return model.predict_proba(x)[:, 1]  # type: ignore[attr-defined]


def _write_run_summary(
    tables_dir: Path,
    *,
    best_model_name: str,
    selected_threshold: float,
    constrained_threshold: float,
    prepared: PreparedData,
    row_count: int,
    split: str,
) -> None:
    summary = pd.DataFrame(
        [
            {"item": "best_classifier", "value": best_model_name},
            {"item": "selected_threshold", "value": selected_threshold},
            {"item": "constrained_selected_threshold", "value": constrained_threshold},
            {"item": "input_rows", "value": row_count},
            {"item": "validation_split", "value": split},
            {"item": "target_column", "value": prepared.target_column},
            {"item": "customer_id_column", "value": prepared.customer_id_column},
            {"item": "ltv_column", "value": prepared.ltv_column},
            {"item": "duration_column", "value": prepared.duration_column or ""},
            {"item": "feature_count", "value": len(prepared.feature_columns)},
        ]
    )
    summary.to_csv(tables_dir / "run_summary.csv", index=False)


def run_workflow(args: argparse.Namespace) -> None:
    """Run the complete modeling and retention workflow."""

    plots_dir, shap_dir, tables_dir = _make_output_dirs(args.outputs_dir, args.models_dir)

    frame = load_processed_data(args.input)
    prepared = prepare_modeling_data(frame)

    frequency_column = _find_alias(prepared.frame.columns.tolist(), COLUMN_ALIASES["frequency"])
    recency_column = _find_alias(prepared.frame.columns.tolist(), COLUMN_ALIASES["recency"])
    segmented_frame = create_segments(prepared.frame, prepared.ltv_column, frequency_column)
    create_churn_label_diagnostics(
        prepared.frame,
        target_column=prepared.target_column,
        customer_id_column=prepared.customer_id_column,
        tables_dir=tables_dir,
        plots_dir=plots_dir,
    )

    plot_survival_curves(
        segmented_frame,
        duration_column=prepared.duration_column,
        event_column=prepared.target_column,
        plots_dir=plots_dir,
    )
    plot_eda(
        segmented_frame,
        target_column=prepared.target_column,
        monetary_column=prepared.ltv_column,
        recency_column=recency_column,
        numeric_features=prepared.numeric_features,
        plots_dir=plots_dir,
    )

    if args.split == "temporal":
        x_train, x_test, y_train, y_test, split_warning = split_train_test_temporal(
            prepared,
            test_size=args.test_size,
            random_state=args.random_state,
            warning_path=tables_dir / "validation_warning.txt",
        )
        if split_warning:
            LOGGER.warning(split_warning)
    else:
        x_train, x_test, y_train, y_test = split_train_test(
            prepared,
            test_size=args.test_size,
            random_state=args.random_state,
        )

    LOGGER.info("Training Logistic Regression baseline.")
    logistic_model = train_logistic_regression(
        x_train,
        y_train,
        prepared.numeric_features,
        prepared.categorical_features,
        random_state=args.random_state,
    )
    save_model_artifact(logistic_model, args.models_dir / "logistic_regression.pkl")
    logistic_coefficients = get_logistic_coefficients(logistic_model)
    logistic_coefficients.to_csv(tables_dir / "logistic_coefficients.csv", index=False)
    plot_feature_importance(
        logistic_coefficients,
        feature_column="feature",
        value_column="coefficient",
        title="Logistic Regression Coefficients",
        output_path=plots_dir / "logistic_coefficients.png",
    )

    LOGGER.info("Training XGBoost classifier.")
    xgb_artifact = train_xgboost_model(
        x_train,
        y_train,
        prepared.numeric_features,
        prepared.categorical_features,
        random_state=args.random_state,
    )
    save_model_artifact(xgb_artifact, args.models_dir / "xgboost_churn.pkl")
    xgb_importance = get_xgboost_gain_importance(xgb_artifact)
    xgb_importance.to_csv(tables_dir / "xgboost_feature_importance.csv", index=False)
    plot_feature_importance(
        xgb_importance,
        feature_column="feature",
        value_column="gain",
        title="XGBoost Feature Importance by Gain",
        output_path=plots_dir / "xgboost_feature_importance_gain.png",
    )
    if xgb_artifact.get("calibration_warning"):
        (tables_dir / "xgboost_calibration_warning.txt").write_text(
            str(xgb_artifact["calibration_warning"]) + "\n", encoding="utf-8"
        )

    LOGGER.info("Training Cox Proportional Hazards model.")
    cox_model, _cox_warning = train_cox_model(
        prepared,
        output_tables_dir=tables_dir,
        model_path=args.models_dir / "cox_ph_model.pkl",
    )

    probability_by_model: dict[str, np.ndarray] = {
        "Logistic Regression": _probability_from_model(logistic_model, x_test),
        "XGBoost": predict_xgboost_probability(xgb_artifact, x_test),
    }
    if cox_model is not None:
        cox_probability = predict_cox_probability(cox_model, prepared, x_test)
        if cox_probability is not None:
            probability_by_model["Cox PH"] = cox_probability

    baseline_aliases = {
        "recency_only_model": COLUMN_ALIASES["recency"],
        "frequency_only_model": COLUMN_ALIASES["frequency"],
        "monetary_only_model": COLUMN_ALIASES["monetary_value"],
    }
    baseline_probability_by_model: dict[str, np.ndarray] = {}
    for model_name, aliases in baseline_aliases.items():
        feature = _find_alias(prepared.feature_columns, aliases)
        if feature and feature in prepared.numeric_features:
            baseline_model = train_single_feature_baseline(
                x_train,
                y_train,
                feature,
                random_state=args.random_state,
            )
            baseline_probability_by_model[model_name] = baseline_model.predict_proba(x_test[[feature]])[:, 1]
        else:
            LOGGER.warning("Skipping %s because no numeric feature matching %s was found.", model_name, aliases)

    preliminary_metrics = pd.DataFrame(
        [
            evaluate_probabilities(y_test, probabilities, threshold=0.5, model_name=model_name)
            for model_name, probabilities in probability_by_model.items()
        ]
    )
    best_model_name = choose_best_classifier(preliminary_metrics)
    best_probability_for_test = probability_by_model[best_model_name]

    threshold_frame = evaluate_thresholds(
        y_test,
        best_probability_for_test,
        coupon_cost=args.coupon_cost,
        churn_cost=args.churn_cost,
    )
    selected_threshold = select_best_threshold(threshold_frame)
    threshold_frame.to_csv(tables_dir / "threshold_analysis.csv", index=False)
    save_selected_threshold(
        tables_dir / "selected_threshold.json",
        threshold=selected_threshold,
        coupon_cost=args.coupon_cost,
        churn_cost=args.churn_cost,
    )
    plot_threshold_profit(threshold_frame, selected_threshold, plots_dir)

    constrained_threshold_frame = evaluate_thresholds_constrained(
        y_test,
        best_probability_for_test,
        coupon_cost=args.coupon_cost,
        churn_cost=args.churn_cost,
        max_target_rate=args.max_target_rate,
        minimum_precision=args.minimum_precision,
        budget_cap=args.budget_cap,
    )
    constrained_threshold, constrained_warning = select_constrained_threshold(constrained_threshold_frame)
    constrained_threshold_frame.to_csv(tables_dir / "threshold_analysis_constrained.csv", index=False)
    save_selected_threshold_constrained(
        tables_dir / "selected_threshold_constrained.json",
        threshold=constrained_threshold,
        coupon_cost=args.coupon_cost,
        churn_cost=args.churn_cost,
        max_target_rate=args.max_target_rate,
        minimum_precision=args.minimum_precision,
        budget_cap=args.budget_cap,
        warning=constrained_warning,
    )
    if constrained_warning:
        (tables_dir / "threshold_constraint_warning.txt").write_text(
            constrained_warning + "\n", encoding="utf-8"
        )
        LOGGER.warning(constrained_warning)
    plot_threshold_profit(
        constrained_threshold_frame,
        constrained_threshold,
        plots_dir,
        filename="threshold_profit_curve_constrained.png",
        title="Constrained Expected Profit by Churn Targeting Threshold",
    )

    metrics_frame = pd.DataFrame(
        [
            evaluate_probabilities(
                y_test,
                probabilities,
                threshold=selected_threshold,
                model_name=model_name,
            )
            for model_name, probabilities in probability_by_model.items()
        ]
    )
    metrics_frame.to_csv(tables_dir / "model_metrics.csv", index=False)

    baseline_metrics = pd.DataFrame(
        [
            evaluate_probabilities(
                y_test,
                probabilities,
                threshold=constrained_threshold,
                model_name=model_name,
            )
            for model_name, probabilities in {
                **baseline_probability_by_model,
                "Logistic Regression": probability_by_model["Logistic Regression"],
                "XGBoost": probability_by_model["XGBoost"],
            }.items()
        ]
    )
    baseline_metrics.to_csv(tables_dir / "baseline_model_metrics.csv", index=False)
    plot_baseline_vs_ml_auc(baseline_metrics, plots_dir)

    confusion_matrix_frame(y_test, best_probability_for_test, selected_threshold).to_csv(
        tables_dir / "confusion_matrix_selected_threshold.csv"
    )
    plot_model_evaluation(
        y_test,
        probability_by_model,
        selected_threshold=selected_threshold,
        best_model_name=best_model_name,
        plots_dir=plots_dir,
    )

    x_all = prepared.frame[prepared.feature_columns].copy()
    if best_model_name == "XGBoost":
        all_probabilities = predict_xgboost_probability(xgb_artifact, x_all)
    else:
        all_probabilities = _probability_from_model(logistic_model, x_all)

    xgb_all_probabilities = predict_xgboost_probability(xgb_artifact, x_all)
    shap_scores = generate_shap_outputs(
        xgb_artifact,
        x_all,
        prepared.frame[prepared.customer_id_column],
        xgb_all_probabilities,
        plots_dir=plots_dir,
        shap_dir=shap_dir,
        tables_dir=tables_dir,
    )

    scoring_frame = prepared.frame[[prepared.customer_id_column, prepared.ltv_column]].copy()
    scoring_frame["predicted_churn_probability"] = all_probabilities
    shap_scores_for_merge = shap_scores.rename(columns={"customer_id": prepared.customer_id_column})
    scoring_frame = scoring_frame.merge(shap_scores_for_merge, on=prepared.customer_id_column, how="left")

    scored_customers = assign_retention_policy(
        scoring_frame,
        customer_id_column=prepared.customer_id_column,
        probability_column="predicted_churn_probability",
        ltv_column=prepared.ltv_column,
        selected_threshold=selected_threshold,
        ltv_threshold=args.ltv_threshold,
        shap_threshold=args.shap_threshold,
    )
    scored_customers.to_csv(tables_dir / "scored_customers.csv", index=False)
    summarize_retention_budget(scored_customers).to_csv(
        tables_dir / "retention_budget_summary.csv", index=False
    )

    scored_customers_constrained = assign_retention_policy_constrained(
        scoring_frame,
        customer_id_column=prepared.customer_id_column,
        probability_column="predicted_churn_probability",
        ltv_column=prepared.ltv_column,
        selected_threshold=constrained_threshold,
        ltv_threshold=args.ltv_threshold,
        budget_cap=args.budget_cap,
    )
    scored_customers_constrained.to_csv(tables_dir / "scored_customers_constrained.csv", index=False)
    summarize_retention_budget(scored_customers_constrained).to_csv(
        tables_dir / "retention_budget_summary_constrained.csv", index=False
    )

    _write_run_summary(
        tables_dir,
        best_model_name=best_model_name,
        selected_threshold=selected_threshold,
        constrained_threshold=constrained_threshold,
        prepared=prepared,
        row_count=len(prepared.frame),
        split=args.split,
    )

    LOGGER.info("Workflow complete.")
    LOGGER.info("Best classifier: %s", best_model_name)
    LOGGER.info("Selected threshold: %.2f", selected_threshold)
    LOGGER.info("Constrained selected threshold: %.2f", constrained_threshold)
    LOGGER.info("Outputs saved under %s and %s.", args.outputs_dir, args.models_dir)


def main() -> None:
    """CLI entry point."""

    configure_logging()
    args = parse_args()
    try:
        run_workflow(args)
    except Exception as exc:
        LOGGER.error("Training workflow failed: %s", exc)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()

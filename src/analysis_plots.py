"""Plotting utilities for the churn modeling workflow."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.calibration import calibration_curve
from sklearn.metrics import auc, precision_recall_curve, roc_curve

LOGGER = logging.getLogger(__name__)


def set_plot_style() -> None:
    """Apply a clean, consistent visual style for all saved plots."""

    sns.set_theme(
        context="notebook",
        style="whitegrid",
        palette="deep",
        rc={
            "figure.figsize": (9, 6),
            "axes.titlesize": 15,
            "axes.labelsize": 12,
            "legend.fontsize": 10,
            "font.family": "DejaVu Sans",
        },
    )


def save_current_figure(path: Path) -> None:
    """Save and close the current matplotlib figure."""

    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=180, bbox_inches="tight")
    plt.close()
    LOGGER.info("Saved plot to %s", path)


def _segment_by_quantiles(series: pd.Series, labels: list[str]) -> pd.Series:
    """Create quantile-based segments with fallbacks for low-cardinality data."""

    numeric = pd.to_numeric(series, errors="coerce").fillna(series.median() if series.notna().any() else 0)
    try:
        return pd.qcut(numeric.rank(method="first"), q=len(labels), labels=labels)
    except ValueError:
        return pd.Series(labels[0], index=series.index, dtype="object")


def create_segments(frame: pd.DataFrame, monetary_column: str, frequency_column: str | None) -> pd.DataFrame:
    """Add value and frequency segment columns used by EDA/survival plots."""

    segmented = frame.copy()
    segmented["value_segment"] = _segment_by_quantiles(
        pd.to_numeric(segmented[monetary_column], errors="coerce"),
        ["Low value", "Medium value", "High value"],
    )
    if frequency_column and frequency_column in segmented.columns:
        segmented["frequency_segment"] = _segment_by_quantiles(
            pd.to_numeric(segmented[frequency_column], errors="coerce"),
            ["Low frequency", "Medium frequency", "High frequency"],
        )
    else:
        segmented["frequency_segment"] = "Unknown frequency"
    return segmented


def plot_eda(
    frame: pd.DataFrame,
    *,
    target_column: str,
    monetary_column: str,
    recency_column: str | None,
    numeric_features: list[str],
    plots_dir: Path,
) -> None:
    """Create portfolio-ready exploratory plots from the processed table."""

    set_plot_style()

    if "value_segment" in frame.columns and target_column in frame.columns:
        plt.figure(figsize=(9, 5))
        segment_order = ["Low value", "Medium value", "High value"]
        churn_by_segment = (
            frame.groupby("value_segment", observed=True)[target_column]
            .mean()
            .reindex(segment_order)
            .reset_index()
        )
        ax = sns.barplot(data=churn_by_segment, x="value_segment", y=target_column, order=segment_order)
        ax.set_title("Churn Rate by Customer Value Segment")
        ax.set_xlabel("Customer value segment")
        ax.set_ylabel("30-day churn rate")
        ax.yaxis.set_major_formatter(lambda value, _: f"{value:.0%}")
        save_current_figure(plots_dir / "churn_rate_by_segment.png")

    features_to_plot = [column for column in numeric_features if column in frame.columns][:8]
    if features_to_plot:
        n_cols = 2
        n_rows = int(np.ceil(len(features_to_plot) / n_cols))
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, max(4, 3.2 * n_rows)))
        axes_array = np.asarray(axes).reshape(-1)
        for axis, column in zip(axes_array, features_to_plot):
            sns.histplot(data=frame, x=column, hue=target_column, bins=30, kde=False, ax=axis, element="step")
            axis.set_title(column.replace("_", " ").title())
            axis.set_xlabel(column.replace("_", " ").title())
            axis.set_ylabel("Customers")
        for axis in axes_array[len(features_to_plot) :]:
            axis.axis("off")
        fig.suptitle("Feature Distributions by Churn Outcome", y=1.01)
        save_current_figure(plots_dir / "feature_distributions.png")

    corr_features = [column for column in numeric_features if column in frame.columns]
    if target_column in frame.columns:
        corr_features = [*corr_features, target_column]
    if len(corr_features) >= 2:
        plt.figure(figsize=(10, 8))
        corr = frame[corr_features].corr(numeric_only=True)
        sns.heatmap(corr, cmap="vlag", center=0, annot=False, linewidths=0.4)
        plt.title("Correlation Heatmap of Churn Features")
        save_current_figure(plots_dir / "correlation_heatmap.png")

    if recency_column and recency_column in frame.columns and monetary_column in frame.columns:
        sample = frame.sample(min(len(frame), 5000), random_state=42)
        plt.figure(figsize=(9, 6))
        ax = sns.scatterplot(
            data=sample,
            x=recency_column,
            y=monetary_column,
            hue=target_column,
            alpha=0.55,
            edgecolor=None,
        )
        ax.set_yscale("symlog")
        ax.set_title("Recency vs. Monetary Value")
        ax.set_xlabel("Recency")
        ax.set_ylabel("Monetary value / LTV proxy")
        save_current_figure(plots_dir / "recency_vs_monetary_scatter.png")


def plot_survival_curves(
    frame: pd.DataFrame,
    *,
    duration_column: str | None,
    event_column: str,
    plots_dir: Path,
) -> None:
    """Create Kaplan-Meier survival curves overall and by customer segment."""

    try:
        from lifelines import KaplanMeierFitter
    except ImportError:  # pragma: no cover - missing environment
        warning_path = plots_dir / "kaplan_meier_warning.txt"
        warning_path.parent.mkdir(parents=True, exist_ok=True)
        warning_path.write_text(
            "Kaplan-Meier plots skipped because lifelines is not installed.\n",
            encoding="utf-8",
        )
        LOGGER.warning("Kaplan-Meier plots skipped because lifelines is not installed.")
        return

    if duration_column is None or duration_column not in frame.columns:
        LOGGER.warning("Kaplan-Meier plots skipped because no duration column was available.")
        return

    survival_frame = frame[[duration_column, event_column, "value_segment", "frequency_segment"]].copy()
    survival_frame[duration_column] = pd.to_numeric(survival_frame[duration_column], errors="coerce")
    survival_frame[event_column] = pd.to_numeric(survival_frame[event_column], errors="coerce")
    survival_frame = survival_frame.dropna(subset=[duration_column, event_column])
    survival_frame = survival_frame[survival_frame[duration_column] > 0]
    if survival_frame.empty or survival_frame[event_column].nunique() < 2:
        LOGGER.warning("Kaplan-Meier plots skipped because survival data lacks events/censoring variation.")
        return

    set_plot_style()
    kmf = KaplanMeierFitter()

    plt.figure(figsize=(9, 6))
    kmf.fit(survival_frame[duration_column], event_observed=survival_frame[event_column], label="All customers")
    kmf.plot_survival_function(ci_show=True)
    plt.title("Kaplan-Meier Survival Curve: Overall")
    plt.xlabel("Duration proxy (days)")
    plt.ylabel("Estimated retention survival probability")
    save_current_figure(plots_dir / "kaplan_meier_overall.png")

    _plot_grouped_km(
        survival_frame,
        duration_column=duration_column,
        event_column=event_column,
        group_column="value_segment",
        title="Kaplan-Meier Survival by Customer Value Segment",
        output_path=plots_dir / "kaplan_meier_by_value_segment.png",
    )
    _plot_grouped_km(
        survival_frame,
        duration_column=duration_column,
        event_column=event_column,
        group_column="frequency_segment",
        title="Kaplan-Meier Survival by Purchase Frequency Segment",
        output_path=plots_dir / "kaplan_meier_by_frequency_segment.png",
    )


def _plot_grouped_km(
    frame: pd.DataFrame,
    *,
    duration_column: str,
    event_column: str,
    group_column: str,
    title: str,
    output_path: Path,
) -> None:
    from lifelines import KaplanMeierFitter

    kmf = KaplanMeierFitter()
    plt.figure(figsize=(9, 6))
    for group_name, group_frame in frame.groupby(group_column, observed=True):
        if len(group_frame) < 2:
            continue
        kmf.fit(group_frame[duration_column], event_observed=group_frame[event_column], label=str(group_name))
        kmf.plot_survival_function(ci_show=False)
    plt.title(title)
    plt.xlabel("Duration proxy (days)")
    plt.ylabel("Estimated retention survival probability")
    plt.legend(title=group_column.replace("_", " ").title())
    save_current_figure(output_path)


def plot_model_evaluation(
    y_true: pd.Series,
    probability_by_model: dict[str, np.ndarray],
    *,
    selected_threshold: float,
    best_model_name: str,
    plots_dir: Path,
) -> None:
    """Create ROC, PR, calibration, Brier, confusion, and probability plots."""

    set_plot_style()
    y_array = np.asarray(y_true).astype(int)

    plt.figure(figsize=(8, 6))
    for model_name, probabilities in probability_by_model.items():
        if len(np.unique(y_array)) < 2:
            continue
        fpr, tpr, _ = roc_curve(y_array, probabilities)
        plt.plot(fpr, tpr, label=f"{model_name} (AUC={auc(fpr, tpr):.3f})")
    plt.plot([0, 1], [0, 1], "--", color="gray", label="Random")
    plt.title("ROC Curve")
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.legend()
    save_current_figure(plots_dir / "roc_curve.png")

    plt.figure(figsize=(8, 6))
    for model_name, probabilities in probability_by_model.items():
        if len(np.unique(y_array)) < 2:
            continue
        precision, recall, _ = precision_recall_curve(y_array, probabilities)
        plt.plot(recall, precision, label=f"{model_name}")
    plt.title("Precision-Recall Curve")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.legend()
    save_current_figure(plots_dir / "precision_recall_curve.png")

    plt.figure(figsize=(8, 6))
    for model_name, probabilities in probability_by_model.items():
        fraction_positive, mean_predicted = calibration_curve(
            y_array, probabilities, n_bins=min(10, max(2, len(y_array) // 20)), strategy="quantile"
        )
        plt.plot(mean_predicted, fraction_positive, marker="o", label=model_name)
    plt.plot([0, 1], [0, 1], "--", color="gray", label="Perfect calibration")
    plt.title("Calibration Curves")
    plt.xlabel("Mean predicted churn probability")
    plt.ylabel("Observed churn rate")
    plt.legend()
    save_current_figure(plots_dir / "calibration_curves.png")

    brier_values = [
        {"model": model_name, "brier_score": np.mean((np.asarray(probabilities) - y_array) ** 2)}
        for model_name, probabilities in probability_by_model.items()
    ]
    plt.figure(figsize=(8, 5))
    ax = sns.barplot(data=pd.DataFrame(brier_values), x="model", y="brier_score")
    ax.set_title("Brier Score Comparison")
    ax.set_xlabel("Model")
    ax.set_ylabel("Brier score (lower is better)")
    plt.xticks(rotation=20, ha="right")
    save_current_figure(plots_dir / "brier_score_comparison.png")

    best_probability = probability_by_model[best_model_name]
    predictions = (best_probability >= selected_threshold).astype(int)
    matrix = pd.crosstab(
        pd.Series(y_array, name="Actual"),
        pd.Series(predictions, name="Predicted"),
        dropna=False,
    ).reindex(index=[0, 1], columns=[0, 1], fill_value=0)
    plt.figure(figsize=(6, 5))
    sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues", cbar=False)
    plt.title(f"Confusion Matrix at Threshold {selected_threshold:.2f}\nBest model: {best_model_name}")
    plt.xlabel("Predicted churn")
    plt.ylabel("Actual churn")
    save_current_figure(plots_dir / "confusion_matrix_selected_threshold.png")

    probability_frame = pd.DataFrame(
        {
            "predicted_churn_probability": np.concatenate(list(probability_by_model.values())),
            "model": np.repeat(list(probability_by_model.keys()), [len(v) for v in probability_by_model.values()]),
        }
    )
    plt.figure(figsize=(9, 6))
    sns.histplot(
        data=probability_frame,
        x="predicted_churn_probability",
        hue="model",
        bins=30,
        element="step",
        stat="density",
        common_norm=False,
    )
    plt.axvline(selected_threshold, color="black", linestyle="--", label=f"Selected threshold={selected_threshold:.2f}")
    plt.title("Model Probability Distributions")
    plt.xlabel("Predicted churn probability")
    plt.ylabel("Density")
    plt.legend()
    save_current_figure(plots_dir / "model_probability_distributions.png")


def plot_threshold_profit(
    threshold_frame: pd.DataFrame,
    selected_threshold: float,
    plots_dir: Path,
    *,
    filename: str = "threshold_profit_curve.png",
    title: str = "Expected Profit by Churn Targeting Threshold",
) -> None:
    """Plot expected profit by decision threshold."""

    set_plot_style()
    plt.figure(figsize=(9, 5))
    sns.lineplot(data=threshold_frame, x="threshold", y="total_profit")
    if "constraints_satisfied" in threshold_frame.columns:
        feasible = threshold_frame[threshold_frame["constraints_satisfied"] == True]  # noqa: E712
        if not feasible.empty:
            plt.scatter(feasible["threshold"], feasible["total_profit"], s=18, alpha=0.7, label="Feasible")
    plt.axvline(selected_threshold, color="black", linestyle="--", label=f"Selected={selected_threshold:.2f}")
    plt.title(title)
    plt.xlabel("Targeting threshold")
    plt.ylabel("Expected profit on evaluation set")
    plt.legend()
    save_current_figure(plots_dir / filename)


def plot_baseline_vs_ml_auc(metrics_frame: pd.DataFrame, plots_dir: Path) -> None:
    """Plot ROC-AUC for simple RFM baselines versus ML classifiers."""

    set_plot_style()
    if "roc_auc" not in metrics_frame.columns:
        return
    plot_frame = metrics_frame.dropna(subset=["roc_auc"]).copy()
    if plot_frame.empty:
        return
    plt.figure(figsize=(9, 5))
    ax = sns.barplot(data=plot_frame, x="model", y="roc_auc")
    ax.axhline(0.5, color="gray", linestyle="--", label="Random")
    ax.set_title("Simple RFM Baselines vs. ML Models")
    ax.set_xlabel("Model")
    ax.set_ylabel("ROC-AUC")
    ax.set_ylim(0, 1)
    plt.xticks(rotation=25, ha="right")
    plt.legend()
    save_current_figure(plots_dir / "baseline_vs_ml_auc.png")


def plot_feature_importance(
    importance_frame: pd.DataFrame,
    *,
    feature_column: str,
    value_column: str,
    title: str,
    output_path: Path,
    top_n: int = 20,
) -> None:
    """Plot top feature importances or coefficients."""

    set_plot_style()
    if importance_frame.empty:
        return
    plot_frame = importance_frame.copy()
    plot_frame["abs_value"] = plot_frame[value_column].abs()
    plot_frame = plot_frame.sort_values("abs_value", ascending=False).head(top_n)
    plt.figure(figsize=(9, max(5, 0.35 * len(plot_frame))))
    sns.barplot(data=plot_frame, y=feature_column, x=value_column, orient="h")
    plt.title(title)
    plt.xlabel(value_column.replace("_", " ").title())
    plt.ylabel("Feature")
    save_current_figure(output_path)


def generate_shap_outputs(
    xgb_artifact: dict[str, Any],
    x_all: pd.DataFrame,
    customer_ids: pd.Series,
    probabilities: np.ndarray,
    *,
    plots_dir: Path,
    shap_dir: Path,
    tables_dir: Path,
    max_summary_rows: int = 2000,
) -> pd.DataFrame:
    """Generate SHAP summary plots, five local explanations, and driver table.

    The returned table contains one normalized positive-contribution score per
    customer. The retention policy interprets that score as the "SHAP score"
    requested by the project: a 0-1 normalized measure of how strongly SHAP
    contributions push a customer toward churn relative to other customers.
    """

    try:
        import shap
    except ImportError as exc:  # pragma: no cover - missing environment
        raise ImportError(
            "SHAP explanations require `shap`. Install dependencies with `pip install -r requirements.txt`."
        ) from exc

    set_plot_style()
    encoded_all = xgb_artifact["preprocessor"].transform(x_all)
    feature_names = list(xgb_artifact["feature_names"])
    encoded_frame = pd.DataFrame(encoded_all, columns=feature_names, index=x_all.index)

    explainer = shap.TreeExplainer(xgb_artifact["shap_model"])
    shap_values = explainer.shap_values(encoded_frame)
    shap_array = np.asarray(shap_values)
    if shap_array.ndim == 3:
        shap_array = shap_array[:, :, 1]

    summary_sample = encoded_frame
    summary_shap = shap_array
    if len(encoded_frame) > max_summary_rows:
        summary_sample = encoded_frame.sample(max_summary_rows, random_state=42)
        summary_shap = shap_array[encoded_frame.index.get_indexer(summary_sample.index)]

    plt.figure()
    shap.summary_plot(summary_shap, summary_sample, plot_type="bar", show=False, max_display=15)
    plt.title("Top SHAP Drivers of Churn Risk")
    save_current_figure(plots_dir / "shap_summary_bar.png")

    plt.figure()
    shap.summary_plot(summary_shap, summary_sample, show=False, max_display=15)
    save_current_figure(plots_dir / "shap_summary_beeswarm.png")

    positive_contribution = np.maximum(shap_array, 0).sum(axis=1)
    min_contribution = float(np.nanmin(positive_contribution))
    max_contribution = float(np.nanmax(positive_contribution))
    if max_contribution > min_contribution:
        shap_score = (positive_contribution - min_contribution) / (max_contribution - min_contribution)
    else:
        shap_score = np.zeros_like(positive_contribution)

    local_rows: list[dict[str, Any]] = []
    top_indices = np.argsort(probabilities)[-5:][::-1]
    explanation = shap.Explanation(
        values=shap_array,
        base_values=np.repeat(explainer.expected_value, len(encoded_frame))
        if np.isscalar(explainer.expected_value)
        else np.repeat(np.asarray(explainer.expected_value).ravel()[0], len(encoded_frame)),
        data=encoded_frame.to_numpy(),
        feature_names=feature_names,
    )

    for rank, row_position in enumerate(top_indices, start=1):
        shap.waterfall_plot(explanation[row_position], max_display=12, show=False)
        plt.title(f"Customer {rank} SHAP Explanation")
        save_current_figure(shap_dir / f"customer_{rank}_shap_waterfall.png")

        row_values = shap_array[row_position]
        positive_features = [
            feature_names[index]
            for index in np.argsort(row_values)[::-1]
            if row_values[index] > 0
        ][:3]
        negative_features = [
            feature_names[index]
            for index in np.argsort(row_values)
            if row_values[index] < 0
        ][:3]
        local_rows.append(
            {
                "customer_id": customer_ids.iloc[row_position],
                "predicted_churn_probability": probabilities[row_position],
                "top_positive_driver_1": positive_features[0] if len(positive_features) > 0 else "",
                "top_positive_driver_2": positive_features[1] if len(positive_features) > 1 else "",
                "top_positive_driver_3": positive_features[2] if len(positive_features) > 2 else "",
                "top_negative_driver_1": negative_features[0] if len(negative_features) > 0 else "",
                "top_negative_driver_2": negative_features[1] if len(negative_features) > 1 else "",
                "top_negative_driver_3": negative_features[2] if len(negative_features) > 2 else "",
            }
        )

    local_table = pd.DataFrame(local_rows)
    local_table.to_csv(tables_dir / "top_5_customer_shap_explanations.csv", index=False)

    return pd.DataFrame(
        {
            "customer_id": customer_ids.to_numpy(),
            "shap_score": shap_score,
            "shap_positive_contribution": positive_contribution,
        }
    )

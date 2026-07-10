"""Model preparation, training, and evaluation utilities for churn analysis.

The functions in this module intentionally avoid assumptions about one exact
processed-schema version. They resolve common column aliases, exclude leakage
columns, and return plain dictionaries/data frames that the command-line runner
can save as reproducible artifacts.
"""

from __future__ import annotations

import contextlib
import io
import logging
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

LOGGER = logging.getLogger(__name__)

TARGET_COLUMN = "churn_30d"

COLUMN_ALIASES: dict[str, list[str]] = {
    "customer_id": ["customer_id", "customerid", "customer_key", "customer"],
    "target": ["churn_30d", "churn", "is_churned", "label", "target"],
    "recency": ["recency_days", "recency"],
    "frequency": ["frequency", "purchase_frequency"],
    "monetary_value": ["monetary_value", "monetary", "ltv", "customer_ltv", "revenue"],
    "tenure": ["tenure_days", "tenure", "customer_tenure_days"],
    "avg_interpurchase": [
        "avg_interpurchase_days",
        "average_interpurchase_time",
        "average_interpurchase_days",
    ],
    "total_orders": ["total_orders", "order_count", "orders"],
    "avg_order_value": ["avg_order_value", "average_order_value", "aov"],
    "log_monetary_value": ["log_monetary_value", "log_ltv", "log_revenue"],
}

# Columns with these tokens are never used as classifier features. They either
# encode the outcome/future window or are identifiers/date boundaries.
LEAKAGE_TOKENS = (
    "churn",
    "label",
    "target",
    "future",
    "prediction",
    "event",
    "outcome",
    "post_",
)
NON_FEATURE_TOKENS = ("customer_id", "date", "_start", "_end")


@dataclass(frozen=True)
class PreparedData:
    """Container for resolved data needed by the modeling workflow."""

    frame: pd.DataFrame
    customer_id_column: str
    target_column: str
    ltv_column: str
    feature_columns: list[str]
    numeric_features: list[str]
    categorical_features: list[str]
    duration_column: str | None
    observation_end_column: str | None


def _find_first_column(columns: list[str], aliases: list[str]) -> str | None:
    lowered = {column.lower(): column for column in columns}
    for alias in aliases:
        if alias.lower() in lowered:
            return lowered[alias.lower()]
    return None


def load_processed_data(input_path: Path) -> pd.DataFrame:
    """Load the processed churn modeling table with clear validation errors."""

    if not input_path.exists():
        raise FileNotFoundError(
            f"Processed dataset not found: {input_path}. "
            "Run `python src/run_pipeline.py --data-dir <raw_data_folder>` first."
        )

    frame = pd.read_csv(input_path)
    if frame.empty:
        raise ValueError(f"Processed dataset is empty: {input_path}")

    frame.columns = [str(column).strip() for column in frame.columns]
    LOGGER.info("Loaded processed dataset with %s rows and %s columns.", len(frame), len(frame.columns))
    return frame


def prepare_modeling_data(frame: pd.DataFrame) -> PreparedData:
    """Resolve schema, validate target/customer columns, and select features."""

    columns = frame.columns.tolist()
    customer_id_column = _find_first_column(columns, COLUMN_ALIASES["customer_id"])
    target_column = _find_first_column(columns, COLUMN_ALIASES["target"])
    ltv_column = _find_first_column(columns, ["ltv", "customer_ltv", *COLUMN_ALIASES["monetary_value"]])
    duration_column = _find_first_column(columns, COLUMN_ALIASES["tenure"])
    observation_end_column = _find_first_column(columns, ["observation_end", "snapshot_date", "as_of_date"])

    missing: list[str] = []
    if customer_id_column is None:
        missing.append("customer identifier, e.g. customer_id")
    if target_column is None:
        missing.append("binary churn target, e.g. churn_30d")
    if ltv_column is None:
        missing.append("monetary/LTV column, e.g. monetary_value")

    if missing:
        raise ValueError(
            "Missing required processed columns: "
            + "; ".join(missing)
            + f". Available columns: {columns}"
        )

    frame = frame.copy()
    frame[target_column] = pd.to_numeric(frame[target_column], errors="coerce")
    frame = frame.dropna(subset=[target_column])
    frame[target_column] = frame[target_column].astype(int)

    target_values = set(frame[target_column].unique().tolist())
    if not target_values.issubset({0, 1}):
        raise ValueError(
            f"Target column `{target_column}` must be binary 0/1. "
            f"Found values: {sorted(target_values)}"
        )

    feature_columns: list[str] = []
    for column in frame.columns:
        lower = column.lower()
        if column in {customer_id_column, target_column}:
            continue
        if any(token in lower for token in LEAKAGE_TOKENS):
            continue
        if any(token in lower for token in NON_FEATURE_TOKENS):
            continue
        feature_columns.append(column)

    if not feature_columns:
        raise ValueError(
            "No eligible feature columns remain after leakage/id/date exclusion. "
            f"Available columns: {columns}"
        )

    numeric_features = [
        column for column in feature_columns if pd.api.types.is_numeric_dtype(frame[column])
    ]
    categorical_features = [column for column in feature_columns if column not in numeric_features]

    LOGGER.info("Using %s numeric and %s categorical features.", len(numeric_features), len(categorical_features))
    LOGGER.info("Feature columns: %s", feature_columns)

    return PreparedData(
        frame=frame,
        customer_id_column=customer_id_column,
        target_column=target_column,
        ltv_column=ltv_column,
        feature_columns=feature_columns,
        numeric_features=numeric_features,
        categorical_features=categorical_features,
        duration_column=duration_column,
        observation_end_column=observation_end_column,
    )


def build_preprocessor(numeric_features: list[str], categorical_features: list[str]) -> ColumnTransformer:
    """Build preprocessing that imputes/scales numeric features and one-hot encodes categoricals."""

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    transformers: list[tuple[str, Pipeline, list[str]]] = []
    if numeric_features:
        transformers.append(("numeric", numeric_pipeline, numeric_features))
    if categorical_features:
        transformers.append(("categorical", categorical_pipeline, categorical_features))

    return ColumnTransformer(transformers=transformers, remainder="drop", verbose_feature_names_out=False)


def split_train_test(
    prepared: PreparedData,
    *,
    test_size: float,
    random_state: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Create a reproducible stratified train/test split."""

    x = prepared.frame[prepared.feature_columns].copy()
    y = prepared.frame[prepared.target_column].astype(int)

    stratify = y if y.nunique() == 2 and y.value_counts().min() >= 2 else None
    return train_test_split(x, y, test_size=test_size, random_state=random_state, stratify=stratify)


def split_train_test_temporal(
    prepared: PreparedData,
    *,
    test_size: float,
    random_state: int,
    warning_path: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, str | None]:
    """Split by observation date when possible, otherwise fall back to random."""

    if prepared.observation_end_column is None:
        warning = (
            "Temporal split requested but no observation cutoff column was found. "
            "Falling back to stratified random split."
        )
        warning_path.parent.mkdir(parents=True, exist_ok=True)
        warning_path.write_text(warning + "\n", encoding="utf-8")
        x_train, x_test, y_train, y_test = split_train_test(
            prepared, test_size=test_size, random_state=random_state
        )
        return x_train, x_test, y_train, y_test, warning

    split_dates = pd.to_datetime(prepared.frame[prepared.observation_end_column], errors="coerce")
    if split_dates.nunique(dropna=True) < 2:
        warning = (
            "Temporal split requested but the processed table contains only one "
            "observation cutoff date. Falling back to stratified random split."
        )
        warning_path.parent.mkdir(parents=True, exist_ok=True)
        warning_path.write_text(warning + "\n", encoding="utf-8")
        x_train, x_test, y_train, y_test = split_train_test(
            prepared, test_size=test_size, random_state=random_state
        )
        return x_train, x_test, y_train, y_test, warning

    cutoff = split_dates.quantile(1.0 - test_size)
    train_mask = split_dates <= cutoff
    test_mask = split_dates > cutoff
    if train_mask.sum() == 0 or test_mask.sum() == 0:
        warning = (
            "Temporal split produced an empty train or test set. Falling back to "
            "stratified random split."
        )
        warning_path.parent.mkdir(parents=True, exist_ok=True)
        warning_path.write_text(warning + "\n", encoding="utf-8")
        x_train, x_test, y_train, y_test = split_train_test(
            prepared, test_size=test_size, random_state=random_state
        )
        return x_train, x_test, y_train, y_test, warning

    x = prepared.frame[prepared.feature_columns].copy()
    y = prepared.frame[prepared.target_column].astype(int)
    return x.loc[train_mask], x.loc[test_mask], y.loc[train_mask], y.loc[test_mask], None


def train_single_feature_baseline(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    feature: str,
    *,
    random_state: int,
) -> Pipeline:
    """Train a one-feature logistic baseline for sanity checking."""

    model = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "classifier",
                LogisticRegression(max_iter=1000, solver="lbfgs", random_state=random_state),
            ),
        ]
    )
    model.fit(x_train[[feature]], y_train)
    return model


def get_logistic_coefficients(model: Pipeline) -> pd.DataFrame:
    """Return fitted Logistic Regression coefficients with transformed feature names."""

    preprocessor = model.named_steps["preprocess"]
    classifier = model.named_steps["classifier"]
    feature_names = list(preprocessor.get_feature_names_out())
    coefficients = classifier.coef_[0]
    return pd.DataFrame({"feature": feature_names, "coefficient": coefficients}).sort_values(
        "coefficient", key=lambda series: series.abs(), ascending=False
    )


def get_xgboost_gain_importance(xgb_artifact: dict[str, Any]) -> pd.DataFrame:
    """Return XGBoost gain-based feature importance."""

    booster = xgb_artifact["shap_model"].get_booster()
    scores = booster.get_score(importance_type="gain")
    feature_names = list(xgb_artifact["feature_names"])
    rows = []
    for raw_feature, gain in scores.items():
        if raw_feature.startswith("f") and raw_feature[1:].isdigit():
            index = int(raw_feature[1:])
            feature = feature_names[index] if index < len(feature_names) else raw_feature
        else:
            feature = raw_feature
        rows.append({"feature": feature, "gain": float(gain)})
    return pd.DataFrame(rows).sort_values("gain", ascending=False)


def train_logistic_regression(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    numeric_features: list[str],
    categorical_features: list[str],
    *,
    random_state: int,
) -> Pipeline:
    """Train the transparent logistic-regression baseline."""

    model = Pipeline(
        steps=[
            ("preprocess", build_preprocessor(numeric_features, categorical_features)),
            (
                "classifier",
                LogisticRegression(max_iter=2000, solver="lbfgs", random_state=random_state),
            ),
        ]
    )
    model.fit(x_train, y_train)
    return model


def train_xgboost_model(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    numeric_features: list[str],
    categorical_features: list[str],
    *,
    random_state: int,
) -> dict[str, Any]:
    """Train XGBoost and return calibrated-prediction and SHAP-ready artifacts.

    A CalibratedClassifierCV wrapper is attempted for probability quality. A
    separately fitted raw XGBoost estimator is kept for SHAP because SHAP works
    best against the tree model directly.
    """

    try:
        from sklearn.calibration import CalibratedClassifierCV
        from xgboost import XGBClassifier
    except ImportError as exc:  # pragma: no cover - exercised in missing envs
        raise ImportError(
            "XGBoost modeling requires `xgboost`. Install dependencies with "
            "`pip install -r requirements.txt`."
        ) from exc

    preprocessor = build_preprocessor(numeric_features, categorical_features)
    x_train_encoded = preprocessor.fit_transform(x_train)
    feature_names = list(preprocessor.get_feature_names_out())

    base_params = dict(
        n_estimators=300,
        max_depth=3,
        learning_rate=0.05,
        subsample=0.85,
        colsample_bytree=0.85,
        eval_metric="logloss",
        random_state=random_state,
        n_jobs=1,
    )

    shap_model = XGBClassifier(**base_params)
    shap_model.fit(x_train_encoded, y_train)

    calibrated_model: Any | None = None
    calibration_warning: str | None = None
    if y_train.nunique() == 2 and y_train.value_counts().min() >= 3:
        cv = min(3, int(y_train.value_counts().min()))
        try:
            calibration_model = XGBClassifier(**base_params)
            calibrated_model = CalibratedClassifierCV(calibration_model, method="sigmoid", cv=cv)
            calibrated_model.fit(x_train_encoded, y_train)
        except Exception as exc:  # pragma: no cover - version/data dependent
            calibration_warning = f"XGBoost calibration failed; using uncalibrated probabilities. Reason: {exc}"
            LOGGER.warning(calibration_warning)
            calibrated_model = None
    else:
        calibration_warning = "XGBoost calibration skipped because a class has fewer than 3 training rows."
        LOGGER.warning(calibration_warning)

    return {
        "preprocessor": preprocessor,
        "classifier": calibrated_model if calibrated_model is not None else shap_model,
        "shap_model": shap_model,
        "feature_names": feature_names,
        "calibration_warning": calibration_warning,
    }


def predict_xgboost_probability(xgb_artifact: dict[str, Any], x: pd.DataFrame) -> np.ndarray:
    """Predict churn probability from the saved XGBoost artifact."""

    encoded = xgb_artifact["preprocessor"].transform(x)
    return xgb_artifact["classifier"].predict_proba(encoded)[:, 1]


def evaluate_probabilities(
    y_true: pd.Series | np.ndarray,
    y_probability: np.ndarray,
    *,
    threshold: float,
    model_name: str,
) -> dict[str, float | int | str]:
    """Compute classification and probability-quality metrics."""

    y_true_array = np.asarray(y_true).astype(int)
    probabilities = np.clip(np.asarray(y_probability, dtype=float), 1e-6, 1 - 1e-6)
    predictions = (probabilities >= threshold).astype(int)

    if len(np.unique(y_true_array)) < 2:
        roc_auc = np.nan
        pr_auc = np.nan
    else:
        roc_auc = roc_auc_score(y_true_array, probabilities)
        pr_auc = average_precision_score(y_true_array, probabilities)

    return {
        "model": model_name,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "brier_score": brier_score_loss(y_true_array, probabilities),
        "log_loss": log_loss(y_true_array, probabilities, labels=[0, 1]),
        "accuracy_at_threshold": accuracy_score(y_true_array, predictions),
        "precision_at_threshold": precision_score(y_true_array, predictions, zero_division=0),
        "recall_at_threshold": recall_score(y_true_array, predictions, zero_division=0),
        "f1_at_threshold": f1_score(y_true_array, predictions, zero_division=0),
        "threshold": threshold,
        "n_test_rows": int(len(y_true_array)),
    }


def train_cox_model(
    prepared: PreparedData,
    *,
    output_tables_dir: Path,
    model_path: Path,
) -> tuple[Any | None, str | None]:
    """Fit Cox PH if possible and save summary/diagnostics.

    Cox is treated as a survival-analysis complement to the classifiers. If the
    processed snapshot is not suitable for Cox, the function writes a warning
    and returns without interrupting the rest of the workflow.
    """

    try:
        from lifelines import CoxPHFitter
    except ImportError as exc:  # pragma: no cover - exercised in missing envs
        message = (
            "Cox PH skipped because `lifelines` is not installed. "
            "Run `pip install -r requirements.txt`."
        )
        _write_warning(output_tables_dir / "cox_warning.txt", message)
        LOGGER.warning(message)
        return None, message

    if prepared.duration_column is None:
        message = (
            "Cox PH skipped because no duration column was found. Expected a tenure-like "
            f"column. Available columns: {prepared.frame.columns.tolist()}"
        )
        _write_warning(output_tables_dir / "cox_warning.txt", message)
        LOGGER.warning(message)
        return None, message

    survival_features = [
        column
        for column in prepared.feature_columns
        if column != prepared.duration_column and pd.api.types.is_numeric_dtype(prepared.frame[column])
    ]
    if not survival_features:
        message = "Cox PH skipped because no numeric covariates were available after leakage filtering."
        _write_warning(output_tables_dir / "cox_warning.txt", message)
        LOGGER.warning(message)
        return None, message

    cox_frame = prepared.frame[[prepared.duration_column, prepared.target_column, *survival_features]].copy()
    cox_frame = cox_frame.rename(
        columns={prepared.duration_column: "duration", prepared.target_column: "event"}
    )
    cox_frame["duration"] = pd.to_numeric(cox_frame["duration"], errors="coerce")
    cox_frame["event"] = pd.to_numeric(cox_frame["event"], errors="coerce")
    cox_frame = cox_frame.dropna(subset=["duration", "event"])
    cox_frame = cox_frame[cox_frame["duration"] > 0].copy()
    cox_frame["event"] = cox_frame["event"].astype(int)

    for column in survival_features:
        cox_frame[column] = pd.to_numeric(cox_frame[column], errors="coerce")
        cox_frame[column] = cox_frame[column].fillna(cox_frame[column].median())

    constant_columns = [
        column for column in survival_features if cox_frame[column].nunique(dropna=True) <= 1
    ]
    if constant_columns:
        cox_frame = cox_frame.drop(columns=constant_columns)
        survival_features = [column for column in survival_features if column not in constant_columns]

    if len(cox_frame) < 20 or cox_frame["event"].nunique() < 2:
        message = (
            "Cox PH skipped because the survival dataset is too small or has only one event class "
            f"(rows={len(cox_frame)}, event_classes={cox_frame['event'].nunique()})."
        )
        _write_warning(output_tables_dir / "cox_warning.txt", message)
        LOGGER.warning(message)
        return None, message

    try:
        cph = CoxPHFitter(penalizer=0.1)
        cph.fit(cox_frame, duration_col="duration", event_col="event")
        cph.summary.to_csv(output_tables_dir / "cox_summary.csv")

        diagnostics_buffer = io.StringIO()
        diagnostics_warning: str | None = None
        with contextlib.redirect_stdout(diagnostics_buffer):
            try:
                cph.check_assumptions(cox_frame, p_value_threshold=0.05, show_plots=False)
            except Exception as exc:  # diagnostics can fail even when fit works
                diagnostics_warning = f"Proportional hazards diagnostics failed: {exc}"
                print(diagnostics_warning)

        diagnostic_text = diagnostics_buffer.getvalue().strip()
        if not diagnostic_text:
            diagnostic_text = "No proportional hazards diagnostic warnings were emitted."
        if diagnostics_warning:
            LOGGER.warning(diagnostics_warning)
        (output_tables_dir / "proportional_hazards_diagnostics.txt").write_text(
            diagnostic_text + "\n", encoding="utf-8"
        )

        with model_path.open("wb") as file:
            pickle.dump(cph, file)
        return cph, None
    except Exception as exc:  # pragma: no cover - data dependent
        message = f"Cox PH skipped because model fitting failed: {exc}"
        _write_warning(output_tables_dir / "cox_warning.txt", message)
        LOGGER.warning(message)
        return None, message


def predict_cox_probability(
    cox_model: Any,
    prepared: PreparedData,
    x_test: pd.DataFrame,
) -> np.ndarray | None:
    """Approximate event probability for Cox at the median observed duration."""

    if prepared.duration_column is None:
        return None

    covariates = [
        column
        for column in cox_model.params_.index.tolist()
        if column in x_test.columns and pd.api.types.is_numeric_dtype(x_test[column])
    ]
    if not covariates:
        return None

    cox_x = x_test[covariates].copy()
    for column in covariates:
        cox_x[column] = pd.to_numeric(cox_x[column], errors="coerce")
        cox_x[column] = cox_x[column].fillna(cox_x[column].median())

    horizon = float(pd.to_numeric(prepared.frame[prepared.duration_column], errors="coerce").median())
    survival = cox_model.predict_survival_function(cox_x, times=[horizon]).T.iloc[:, 0].to_numpy()
    return np.clip(1.0 - survival, 0.0, 1.0)


def choose_best_classifier(metrics_frame: pd.DataFrame) -> str:
    """Choose the classifier for scoring, preferring calibrated XGBoost unless clearly worse."""

    candidates = metrics_frame[metrics_frame["model"].isin(["Logistic Regression", "XGBoost"])].copy()
    if candidates.empty:
        raise ValueError("No classifier metrics available for best-model selection.")

    if "XGBoost" not in candidates["model"].values:
        return "Logistic Regression"
    if "Logistic Regression" not in candidates["model"].values:
        return "XGBoost"

    logistic = candidates[candidates["model"] == "Logistic Regression"].iloc[0]
    xgboost = candidates[candidates["model"] == "XGBoost"].iloc[0]

    logistic_brier_better = logistic["brier_score"] <= xgboost["brier_score"] - 0.01
    logistic_auc_better = logistic["roc_auc"] >= xgboost["roc_auc"] + 0.02
    if logistic_brier_better or logistic_auc_better:
        return "Logistic Regression"
    return "XGBoost"


def save_model_artifact(model: Any, path: Path) -> None:
    """Persist a model artifact with joblib."""

    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
    LOGGER.info("Saved model artifact to %s", path)


def confusion_matrix_frame(y_true: pd.Series, y_probability: np.ndarray, threshold: float) -> pd.DataFrame:
    """Return a labeled confusion matrix data frame."""

    predictions = (np.asarray(y_probability) >= threshold).astype(int)
    matrix = confusion_matrix(y_true, predictions, labels=[0, 1])
    return pd.DataFrame(matrix, index=["actual_not_churn", "actual_churn"], columns=["pred_not_churn", "pred_churn"])


def _write_warning(path: Path, message: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(message + "\n", encoding="utf-8")

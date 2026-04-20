from __future__ import annotations

from dataclasses import dataclass
from statistics import mean
from typing import Any

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


MODEL_FEATURE_EXCLUDE = {
    "setup_id",
    "label",
    "label_name",
    "label_source",
    "realized_return",
    "realized_mae",
    "realized_mfe",
    "quality_bucket",
    "setup_timestamp",
    "session_date",
    "setup_date",
    "setup_year_month",
    "walk_forward_fold",
}
SAFE_BASELINE_COLUMNS = {"setup_name", "symbol", "direction"}
SAFE_FEATURE_PREFIXES = ("feature_",)
LEAKAGE_PRONE_FEATURES = {
    "feature_rr_to_first_target",
    "feature_first_liquidity_target",
}


@dataclass(slots=True)
class BaselineFoldResult:
    model_name: str
    fold_id: int
    train_rows: int
    test_rows: int
    roc_auc: float | None
    log_loss: float | None
    brier_score: float | None
    threshold: float
    selected_setups: int
    selected_win_rate: float | None
    selected_average_r: float | None
    selected_profit_factor: float | None


@dataclass(slots=True)
class BaselineRunSummary:
    model_name: str
    folds_used: int
    average_roc_auc: float | None
    average_log_loss: float | None
    average_brier_score: float | None
    selected_setups: int
    selected_win_rate: float | None
    selected_average_r: float | None
    selected_profit_factor: float | None


class BaselineSetupQualityModel:
    def __init__(self, estimator: Pipeline, feature_columns: list[str]) -> None:
        self.estimator = estimator
        self.feature_columns = feature_columns

    def predict_proba(self, dataset: pd.DataFrame) -> pd.Series:
        probabilities = self.estimator.predict_proba(dataset[self.feature_columns])[:, 1]
        return pd.Series(probabilities, index=dataset.index, dtype="float64")



def prepare_baseline_features(dataset: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    if "label" not in dataset.columns:
        raise ValueError("Dataset must include a label column")
    feature_columns = _select_model_feature_columns(dataset)
    features = dataset.loc[:, feature_columns].copy()
    target = dataset["label"].astype(int)
    return features, target


def _select_model_feature_columns(dataset: pd.DataFrame) -> list[str]:
    selected: list[str] = []
    for column in dataset.columns:
        if column in MODEL_FEATURE_EXCLUDE:
            continue
        if column in SAFE_BASELINE_COLUMNS:
            selected.append(column)
            continue
        if any(column.startswith(prefix) for prefix in SAFE_FEATURE_PREFIXES) and column not in LEAKAGE_PRONE_FEATURES:
            selected.append(column)
    return selected



def fit_baseline_model(model_name: str, dataset: pd.DataFrame) -> BaselineSetupQualityModel:
    features, target = prepare_baseline_features(dataset)
    if target.nunique() < 2:
        raise ValueError("Training data must contain at least two classes")
    estimator = _make_pipeline(features, model_name)
    estimator.fit(features, target)
    return BaselineSetupQualityModel(estimator=estimator, feature_columns=list(features.columns))



def evaluate_baseline_fold(
    model_name: str,
    fold_id: int,
    train_frame: pd.DataFrame,
    test_frame: pd.DataFrame,
    *,
    threshold: float = 0.55,
) -> BaselineFoldResult:
    model = fit_baseline_model(model_name, train_frame)
    probabilities = model.predict_proba(test_frame)
    y_true = test_frame["label"].astype(int)
    selected = test_frame.loc[probabilities >= threshold].copy()

    return BaselineFoldResult(
        model_name=model_name,
        fold_id=fold_id,
        train_rows=len(train_frame),
        test_rows=len(test_frame),
        roc_auc=_safe_roc_auc(y_true, probabilities),
        log_loss=_safe_log_loss(y_true, probabilities),
        brier_score=_safe_brier_score(y_true, probabilities),
        threshold=threshold,
        selected_setups=len(selected),
        selected_win_rate=(float(selected["label"].mean()) if not selected.empty else None),
        selected_average_r=(float(selected["realized_return"].mean()) if not selected.empty else None),
        selected_profit_factor=_selected_profit_factor(selected),
    )



def summarize_baseline_results(results: list[BaselineFoldResult]) -> BaselineRunSummary:
    if not results:
        return BaselineRunSummary(
            model_name="",
            folds_used=0,
            average_roc_auc=None,
            average_log_loss=None,
            average_brier_score=None,
            selected_setups=0,
            selected_win_rate=None,
            selected_average_r=None,
            selected_profit_factor=None,
        )
    model_name = results[0].model_name
    all_selected_setups = sum(result.selected_setups for result in results)
    selected_win_rates = [result.selected_win_rate for result in results if result.selected_win_rate is not None]
    selected_average_rs = [result.selected_average_r for result in results if result.selected_average_r is not None]
    selected_profit_factors = [result.selected_profit_factor for result in results if result.selected_profit_factor is not None]
    return BaselineRunSummary(
        model_name=model_name,
        folds_used=len(results),
        average_roc_auc=_mean_or_none([result.roc_auc for result in results]),
        average_log_loss=_mean_or_none([result.log_loss for result in results]),
        average_brier_score=_mean_or_none([result.brier_score for result in results]),
        selected_setups=all_selected_setups,
        selected_win_rate=(mean(selected_win_rates) if selected_win_rates else None),
        selected_average_r=(mean(selected_average_rs) if selected_average_rs else None),
        selected_profit_factor=(mean(selected_profit_factors) if selected_profit_factors else None),
    )



def _make_pipeline(features: pd.DataFrame, model_name: str) -> Pipeline:
    numeric_columns = [column for column in features.columns if pd.api.types.is_numeric_dtype(features[column])]
    categorical_columns = [column for column in features.columns if column not in numeric_columns]

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric_columns,
            ),
            (
                "categorical",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                    ]
                ),
                categorical_columns,
            ),
        ],
        remainder="drop",
    )

    if model_name == "logistic_regression":
        estimator: Any = LogisticRegression(max_iter=1000, class_weight="balanced")
    elif model_name == "gradient_boosting":
        estimator = HistGradientBoostingClassifier(max_depth=4, learning_rate=0.05, max_iter=200)
    else:
        raise ValueError(f"Unsupported baseline model: {model_name}")

    return Pipeline(steps=[("preprocessor", preprocessor), ("estimator", estimator)])



def _safe_roc_auc(y_true: pd.Series, probabilities: pd.Series) -> float | None:
    if y_true.nunique() < 2:
        return None
    return float(roc_auc_score(y_true, probabilities))



def _safe_log_loss(y_true: pd.Series, probabilities: pd.Series) -> float | None:
    if y_true.nunique() < 2:
        return None
    return float(log_loss(y_true, probabilities, labels=[0, 1]))



def _safe_brier_score(y_true: pd.Series, probabilities: pd.Series) -> float | None:
    if y_true.nunique() < 2:
        return None
    return float(brier_score_loss(y_true, probabilities))



def _selected_profit_factor(selected: pd.DataFrame) -> float | None:
    if selected.empty:
        return None
    wins = selected.loc[selected["realized_return"] > 0, "realized_return"].sum()
    losses = selected.loc[selected["realized_return"] < 0, "realized_return"].sum()
    if losses >= 0:
        return None
    return float(wins / abs(losses))



def _mean_or_none(values: list[float | None]) -> float | None:
    available = [value for value in values if value is not None]
    if not available:
        return None
    return float(mean(available))


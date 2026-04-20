from __future__ import annotations

import pandas as pd

from models.baselines import evaluate_baseline_fold, fit_baseline_model, prepare_baseline_features, summarize_baseline_results



def make_dataset() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for idx in range(24):
        label = 1 if idx % 2 == 0 else 0
        rows.append(
            {
                "setup_id": f"s{idx}",
                "setup_name": "orb",
                "symbol": "NQ",
                "setup_timestamp": pd.Timestamp("2024-01-01", tz="America/New_York") + pd.Timedelta(minutes=idx),
                "session_date": pd.Timestamp("2024-01-01", tz="America/New_York"),
                "direction": "long" if label == 1 else "short",
                "entry_reference": 100.0 + idx,
                "stop_reference": 99.0 + idx,
                "target_reference": 102.0 + idx,
                "risk_points": 1.0,
                "reward_points": 2.0,
                "target_r_multiple": 2.0,
                "feature_breakout_strength": float(idx % 5) + label,
                "feature_or_width": 1.0 + (idx % 3),
                "context_stop_mode": "or_boundary",
                "label": label,
                "label_name": "target_before_stop",
                "label_source": "rule",
                "horizon_bars": 20,
                "realized_return": 2.0 if label == 1 else -1.0,
                "realized_mae": -0.5 if label == 1 else -1.0,
                "realized_mfe": 2.0 if label == 1 else 0.25,
                "quality_bucket": "great" if label == 1 else "poor",
            }
        )
    return pd.DataFrame(rows)



def test_fit_baseline_model_returns_probability_capable_estimator() -> None:
    dataset = make_dataset()

    model = fit_baseline_model("logistic_regression", dataset)
    probabilities = model.predict_proba(dataset)

    assert len(probabilities) == len(dataset)
    assert probabilities.between(0.0, 1.0).all()



def test_evaluate_baseline_fold_returns_metrics() -> None:
    dataset = make_dataset()
    train = dataset.iloc[:16].reset_index(drop=True)
    test = dataset.iloc[16:].reset_index(drop=True)

    result = evaluate_baseline_fold("gradient_boosting", 1, train, test, threshold=0.5)

    assert result.model_name == "gradient_boosting"
    assert result.fold_id == 1
    assert result.roc_auc is not None
    assert result.log_loss is not None
    assert result.brier_score is not None



def test_summarize_baseline_results_aggregates_fold_outputs() -> None:
    dataset = make_dataset()
    train = dataset.iloc[:16].reset_index(drop=True)
    test = dataset.iloc[16:].reset_index(drop=True)
    first = evaluate_baseline_fold("logistic_regression", 1, train, test, threshold=0.5)
    second = evaluate_baseline_fold("logistic_regression", 2, train, test, threshold=0.6)

    summary = summarize_baseline_results([first, second])

    assert summary.model_name == "logistic_regression"
    assert summary.folds_used == 2
    assert summary.average_roc_auc is not None

def test_prepare_baseline_features_excludes_context_and_execution_geometry() -> None:
    dataset = make_dataset()
    dataset["context_target_rule"] = "r_multiple"
    dataset["entry_reference"] = 100.0
    dataset["stop_reference"] = 99.0
    dataset["feature_rr_to_first_target"] = 2.0

    features, target = prepare_baseline_features(dataset)

    assert "context_target_rule" not in features.columns
    assert "entry_reference" not in features.columns
    assert "stop_reference" not in features.columns
    assert "feature_rr_to_first_target" not in features.columns
    assert "feature_breakout_strength" in features.columns
    assert "symbol" in features.columns
    assert len(target) == len(dataset)

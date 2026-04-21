from __future__ import annotations

import numpy as np
import pandas as pd

from models.lstm import LSTMConfig, evaluate_lstm_walk_forward, fit_lstm_sequence_model, predict_lstm_probabilities
from models.setup_quality_sequences import SetupSequenceDataset
from models.walk_forward import WalkForwardSplit


def make_sequence_dataset() -> SetupSequenceDataset:
    sequences = np.array(
        [
            [[0.1, 0.0], [0.2, 0.1], [0.3, 0.2]],
            [[0.2, 0.1], [0.3, 0.2], [0.4, 0.3]],
            [[0.0, -0.1], [0.1, 0.0], [0.2, 0.1]],
            [[-0.3, -0.2], [-0.2, -0.1], [-0.1, 0.0]],
            [[-0.4, -0.3], [-0.3, -0.2], [-0.2, -0.1]],
            [[-0.5, -0.4], [-0.4, -0.3], [-0.3, -0.2]],
        ],
        dtype=np.float32,
    )
    metadata = pd.DataFrame(
        {
            "setup_id": [f"s{i}" for i in range(6)],
            "setup_timestamp": pd.to_datetime(
                [
                    "2024-01-03 10:00",
                    "2024-02-03 10:00",
                    "2024-03-03 10:00",
                    "2024-04-03 10:00",
                    "2024-05-03 10:00",
                    "2024-06-03 10:00",
                ]
            ).tz_localize("America/New_York"),
            "label": [1, 1, 1, 0, 0, 0],
            "realized_return": [1.0, 0.8, 0.6, -1.0, -0.9, -0.8],
        }
    )
    return SetupSequenceDataset(
        sequences=sequences,
        metadata=metadata,
        feature_columns=["f1", "f2"],
        lookback_bars=3,
        include_setup_bar=True,
    )


def test_fit_lstm_sequence_model_predicts_probabilities() -> None:
    dataset = make_sequence_dataset()
    config = LSTMConfig(hidden_size=4, epochs=3, batch_size=2)

    model, scaler = fit_lstm_sequence_model(dataset.sequences, dataset.metadata["label"], config)
    probabilities = predict_lstm_probabilities(model, dataset.sequences, scaler, config)

    assert probabilities.shape == (6,)
    assert np.all(probabilities >= 0.0)
    assert np.all(probabilities <= 1.0)


def test_evaluate_lstm_walk_forward_returns_fold_metrics() -> None:
    dataset = make_sequence_dataset()
    splits = [
        WalkForwardSplit(
            fold_id=1,
            train_period_start="2024Q1",
            train_period_end="2024Q1",
            test_period_start="2024Q2",
            test_period_end="2024Q2",
            train_indices=[0, 1, 3, 4],
            test_indices=[2, 5],
        )
    ]
    config = LSTMConfig(hidden_size=4, epochs=2, batch_size=2, classification_threshold=0.5)

    predictions, fold_results, summary = evaluate_lstm_walk_forward(dataset, splits, config)

    assert len(predictions) == 2
    assert len(fold_results) == 1
    assert summary.folds_used == 1
    assert "probability" in predictions.columns

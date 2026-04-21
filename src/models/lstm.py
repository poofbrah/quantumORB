from __future__ import annotations

from dataclasses import dataclass
from statistics import mean

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .setup_quality_sequences import SetupSequenceDataset
from .walk_forward import WalkForwardSplit


@dataclass(slots=True)
class LSTMConfig:
    hidden_size: int = 32
    num_layers: int = 1
    dropout: float = 0.0
    learning_rate: float = 1e-3
    batch_size: int = 32
    epochs: int = 20
    device: str = "cpu"
    seed: int = 7
    classification_threshold: float = 0.5


@dataclass(slots=True)
class LSTMFoldResult:
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
class LSTMRunSummary:
    folds_used: int
    average_roc_auc: float | None
    average_log_loss: float | None
    average_brier_score: float | None
    selected_setups: int
    selected_win_rate: float | None
    selected_average_r: float | None
    selected_profit_factor: float | None


class LSTMSequenceClassifier(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int, dropout: float) -> None:
        super().__init__()
        effective_dropout = dropout if num_layers > 1 else 0.0
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=effective_dropout,
        )
        self.head = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output, _ = self.lstm(x)
        last_hidden = output[:, -1, :]
        logits = self.head(last_hidden)
        return logits.squeeze(-1)


def evaluate_lstm_walk_forward(
    dataset: SetupSequenceDataset,
    splits: list[WalkForwardSplit],
    config: LSTMConfig | None = None,
) -> tuple[pd.DataFrame, list[LSTMFoldResult], LSTMRunSummary]:
    settings = config or LSTMConfig()
    prediction_frames: list[pd.DataFrame] = []
    fold_results: list[LSTMFoldResult] = []

    for split in splits:
        train_metadata = dataset.metadata.iloc[split.train_indices].reset_index(drop=True)
        test_metadata = dataset.metadata.iloc[split.test_indices].reset_index(drop=True)
        if train_metadata["label"].nunique() < 2 or test_metadata["label"].nunique() < 2:
            continue
        train_sequences = dataset.sequences[split.train_indices]
        test_sequences = dataset.sequences[split.test_indices]
        model, scaler = fit_lstm_sequence_model(train_sequences, train_metadata["label"], settings)
        probabilities = predict_lstm_probabilities(model, test_sequences, scaler, settings)
        fold_frame = test_metadata.copy()
        fold_frame["probability"] = probabilities
        fold_frame["fold_id"] = split.fold_id
        prediction_frames.append(fold_frame)
        fold_results.append(
            _build_fold_result(
                train_rows=len(train_metadata),
                test_rows=len(test_metadata),
                fold_id=split.fold_id,
                test_frame=fold_frame,
                threshold=settings.classification_threshold,
            )
        )

    predictions = pd.concat(prediction_frames, ignore_index=True) if prediction_frames else pd.DataFrame()
    summary = summarize_lstm_results(fold_results)
    return predictions, fold_results, summary


def fit_lstm_sequence_model(
    train_sequences: np.ndarray,
    train_labels: pd.Series | np.ndarray,
    config: LSTMConfig | None = None,
) -> tuple[LSTMSequenceClassifier, dict[str, np.ndarray]]:
    settings = config or LSTMConfig()
    _set_seed(settings.seed)

    features = np.asarray(train_sequences, dtype=np.float32)
    labels = np.asarray(train_labels, dtype=np.float32)
    mean_vector, std_vector = _fit_feature_scaler(features)
    normalized = _transform_sequences(features, mean_vector, std_vector)

    tensor_x = torch.tensor(normalized, dtype=torch.float32)
    tensor_y = torch.tensor(labels, dtype=torch.float32)
    loader = DataLoader(TensorDataset(tensor_x, tensor_y), batch_size=settings.batch_size, shuffle=True)

    device = torch.device(settings.device)
    model = LSTMSequenceClassifier(
        input_size=features.shape[2],
        hidden_size=settings.hidden_size,
        num_layers=settings.num_layers,
        dropout=settings.dropout,
    ).to(device)

    positive_count = max(float(labels.sum()), 1.0)
    negative_count = max(float(len(labels) - labels.sum()), 1.0)
    pos_weight = torch.tensor([negative_count / positive_count], dtype=torch.float32, device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=settings.learning_rate)

    model.train()
    for _ in range(settings.epochs):
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad()
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()

    scaler = {"mean": mean_vector, "std": std_vector}
    return model.cpu(), scaler


def predict_lstm_probabilities(
    model: LSTMSequenceClassifier,
    sequences: np.ndarray,
    scaler: dict[str, np.ndarray],
    config: LSTMConfig | None = None,
) -> np.ndarray:
    settings = config or LSTMConfig()
    normalized = _transform_sequences(np.asarray(sequences, dtype=np.float32), scaler["mean"], scaler["std"])
    tensor_x = torch.tensor(normalized, dtype=torch.float32)
    loader = DataLoader(tensor_x, batch_size=settings.batch_size, shuffle=False)

    model.eval()
    outputs: list[np.ndarray] = []
    with torch.no_grad():
        for batch_x in loader:
            logits = model(batch_x)
            probs = torch.sigmoid(logits).cpu().numpy()
            outputs.append(probs)
    return np.concatenate(outputs, axis=0) if outputs else np.empty((0,), dtype=np.float32)


def summarize_lstm_results(results: list[LSTMFoldResult]) -> LSTMRunSummary:
    if not results:
        return LSTMRunSummary(
            folds_used=0,
            average_roc_auc=None,
            average_log_loss=None,
            average_brier_score=None,
            selected_setups=0,
            selected_win_rate=None,
            selected_average_r=None,
            selected_profit_factor=None,
        )
    selected_win_rates = [result.selected_win_rate for result in results if result.selected_win_rate is not None]
    selected_average_rs = [result.selected_average_r for result in results if result.selected_average_r is not None]
    selected_profit_factors = [result.selected_profit_factor for result in results if result.selected_profit_factor is not None]
    return LSTMRunSummary(
        folds_used=len(results),
        average_roc_auc=_mean_or_none([result.roc_auc for result in results]),
        average_log_loss=_mean_or_none([result.log_loss for result in results]),
        average_brier_score=_mean_or_none([result.brier_score for result in results]),
        selected_setups=sum(result.selected_setups for result in results),
        selected_win_rate=(mean(selected_win_rates) if selected_win_rates else None),
        selected_average_r=(mean(selected_average_rs) if selected_average_rs else None),
        selected_profit_factor=(mean(selected_profit_factors) if selected_profit_factors else None),
    )


def _build_fold_result(train_rows: int, test_rows: int, fold_id: int, test_frame: pd.DataFrame, threshold: float) -> LSTMFoldResult:
    y_true = test_frame["label"].astype(int)
    probabilities = test_frame["probability"].astype(float)
    selected = test_frame.loc[probabilities >= threshold]
    wins = selected.loc[selected["realized_return"] > 0, "realized_return"].sum()
    losses = selected.loc[selected["realized_return"] < 0, "realized_return"].sum()
    profit_factor = float(wins / abs(losses)) if losses < 0 else None
    return LSTMFoldResult(
        fold_id=fold_id,
        train_rows=train_rows,
        test_rows=test_rows,
        roc_auc=(float(roc_auc_score(y_true, probabilities)) if y_true.nunique() >= 2 else None),
        log_loss=(float(log_loss(y_true, probabilities, labels=[0, 1])) if y_true.nunique() >= 2 else None),
        brier_score=(float(brier_score_loss(y_true, probabilities)) if y_true.nunique() >= 2 else None),
        threshold=threshold,
        selected_setups=len(selected),
        selected_win_rate=(float(selected["label"].mean()) if not selected.empty else None),
        selected_average_r=(float(selected["realized_return"].mean()) if not selected.empty else None),
        selected_profit_factor=profit_factor,
    )


def _fit_feature_scaler(sequences: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    flat = sequences.reshape(-1, sequences.shape[-1])
    mean_vector = np.nanmean(flat, axis=0).astype(np.float32)
    std_vector = np.nanstd(flat, axis=0).astype(np.float32)
    std_vector = np.where(std_vector == 0.0, 1.0, std_vector)
    mean_vector = np.nan_to_num(mean_vector, nan=0.0)
    std_vector = np.nan_to_num(std_vector, nan=1.0)
    return mean_vector, std_vector


def _transform_sequences(sequences: np.ndarray, mean_vector: np.ndarray, std_vector: np.ndarray) -> np.ndarray:
    normalized = (sequences - mean_vector.reshape(1, 1, -1)) / std_vector.reshape(1, 1, -1)
    return np.nan_to_num(normalized, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def _set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


def _mean_or_none(values: list[float | None]) -> float | None:
    available = [value for value in values if value is not None]
    if not available:
        return None
    return float(mean(available))

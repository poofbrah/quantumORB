from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from models import (
    LSTMConfig,
    SetupSequenceDataset,
    build_walk_forward_splits,
    evaluate_lstm_walk_forward,
)
from reporting import (
    save_fold_metric_chart,
    save_model_summary_table_image,
    save_threshold_metric_chart,
    select_best_threshold_row,
)

DEFAULT_SEQUENCE_DIR = ROOT / "outputs" / "setup_quality_sequence_dataset_orb_session_vwap_retest_liquidity"
DEFAULT_OUTPUT_DIR = ROOT / "outputs" / "setup_quality_lstm_orb_session_vwap_retest_liquidity"
THRESHOLDS = (0.50, 0.55, 0.60, 0.65, 0.70, 0.75)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run walk-forward LSTM setup-quality evaluation.")
    parser.add_argument("--sequence-dir", default=str(DEFAULT_SEQUENCE_DIR))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--hidden-size", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--threshold", type=float, default=0.50)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sequence_dir = Path(args.sequence_dir)
    output_dir = Path(args.output_dir)
    chart_dir = output_dir / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    chart_dir.mkdir(parents=True, exist_ok=True)

    dataset = load_sequence_dataset(sequence_dir)
    splits = build_walk_forward_splits(
        dataset.metadata,
        frequency="Q",
        train_periods=4,
        test_periods=1,
        step_periods=1,
        min_train_rows=25,
        min_test_rows=10,
    )
    if not splits:
        raise RuntimeError("No walk-forward splits available for the sequence dataset.")

    config = LSTMConfig(
        hidden_size=args.hidden_size,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        classification_threshold=args.threshold,
    )
    predictions, fold_results, summary = evaluate_lstm_walk_forward(dataset, splits, config=config)
    if predictions.empty:
        raise RuntimeError("LSTM evaluation produced no out-of-sample predictions.")

    fold_results_frame = pd.DataFrame([{"model_name": "lstm", **asdict(result)} for result in fold_results])
    threshold_results = build_threshold_sweep(predictions, THRESHOLDS)
    threshold_results_frame = pd.DataFrame(threshold_results)
    best_row = select_best_threshold_row(threshold_results_frame, priority=("profit_factor", "sharpe", "net_pnl"))

    fold_results_path = output_dir / "fold_results.csv"
    predictions_path = output_dir / "oos_predictions.csv"
    threshold_path = output_dir / "threshold_sweep.csv"
    summary_path = output_dir / "lstm_summary.json"
    fold_results_frame.to_csv(fold_results_path, index=False)
    predictions.to_csv(predictions_path, index=False)
    threshold_results_frame.to_csv(threshold_path, index=False)

    summary_frame = pd.DataFrame(
        [
            {
                "model_name": "lstm",
                **asdict(summary),
                "best_threshold": float(best_row["threshold"]),
                "best_profit_factor": best_row["profit_factor"],
                "best_net_pnl": best_row["net_pnl"],
                "best_sharpe": best_row["sharpe"],
                "best_sortino": best_row["sortino"],
                "best_calmar": best_row["calmar"],
                "best_trades": int(best_row["trades_executed"]),
            }
        ]
    )
    payload = {
        "sequence_count": int(dataset.sequences.shape[0]),
        "lookback_bars": dataset.lookback_bars,
        "feature_count": len(dataset.feature_columns),
        "walk_forward_folds": len(splits),
        "config": asdict(config),
        "summary": asdict(summary),
        "best_threshold": best_row.to_dict(),
        "paths": {
            "fold_results": str(fold_results_path),
            "predictions": str(predictions_path),
            "threshold_sweep": str(threshold_path),
            "figures": str(chart_dir),
        },
    }
    summary_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")

    save_fold_metric_chart(fold_results_frame, chart_dir / "walk_forward_roc_auc.png", metric="roc_auc")
    save_fold_metric_chart(fold_results_frame, chart_dir / "walk_forward_brier_score.png", metric="brier_score")
    save_threshold_metric_chart(threshold_results_frame, chart_dir / "threshold_profit_factor.png", metric="profit_factor", title="LSTM Threshold vs Profit Factor")
    save_threshold_metric_chart(threshold_results_frame, chart_dir / "threshold_sharpe.png", metric="sharpe", title="LSTM Threshold vs Sharpe")
    save_threshold_metric_chart(threshold_results_frame, chart_dir / "threshold_net_pnl.png", metric="net_pnl", title="LSTM Threshold vs Net PnL")
    save_model_summary_table_image(summary_frame, chart_dir / "lstm_summary_table.png")

    print(f"sequence_count={payload['sequence_count']}")
    print(f"lookback_bars={payload['lookback_bars']}")
    print(f"feature_count={payload['feature_count']}")
    print(f"walk_forward_folds={payload['walk_forward_folds']}")
    print(f"average_roc_auc={format_metric(summary.average_roc_auc)}")
    print(f"average_log_loss={format_metric(summary.average_log_loss)}")
    print(f"average_brier_score={format_metric(summary.average_brier_score)}")
    print(f"selected_setups={summary.selected_setups}")
    print(f"selected_win_rate={format_metric(summary.selected_win_rate)}")
    print(f"selected_average_r={format_metric(summary.selected_average_r)}")
    print(f"selected_profit_factor={format_metric(summary.selected_profit_factor)}")
    print(f"best_threshold={float(best_row['threshold']):.2f}")
    print(f"best_trades={int(best_row['trades_executed'])}")
    print(f"best_win_rate={format_metric(best_row['win_rate'])}")
    print(f"best_profit_factor={format_metric(best_row['profit_factor'])}")
    print(f"best_net_pnl={format_metric(best_row['net_pnl'])}")
    print(f"best_sharpe={format_metric(best_row['sharpe'])}")
    print(f"saved={output_dir}")


def load_sequence_dataset(sequence_dir: Path) -> SetupSequenceDataset:
    metadata = pd.read_parquet(sequence_dir / "sequence_metadata.parquet")
    payload = np.load(sequence_dir / "sequence_data.npz", allow_pickle=True)
    sequences = payload["sequences"]
    feature_columns = payload["feature_columns"].tolist()
    lookback_bars = int(sequences.shape[1]) if sequences.ndim == 3 else 0
    return SetupSequenceDataset(
        sequences=sequences,
        metadata=metadata,
        feature_columns=list(feature_columns),
        lookback_bars=lookback_bars,
        include_setup_bar=True,
    )


def build_threshold_sweep(predictions: pd.DataFrame, thresholds: tuple[float, ...]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for threshold in thresholds:
        selected = predictions.loc[predictions["probability"] >= threshold].copy()
        realized = selected["realized_return"].astype(float) if not selected.empty else pd.Series(dtype="float64")
        win_rate = float(selected["label"].mean()) if not selected.empty else None
        gross_profit = realized[realized > 0].sum()
        gross_loss = realized[realized < 0].sum()
        profit_factor = float(gross_profit / abs(gross_loss)) if gross_loss < 0 else None
        net_pnl = float(realized.sum()) if not realized.empty else 0.0
        sharpe = _sharpe(realized.tolist()) if not realized.empty else None
        sortino = _sortino(realized.tolist()) if not realized.empty else None
        max_drawdown = _max_drawdown(realized.tolist()) if not realized.empty else 0.0
        total_return = float(realized.sum()) if not realized.empty else 0.0
        calmar = _calmar(total_return, max_drawdown)
        rows.append(
            {
                "model_name": "lstm",
                "threshold": threshold,
                "setups_detected": len(selected),
                "trades_executed": len(selected),
                "win_rate": win_rate,
                "profit_factor": profit_factor,
                "net_pnl": net_pnl,
                "max_drawdown": max_drawdown,
                "sharpe": sharpe,
                "sortino": sortino,
                "calmar": calmar,
            }
        )
    return rows


def _sharpe(values: list[float]) -> float | None:
    if len(values) < 2:
        return None
    series = pd.Series(values, dtype="float64")
    std = float(series.std(ddof=1))
    if std == 0.0:
        return None
    return float((series.mean() / std) * np.sqrt(len(series)))


def _max_drawdown(values: list[float]) -> float:
    if not values:
        return 0.0
    equity = pd.Series(values, dtype="float64").cumsum()
    peaks = equity.cummax()
    drawdown = equity - peaks
    return float(drawdown.min())


def _sortino(values: list[float]) -> float | None:
    if len(values) < 2:
        return None
    series = pd.Series(values, dtype="float64")
    downside = series.clip(upper=0.0)
    downside_std = float((downside.pow(2).mean()) ** 0.5)
    if downside_std == 0.0:
        return None
    return float((series.mean() / downside_std) * np.sqrt(len(series)))


def _calmar(total_return: float, max_drawdown: float) -> float | None:
    if max_drawdown >= 0.0:
        return None
    if max_drawdown == 0.0:
        return None
    return float(total_return / abs(max_drawdown))


def format_metric(value: object) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


if __name__ == "__main__":
    main()

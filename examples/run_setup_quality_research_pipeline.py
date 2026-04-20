from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from backtest.engine import BarBacktestEngine, BacktestConfig, BacktestRunConfig
from data.preprocess import filter_session_hours
from evaluation.metrics import calculate_summary_metrics
from features.pipeline import build_feature_frame
from labeling.future_path import ForwardSetupLabeler, LabelerConfig
from models import (
    build_walk_forward_splits,
    fit_baseline_model,
    labeled_setups_to_frame,
    summarize_baseline_results,
)
from reporting import (
    build_setup_summary_frame,
    build_trade_log_frame,
    save_equity_curve_chart,
    save_fold_metric_chart,
    save_label_distribution_chart,
    save_model_summary_table_image,
    save_threshold_metric_chart,
    select_best_threshold_row,
)
from setups.orb_session_vwap_retest import ORBSessionVWAPRetestConfig, ORBSessionVWAPRetestDetector

INPUT_PATH = ROOT / "data" / "processed" / "nq_1min_2022_2025.parquet"
MODEL_NAMES = ("logistic_regression", "gradient_boosting")
THRESHOLDS = (0.50, 0.55, 0.60, 0.65, 0.70, 0.75)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the single-button setup-quality research pipeline.")
    parser.add_argument("--target-mode", choices=["fixed_r", "liquidity"], default="liquidity")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--disable-trend-alignment", action="store_true")
    parser.add_argument("--disable-fvg-context", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir) if args.output_dir else ROOT / "outputs" / f"setup_quality_research_orb_session_vwap_retest_{args.target_mode}"
    chart_dir = output_dir / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    chart_dir.mkdir(parents=True, exist_ok=True)

    print("stage=load_data")
    frame = pd.read_parquet(INPUT_PATH)
    rth_frame = filter_session_hours(frame, session_start="09:30", session_end="16:00")

    print("stage=build_features")
    featured = build_feature_frame(rth_frame, opening_range_minutes=15)
    featured_path = output_dir / "featured_nq_1min_2022_2025.parquet"
    featured.to_parquet(featured_path, index=False)

    print("stage=detect_setups")
    detector = ORBSessionVWAPRetestDetector(
        ORBSessionVWAPRetestConfig(
            opening_range_start_time="09:30",
            opening_range_end_time="09:45",
            opening_range_minutes=15,
            latest_entry_time="11:30",
            allowed_trade_windows=("09:45-11:30",),
            target_r_multiple=2.0,
            target_mode=args.target_mode,
            require_trend_alignment=not args.disable_trend_alignment,
            require_fvg_context=not args.disable_fvg_context,
        )
    )
    setups = detector.detect(featured)

    print("stage=label_setups")
    labeler = ForwardSetupLabeler(LabelerConfig(horizon_bars=20))
    labeled_setups = labeler.label(featured, setups)
    dataset = labeled_setups_to_frame(labeled_setups)
    dataset_path = output_dir / "setup_quality_dataset.parquet"
    dataset.to_parquet(dataset_path, index=False)
    build_setup_summary_frame(labeled_setups).to_csv(output_dir / "setup_summary.csv", index=False)

    print("stage=baseline_backtest")
    baseline_engine = BarBacktestEngine(BacktestConfig())
    baseline_result = baseline_engine.run(
        featured,
        setups,
        BacktestRunConfig(strategy_name=f"orb_session_vwap_retest_{args.target_mode}_all"),
    )
    baseline_trade_log = build_trade_log_frame(baseline_result.trades)
    baseline_trade_log.to_csv(output_dir / "trade_log_all_setups.csv", index=False)
    baseline_metrics = _result_metrics(baseline_result, len(featured), len(setups))
    (output_dir / "metrics_all_setups.json").write_text(json.dumps(baseline_metrics, indent=2, default=str), encoding="utf-8")

    print("stage=walk_forward")
    splits = build_walk_forward_splits(
        dataset,
        frequency="Q",
        train_periods=4,
        test_periods=1,
        step_periods=1,
        min_train_rows=25,
        min_test_rows=10,
    )
    if not splits:
        raise RuntimeError("No walk-forward splits were produced for the research dataset.")

    setup_by_id = {setup.setup_id: setup for setup in setups}
    fold_rows: list[dict[str, object]] = []
    prediction_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []
    threshold_rows: list[dict[str, object]] = []
    best_model_payload: dict[str, object] = {}

    for model_name in MODEL_NAMES:
        print(f"stage=model_fit model={model_name}")
        fold_results = []
        model_predictions: list[pd.DataFrame] = []
        for split in splits:
            train_frame = dataset.iloc[split.train_indices].reset_index(drop=True)
            test_frame = dataset.iloc[split.test_indices].reset_index(drop=True)
            if train_frame["label"].nunique() < 2 or test_frame["label"].nunique() < 2:
                continue
            model = fit_baseline_model(model_name, train_frame)
            probabilities = model.predict_proba(test_frame)
            test_payload = test_frame.copy()
            test_payload["probability"] = probabilities.values
            test_payload["fold_id"] = split.fold_id
            test_payload["model_name"] = model_name
            test_payload["train_period_start"] = split.train_period_start
            test_payload["train_period_end"] = split.train_period_end
            test_payload["test_period_start"] = split.test_period_start
            test_payload["test_period_end"] = split.test_period_end
            model_predictions.append(test_payload)

            fold_result = _evaluate_fold_from_probabilities(model_name, split.fold_id, train_frame, test_payload, threshold=0.55)
            fold_results.append(fold_result)
            fold_rows.append(
                {
                    "model_name": fold_result["model_name"],
                    "fold_id": fold_result["fold_id"],
                    "train_rows": fold_result["train_rows"],
                    "test_rows": fold_result["test_rows"],
                    "roc_auc": fold_result["roc_auc"],
                    "log_loss": fold_result["log_loss"],
                    "brier_score": fold_result["brier_score"],
                    "threshold": fold_result["threshold"],
                    "selected_setups": fold_result["selected_setups"],
                    "selected_win_rate": fold_result["selected_win_rate"],
                    "selected_average_r": fold_result["selected_average_r"],
                    "selected_profit_factor": fold_result["selected_profit_factor"],
                    "train_period_start": split.train_period_start,
                    "train_period_end": split.train_period_end,
                    "test_period_start": split.test_period_start,
                    "test_period_end": split.test_period_end,
                }
            )

        if not model_predictions:
            continue

        prediction_frame = pd.concat(model_predictions, ignore_index=True)
        prediction_rows.extend(prediction_frame.to_dict(orient="records"))
        summary = asdict(summarize_baseline_results([_fold_result_from_dict(item) for item in fold_results]))
        summary_rows.append({"model_name": model_name, **summary})

        print(f"stage=threshold_sweep model={model_name}")
        for threshold in THRESHOLDS:
            selected_ids = set(prediction_frame.loc[prediction_frame["probability"] >= threshold, "setup_id"])
            selected_setups = [setup_by_id[setup_id] for setup_id in selected_ids if setup_id in setup_by_id]
            run_result = baseline_engine.run(
                featured,
                selected_setups,
                BacktestRunConfig(strategy_name=f"orb_session_vwap_retest_{args.target_mode}_{model_name}_{threshold:.2f}"),
            )
            summary_metrics = _result_metrics(run_result, len(featured), len(selected_setups))
            threshold_rows.append(
                {
                    "model_name": model_name,
                    "threshold": threshold,
                    "setups_detected": len(selected_setups),
                    "trades_executed": run_result.total_trades,
                    "win_rate": summary_metrics["win_rate"],
                    "profit_factor": summary_metrics["profit_factor"],
                    "net_pnl": summary_metrics["net_pnl"],
                    "max_drawdown": summary_metrics["max_drawdown"],
                    "max_drawdown_pct": summary_metrics["max_drawdown_pct"],
                    "sharpe": summary_metrics["sharpe"],
                    "sortino": summary_metrics["sortino"],
                    "calmar": summary_metrics["calmar"],
                }
            )

        model_threshold_frame = pd.DataFrame([row for row in threshold_rows if row["model_name"] == model_name])
        best_row = select_best_threshold_row(model_threshold_frame)
        best_threshold = float(best_row["threshold"])
        best_ids = set(prediction_frame.loc[prediction_frame["probability"] >= best_threshold, "setup_id"])
        best_setups = [setup_by_id[setup_id] for setup_id in best_ids if setup_id in setup_by_id]
        best_result = baseline_engine.run(
            featured,
            best_setups,
            BacktestRunConfig(strategy_name=f"orb_session_vwap_retest_{args.target_mode}_{model_name}_best"),
        )
        build_trade_log_frame(best_result.trades).to_csv(output_dir / f"trade_log_{model_name}_best.csv", index=False)
        (output_dir / f"metrics_{model_name}_best.json").write_text(
            json.dumps(_result_metrics(best_result, len(featured), len(best_setups)), indent=2, default=str),
            encoding="utf-8",
        )
        save_equity_curve_chart(
            (
                ("all_setups", baseline_result.equity_curve),
                (f"{model_name}_best", best_result.equity_curve),
            ),
            chart_dir / f"equity_curve_all_vs_{model_name}_best.png",
            title=f"Equity Curve Comparison: All Setups vs {model_name}",
        )
        best_model_payload[model_name] = {
            "best_threshold": best_threshold,
            "setups_detected": int(best_row["setups_detected"]),
            "trades_executed": int(best_row["trades_executed"]),
            "win_rate": best_row["win_rate"],
            "profit_factor": best_row["profit_factor"],
            "net_pnl": best_row["net_pnl"],
            "max_drawdown": best_row["max_drawdown"],
            "sharpe": best_row["sharpe"],
        }

    fold_results_frame = pd.DataFrame(fold_rows)
    predictions_frame = pd.DataFrame(prediction_rows)
    model_summary_frame = pd.DataFrame(summary_rows)
    threshold_results_frame = pd.DataFrame(threshold_rows)

    fold_results_frame.to_csv(output_dir / "fold_results.csv", index=False)
    predictions_frame.to_csv(output_dir / "oos_predictions.csv", index=False)
    model_summary_frame.to_csv(output_dir / "model_summary.csv", index=False)
    threshold_results_frame.to_csv(output_dir / "threshold_sweep.csv", index=False)

    payload = {
        "rows": len(featured),
        "dataset_rows": len(dataset),
        "setups_detected": len(setups),
        "walk_forward_folds": len(splits),
        "target_mode": args.target_mode,
        "require_trend_alignment": not args.disable_trend_alignment,
        "require_fvg_context": not args.disable_fvg_context,
        "baseline_all_setups": baseline_metrics,
        "model_summaries": model_summary_frame.to_dict(orient="records"),
        "best_thresholds": best_model_payload,
        "paths": {
            "featured": str(featured_path),
            "dataset": str(dataset_path),
            "fold_results": str(output_dir / "fold_results.csv"),
            "predictions": str(output_dir / "oos_predictions.csv"),
            "threshold_sweep": str(output_dir / "threshold_sweep.csv"),
            "figures": str(chart_dir),
        },
    }
    (output_dir / "research_summary.json").write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")

    print("stage=charts")
    save_label_distribution_chart(dataset, chart_dir / "label_distribution.png")
    save_fold_metric_chart(fold_results_frame, chart_dir / "walk_forward_roc_auc.png", metric="roc_auc")
    save_fold_metric_chart(fold_results_frame, chart_dir / "walk_forward_brier_score.png", metric="brier_score")
    save_threshold_metric_chart(threshold_results_frame, chart_dir / "threshold_profit_factor.png", metric="profit_factor", title="Threshold vs Profit Factor")
    save_threshold_metric_chart(threshold_results_frame, chart_dir / "threshold_sharpe.png", metric="sharpe", title="Threshold vs Sharpe")
    save_threshold_metric_chart(threshold_results_frame, chart_dir / "threshold_net_pnl.png", metric="net_pnl", title="Threshold vs Net PnL")
    save_model_summary_table_image(model_summary_frame, chart_dir / "model_summary_table.png")

    print(f"rows={len(featured)}")
    print(f"dataset_rows={len(dataset)}")
    print(f"setups_detected={len(setups)}")
    print(f"walk_forward_folds={len(splits)}")
    print(f"baseline_win_rate={format_metric(baseline_metrics.get('win_rate'))}")
    print(f"baseline_profit_factor={format_metric(baseline_metrics.get('profit_factor'))}")
    for model_name, details in best_model_payload.items():
        print(f"model={model_name}")
        print(f"  best_threshold={details['best_threshold']:.2f}")
        print(f"  trades_executed={details['trades_executed']}")
        print(f"  win_rate={format_metric(details['win_rate'])}")
        print(f"  profit_factor={format_metric(details['profit_factor'])}")
        print(f"  net_pnl={format_metric(details['net_pnl'])}")
        print(f"  sharpe={format_metric(details['sharpe'])}")
    print(f"saved={output_dir}")


def _evaluate_fold_from_probabilities(
    model_name: str,
    fold_id: int,
    train_frame: pd.DataFrame,
    test_frame: pd.DataFrame,
    *,
    threshold: float,
) -> dict[str, object]:
    from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

    y_true = test_frame["label"].astype(int)
    probabilities = test_frame["probability"].astype(float)
    selected = test_frame.loc[probabilities >= threshold].copy()

    roc_auc = float(roc_auc_score(y_true, probabilities)) if y_true.nunique() >= 2 else None
    logloss = float(log_loss(y_true, probabilities, labels=[0, 1])) if y_true.nunique() >= 2 else None
    brier = float(brier_score_loss(y_true, probabilities)) if y_true.nunique() >= 2 else None
    wins = selected.loc[selected["realized_return"] > 0, "realized_return"].sum()
    losses = selected.loc[selected["realized_return"] < 0, "realized_return"].sum()
    profit_factor = float(wins / abs(losses)) if losses < 0 else None
    return {
        "model_name": model_name,
        "fold_id": fold_id,
        "train_rows": len(train_frame),
        "test_rows": len(test_frame),
        "roc_auc": roc_auc,
        "log_loss": logloss,
        "brier_score": brier,
        "threshold": threshold,
        "selected_setups": len(selected),
        "selected_win_rate": float(selected["label"].mean()) if not selected.empty else None,
        "selected_average_r": float(selected["realized_return"].mean()) if not selected.empty else None,
        "selected_profit_factor": profit_factor,
    }


def _fold_result_from_dict(payload: dict[str, object]):
    from models import BaselineFoldResult

    return BaselineFoldResult(
        model_name=str(payload["model_name"]),
        fold_id=int(payload["fold_id"]),
        train_rows=int(payload["train_rows"]),
        test_rows=int(payload["test_rows"]),
        roc_auc=payload["roc_auc"],
        log_loss=payload["log_loss"],
        brier_score=payload["brier_score"],
        threshold=float(payload["threshold"]),
        selected_setups=int(payload["selected_setups"]),
        selected_win_rate=payload["selected_win_rate"],
        selected_average_r=payload["selected_average_r"],
        selected_profit_factor=payload["selected_profit_factor"],
    )


def _result_metrics(result, rows: int, setups_detected: int) -> dict[str, object]:
    summary = calculate_summary_metrics(result.trades, result.equity_curve, 100000.0)
    return {
        "rows": rows,
        "setups_detected": setups_detected,
        "trades_executed": result.total_trades,
        "win_rate": result.win_rate,
        "profit_factor": result.profit_factor,
        "net_pnl": result.net_pnl,
        "max_drawdown": result.max_drawdown,
        "max_drawdown_pct": summary.get("max_drawdown_pct"),
        "total_return": summary.get("total_return"),
        "sharpe": summary.get("sharpe"),
        "sortino": summary.get("sortino"),
        "calmar": summary.get("calmar"),
    }


def format_metric(value: object) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


if __name__ == "__main__":
    main()

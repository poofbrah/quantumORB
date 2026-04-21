from __future__ import annotations

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
from ga import (
    GAOptimizerConfig,
    SetupQualityGenome,
    apply_setup_quality_genome,
    evaluate_setup_quality_genome,
    genome_to_dict,
    optimize_setup_quality_genome,
)
from labeling.future_path import ForwardSetupLabeler, LabelerConfig
from models import build_walk_forward_splits, fit_baseline_model
from reporting import build_trade_log_frame
from setups.orb_session_vwap_retest import ORBSessionVWAPRetestConfig, ORBSessionVWAPRetestDetector

INPUT_PATH = ROOT / "data" / "processed" / "nq_1min_2022_2025.parquet"
OUTPUT_DIR = ROOT / "outputs" / "setup_quality_ga_optimizer"
THRESHOLDS = (0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    frame = pd.read_parquet(INPUT_PATH)
    rth_frame = filter_session_hours(frame, session_start="09:30", session_end="16:00")
    featured = build_feature_frame(rth_frame, opening_range_minutes=15)

    detector = ORBSessionVWAPRetestDetector(
        ORBSessionVWAPRetestConfig(
            opening_range_start_time="09:30",
            opening_range_end_time="09:45",
            opening_range_minutes=15,
            latest_entry_time="11:30",
            allowed_trade_windows=("09:45-11:30",),
            target_r_multiple=2.0,
            target_mode="liquidity",
            require_trend_alignment=True,
            require_fvg_context=False,
        )
    )
    setups = detector.detect(featured)
    setup_by_id = {setup.setup_id: setup for setup in setups}
    labeled = ForwardSetupLabeler(LabelerConfig(horizon_bars=20)).label(featured, setups)
    dataset = pd.DataFrame(
        [
            {
                **{
                    "setup_id": item.setup.setup_id,
                    "setup_timestamp": pd.Timestamp(item.setup.timestamp),
                    "label": item.label,
                    "realized_return": item.realized_return,
                },
                **{f"feature_{key}": value for key, value in item.setup.features.items()},
            }
            for item in labeled
        ]
    )
    if dataset.empty:
        raise RuntimeError("No labeled setups were produced for GA optimization.")
    dataset = dataset.sort_values("setup_timestamp", kind="stable").reset_index(drop=True)
    dataset["symbol"] = "NQ"
    dataset["setup_name"] = "orb_session_vwap_retest"
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
        raise RuntimeError("No walk-forward splits available for GA optimization.")

    engine = BarBacktestEngine(BacktestConfig())
    ga_config = GAOptimizerConfig()
    fold_rows: list[dict[str, object]] = []
    threshold_selected_ids: list[str] = []
    ga_selected_ids: list[str] = []

    for split in splits:
        train_frame = dataset.iloc[split.train_indices].reset_index(drop=True)
        test_frame = dataset.iloc[split.test_indices].reset_index(drop=True)
        if train_frame["label"].nunique() < 2 or test_frame["label"].nunique() < 2:
            continue
        model = fit_baseline_model("gradient_boosting", train_frame)
        train_scored = train_frame.copy()
        train_scored["probability"] = model.predict_proba(train_frame).values
        test_scored = test_frame.copy()
        test_scored["probability"] = model.predict_proba(test_frame).values

        opt_train, val_train = _split_train_validation(train_scored)
        best_threshold = _select_best_threshold(opt_train)
        threshold_selected = test_scored.loc[test_scored["probability"] >= best_threshold].copy()
        threshold_selected_ids.extend(threshold_selected["setup_id"].tolist())

        candidate_scores = []
        threshold_genome = _threshold_only_genome(best_threshold)
        candidate_scores.append(evaluate_setup_quality_genome(val_train, threshold_genome, ga_config))
        for seed in (7, 11, 19, 23):
            tuned_config = GAOptimizerConfig(**{**asdict(ga_config), "seed": seed})
            candidate_scores.append(evaluate_setup_quality_genome(val_train, optimize_setup_quality_genome(opt_train, tuned_config).genome, ga_config))
        best_genome = max(candidate_scores, key=lambda item: item.fitness)
        ga_selected = apply_setup_quality_genome(test_scored, best_genome.genome)
        ga_selected_ids.extend(ga_selected["setup_id"].tolist())

        fold_rows.append(
            {
                "fold_id": split.fold_id,
                "train_period_start": split.train_period_start,
                "train_period_end": split.train_period_end,
                "test_period_start": split.test_period_start,
                "test_period_end": split.test_period_end,
                "threshold_only_threshold": best_threshold,
                "threshold_only_trades": len(threshold_selected),
                "ga_trades": len(ga_selected),
                "ga_validation_fitness": best_genome.fitness,
                "ga_validation_profit_factor": best_genome.profit_factor,
                "ga_validation_sharpe": best_genome.sharpe,
                **{f"ga_{key}": value for key, value in genome_to_dict(best_genome.genome).items()},
            }
        )

    threshold_setups = [setup_by_id[setup_id] for setup_id in dict.fromkeys(threshold_selected_ids) if setup_id in setup_by_id]
    ga_setups = [setup_by_id[setup_id] for setup_id in dict.fromkeys(ga_selected_ids) if setup_id in setup_by_id]

    threshold_result = engine.run(featured, threshold_setups, BacktestRunConfig(strategy_name="gb_threshold_only"))
    ga_result = engine.run(featured, ga_setups, BacktestRunConfig(strategy_name="gb_ga_optimized"))
    threshold_metrics = _result_metrics(threshold_result, len(threshold_setups))
    ga_metrics = _result_metrics(ga_result, len(ga_setups))

    pd.DataFrame(fold_rows).to_csv(OUTPUT_DIR / "ga_fold_parameters.csv", index=False)
    build_trade_log_frame(threshold_result.trades).to_csv(OUTPUT_DIR / "trade_log_threshold_only.csv", index=False)
    build_trade_log_frame(ga_result.trades).to_csv(OUTPUT_DIR / "trade_log_ga_optimized.csv", index=False)

    payload = {
        "rows": len(featured),
        "setups_detected": len(setups),
        "walk_forward_folds": len(splits),
        "baseline_threshold_only": threshold_metrics,
        "ga_optimized": ga_metrics,
        "improvement": {
            "net_pnl_delta": ga_metrics["net_pnl"] - threshold_metrics["net_pnl"],
            "profit_factor_delta": _safe_delta(ga_metrics["profit_factor"], threshold_metrics["profit_factor"]),
            "sharpe_delta": _safe_delta(ga_metrics["sharpe"], threshold_metrics["sharpe"]),
        },
        "paths": {
            "fold_parameters": str(OUTPUT_DIR / "ga_fold_parameters.csv"),
            "threshold_trade_log": str(OUTPUT_DIR / "trade_log_threshold_only.csv"),
            "ga_trade_log": str(OUTPUT_DIR / "trade_log_ga_optimized.csv"),
        },
    }
    (OUTPUT_DIR / "ga_summary.json").write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")

    print(f"rows={payload['rows']}")
    print(f"setups_detected={payload['setups_detected']}")
    print(f"walk_forward_folds={payload['walk_forward_folds']}")
    print("strategy=threshold_only")
    print(f"  trades={threshold_metrics['trades_executed']}")
    print(f"  win_rate={format_metric(threshold_metrics['win_rate'])}")
    print(f"  profit_factor={format_metric(threshold_metrics['profit_factor'])}")
    print(f"  net_pnl={format_metric(threshold_metrics['net_pnl'])}")
    print(f"  sharpe={format_metric(threshold_metrics['sharpe'])}")
    print("strategy=ga_optimized")
    print(f"  trades={ga_metrics['trades_executed']}")
    print(f"  win_rate={format_metric(ga_metrics['win_rate'])}")
    print(f"  profit_factor={format_metric(ga_metrics['profit_factor'])}")
    print(f"  net_pnl={format_metric(ga_metrics['net_pnl'])}")
    print(f"  sharpe={format_metric(ga_metrics['sharpe'])}")
    print(f"profit_factor_delta={format_metric(payload['improvement']['profit_factor_delta'])}")
    print(f"sharpe_delta={format_metric(payload['improvement']['sharpe_delta'])}")
    print(f"saved={OUTPUT_DIR}")


def _select_best_threshold(train_scored: pd.DataFrame) -> float:
    best_threshold = THRESHOLDS[0]
    best_score = float("-inf")
    for threshold in THRESHOLDS:
        selected = train_scored.loc[train_scored["probability"] >= threshold]
        if selected.empty:
            continue
        realized = selected["realized_return"].astype(float)
        gross_profit = float(realized[realized > 0].sum())
        gross_loss = float(realized[realized < 0].sum())
        profit_factor = (gross_profit / abs(gross_loss)) if gross_loss < 0 else 0.0
        sharpe = _sharpe(realized.tolist())
        net_pnl = float(realized.sum())
        penalty = max(0, 20 - len(selected)) * 0.15
        score = (0.6 * profit_factor) + (0.8 * (sharpe or 0.0)) + (0.05 * net_pnl) - penalty
        if score > best_score:
            best_score = score
            best_threshold = threshold
    return best_threshold


def _split_train_validation(train_scored: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    ordered = train_scored.sort_values("setup_timestamp", kind="stable").reset_index(drop=True)
    split_idx = max(1, int(len(ordered) * 0.75))
    if split_idx >= len(ordered):
        split_idx = len(ordered) - 1
    return ordered.iloc[:split_idx].reset_index(drop=True), ordered.iloc[split_idx:].reset_index(drop=True)


def _threshold_only_genome(threshold: float) -> SetupQualityGenome:
    return SetupQualityGenome(
        probability_threshold=threshold,
        min_breakout_strength=0.0,
        min_abs_trend_spread=0.0,
        max_distance_to_vwap=1_000_000.0,
        min_relative_volume=0.0,
        min_range_width=0.0,
        max_range_width=1_000_000.0,
        min_rsi_center_distance=0.0,
    )


def _result_metrics(result, setups_detected: int) -> dict[str, object]:
    summary = calculate_summary_metrics(result.trades, result.equity_curve, 100000.0)
    return {
        "setups_detected": setups_detected,
        "trades_executed": result.total_trades,
        "win_rate": result.win_rate,
        "profit_factor": result.profit_factor,
        "net_pnl": result.net_pnl,
        "max_drawdown": result.max_drawdown,
        "sharpe": summary.get("sharpe"),
        "sortino": summary.get("sortino"),
        "calmar": summary.get("calmar"),
    }


def _sharpe(values: list[float]) -> float | None:
    if len(values) < 2:
        return None
    series = pd.Series(values, dtype="float64")
    std = float(series.std(ddof=1))
    if std == 0.0:
        return None
    return float((series.mean() / std) * (len(series) ** 0.5))


def _safe_delta(left: object, right: object) -> float | None:
    if left is None or right is None:
        return None
    return float(left) - float(right)


def format_metric(value: object) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


if __name__ == "__main__":
    main()

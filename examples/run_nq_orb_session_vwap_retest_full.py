from __future__ import annotations

import argparse
import json
import sys
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
from setups.orb_session_vwap_retest import ORBSessionVWAPRetestConfig, ORBSessionVWAPRetestDetector

INPUT_PATH = ROOT / "data" / "processed" / "nq_1min_2022_2025.parquet"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run 9:30-9:45 ORB with session-VWAP retest on the full dataset.")
    parser.add_argument("--target-mode", choices=["fixed_r", "liquidity"], default="fixed_r")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = ROOT / "outputs" / f"nq_orb_session_vwap_retest_full_{args.target_mode}"
    featured_path = output_dir / "featured_nq_1min_2022_2025.parquet"
    setups_path = output_dir / "setup_summary.csv"
    trades_path = output_dir / "trade_log.csv"
    metrics_path = output_dir / "metrics.json"
    output_dir.mkdir(parents=True, exist_ok=True)

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
            target_mode=args.target_mode,
        )
    )
    setups = detector.detect(featured)
    labeled = ForwardSetupLabeler(LabelerConfig(horizon_bars=20)).label(featured, setups)
    backtest_config = BacktestConfig()
    result = BarBacktestEngine(backtest_config).run(
        featured,
        setups,
        BacktestRunConfig(strategy_name=f"orb_session_vwap_retest_{args.target_mode}"),
    )
    summary = calculate_summary_metrics(result.trades, result.equity_curve, backtest_config.initial_capital)

    featured.to_parquet(featured_path, index=False)
    build_setup_summary(labeled).to_csv(setups_path, index=False)
    build_trade_log(result).to_csv(trades_path, index=False)
    metrics = {
        "rows": len(featured),
        "target_mode": args.target_mode,
        "setups_detected": len(setups),
        "trades_executed": result.total_trades,
        "win_rate": result.win_rate,
        "profit_factor": result.profit_factor,
        "net_pnl": result.net_pnl,
        "max_drawdown": result.max_drawdown,
        "sharpe": summary.get("sharpe"),
        "sortino": summary.get("sortino"),
        "calmar": summary.get("calmar"),
        "max_drawdown_pct": summary.get("max_drawdown_pct"),
        "total_return": summary.get("total_return"),
        "input_path": str(INPUT_PATH),
        "setup_summary_path": str(setups_path),
        "trade_log_path": str(trades_path),
    }
    metrics_path.write_text(json.dumps(metrics, indent=2, default=str), encoding="utf-8")

    print(f"rows={metrics['rows']}")
    print(f"target_mode={metrics['target_mode']}")
    print(f"setups_detected={metrics['setups_detected']}")
    print(f"trades_executed={metrics['trades_executed']}")
    print(f"win_rate={format_metric(metrics['win_rate'])}")
    print(f"profit_factor={format_metric(metrics['profit_factor'])}")
    print(f"net_pnl={format_metric(metrics['net_pnl'])}")
    print(f"max_drawdown={format_metric(metrics['max_drawdown'])}")
    print(f"max_drawdown_pct={format_metric(metrics['max_drawdown_pct'])}")
    print(f"total_return={format_metric(metrics['total_return'])}")
    print(f"sharpe={format_metric(metrics['sharpe'])}")
    print(f"sortino={format_metric(metrics['sortino'])}")
    print(f"calmar={format_metric(metrics['calmar'])}")
    print(f"saved={output_dir}")


def build_setup_summary(labeled_setups) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for item in labeled_setups:
        setup = item.setup
        rows.append(
            {
                "setup_id": setup.setup_id,
                "timestamp": setup.timestamp,
                "session_date": setup.session_date,
                "symbol": setup.symbol,
                "side": setup.direction.value,
                "entry_reference": setup.entry_reference,
                "stop_reference": setup.stop_reference,
                "target_reference": setup.target_reference,
                "retest_vwap_name": setup.context.get("retest_vwap_name"),
                "retest_vwap_price": setup.context.get("retest_vwap_price"),
                "target_name": setup.context.get("target_name"),
                "target_source": setup.context.get("target_source"),
                "label": item.label,
                "realized_return": item.realized_return,
                "realized_mae": item.realized_mae,
                "realized_mfe": item.realized_mfe,
            }
        )
    return pd.DataFrame(rows)


def build_trade_log(result) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for trade in result.trades:
        rows.append(
            {
                "trade_id": trade.trade_id,
                "setup_id": trade.setup_id,
                "entry_time": trade.entry_time,
                "exit_time": trade.exit_time,
                "symbol": trade.symbol,
                "side": trade.side.value,
                "entry_price": trade.entry_price,
                "exit_price": trade.exit_price,
                "pnl": trade.pnl,
                "bars_held": trade.bars_held,
                "exit_reason": trade.exit_reason,
            }
        )
    return pd.DataFrame(rows)


def format_metric(value: object) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


if __name__ == "__main__":
    main()

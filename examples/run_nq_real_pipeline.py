from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from backtest.engine import BarBacktestEngine, BacktestConfig, BacktestRunConfig
from config.models import StrategyConfig
from features.pipeline import build_feature_frame
from labeling.future_path import ForwardSetupLabeler, LabelerConfig
from setups.orb import ORBConfig, ORBSetupDetector

INPUT_PATH = ROOT / "data" / "processed" / "nq_1min_2022_2025.parquet"
OUTPUT_DIR = ROOT / "outputs" / "nq_am_displacement_orb_real_run"
FEATURED_PATH = OUTPUT_DIR / "featured_nq_1min_2022_2025.parquet"
AUDIT_PATH = OUTPUT_DIR / "candidate_audit.csv"
SETUPS_PATH = OUTPUT_DIR / "setup_summary.csv"
TRADES_PATH = OUTPUT_DIR / "trade_log.csv"
METRICS_PATH = OUTPUT_DIR / "metrics.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run nq_am_displacement_orb on processed NQ/MNQ parquet data.")
    parser.add_argument("--instrument", default="NQ", choices=["NQ", "MNQ"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    price_frame = pd.read_parquet(INPUT_PATH)
    featured = build_feature_frame(price_frame, opening_range_minutes=15)

    strategy_config = StrategyConfig(
        strategy_profile="nq_am_displacement_orb",
        instrument=args.instrument,
        latest_entry_time="11:30",
        target_rule="r_multiple",
        target_r_multiple=2.0,
        require_retest=False,
        runner_trail_rule=None,
        breakeven_enabled=False,
        breakeven_after_first_draw=True,
        partial_take_profit_enabled=False,
    )
    detector = ORBSetupDetector(ORBConfig.from_strategy_config(strategy_config))
    setups, diagnostics = detector.detect_with_diagnostics(featured)

    labeler = ForwardSetupLabeler(LabelerConfig(horizon_bars=20))
    labeled_setups = labeler.label(featured, setups)

    engine = BarBacktestEngine(BacktestConfig(position_size=2.0))
    result = engine.run(featured, setups, BacktestRunConfig(strategy_name="nq_am_displacement_orb"))

    featured.to_parquet(FEATURED_PATH, index=False)
    diagnostics.audit_frame.to_csv(AUDIT_PATH, index=False)
    build_setup_summary(labeled_setups).to_csv(SETUPS_PATH, index=False)
    build_trade_log(result).to_csv(TRADES_PATH, index=False)
    metrics = build_metrics_payload(featured, setups, diagnostics.counts, result, strategy_config)
    METRICS_PATH.write_text(json.dumps(metrics, indent=2, default=str), encoding="utf-8")

    rejection_counts = top_counts(diagnostics.audit_frame.get("rejection_reason", pd.Series(dtype=object)), exclude={"emitted"})
    exit_counts = top_counts(pd.Series([trade.exit_reason for trade in result.trades], dtype=object), exclude=set())

    print(f"rows={len(featured)}")
    print(f"candidate_ny_open_windows_scanned={diagnostics.counts.get('candidate_windows_scanned', 0)}")
    print(f"breakout_candidates={diagnostics.counts.get('breakout_candidates', 0)}")
    print(f"setups_emitted={len(setups)}")
    print(f"trades_taken={result.total_trades}")
    print(f"top_rejection_reasons={rejection_counts}")
    print(f"top_exit_reasons={exit_counts}")
    print(f"position_size_used=2.0")
    print(f"latest_entry_time_used={strategy_config.latest_entry_time}")
    print(f"target_logic_used={strategy_config.target_rule}_{strategy_config.target_r_multiple}R")
    print(f"win_rate={format_metric(result.win_rate)}")
    print(f"profit_factor={format_metric(result.profit_factor)}")
    print(f"net_pnl={format_metric(result.net_pnl)}")
    print(f"outputs={OUTPUT_DIR}")


def build_setup_summary(labeled_setups) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for labeled in labeled_setups:
        setup = labeled.setup
        rows.append(
            {
                "setup_id": setup.setup_id,
                "timestamp": setup.timestamp,
                "session_date": setup.session_date,
                "symbol": setup.symbol,
                "side": setup.direction.value,
                "setup_name": setup.setup_name,
                "entry_reference": setup.entry_reference,
                "stop_reference": setup.stop_reference,
                "target_reference": setup.target_reference,
                "rr_to_target": setup.context.get("rr_to_first_target"),
                "label": labeled.label,
                "realized_return": labeled.realized_return,
                "realized_mae": labeled.realized_mae,
                "realized_mfe": labeled.realized_mfe,
                "quality_bucket": labeled.quality_bucket,
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
                "size": trade.size,
                "pnl": trade.pnl,
                "return_pct": trade.return_pct,
                "bars_held": trade.bars_held,
                "exit_reason": trade.exit_reason,
                "setup_name": trade.setup_name,
                "stop_mode_used": trade.stop_mode_used,
            }
        )
    return pd.DataFrame(rows)


def build_metrics_payload(featured: pd.DataFrame, setups, diagnostics: dict[str, int], result, strategy_config: StrategyConfig) -> dict[str, object]:
    return {
        "input_path": str(INPUT_PATH),
        "featured_path": str(FEATURED_PATH),
        "candidate_audit_path": str(AUDIT_PATH),
        "setup_summary_path": str(SETUPS_PATH),
        "trade_log_path": str(TRADES_PATH),
        "rows": len(featured),
        "candidate_windows_scanned": diagnostics.get("candidate_windows_scanned", 0),
        "breakout_candidates": diagnostics.get("breakout_candidates", 0),
        "setups_found": len(setups),
        "trades_taken": result.total_trades,
        "position_size": 2.0,
        "latest_entry_time": strategy_config.latest_entry_time,
        "target_logic": f"{strategy_config.target_rule}_{strategy_config.target_r_multiple}R",
        "diagnostics": diagnostics,
        "backtest_metadata": result.metadata,
        "net_pnl": result.net_pnl,
        "gross_pnl": result.gross_pnl,
        "total_return": result.total_return,
        "win_rate": result.win_rate,
        "average_win": result.average_win,
        "average_loss": result.average_loss,
        "profit_factor": result.profit_factor,
        "expectancy": result.expectancy,
        "max_drawdown": result.max_drawdown,
        "sharpe_ratio": result.sharpe_ratio,
        "average_trade_duration": result.average_trade_duration,
    }


def top_counts(series: pd.Series, exclude: set[str]) -> dict[str, int]:
    cleaned = [str(value) for value in series.dropna().tolist() if str(value) not in exclude]
    return dict(Counter(cleaned).most_common(5))


def format_metric(value: object) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


if __name__ == "__main__":
    main()




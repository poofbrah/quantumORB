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
from config.models import StrategyConfig
from evaluation.metrics import calculate_summary_metrics
from setups.rp_profits_8am_orb import RPProfits8AMConfig, RPProfits8AMSetupDetector

INPUT_PATH = ROOT / "data" / "processed" / "nq_1min_2022_2025.parquet"
BASE_OUTPUT_DIR = ROOT / "outputs"
TRADE_WINDOW = "09:00-10:30"


def build_strategy_config(entry_mode: str) -> StrategyConfig:
    return StrategyConfig(
        strategy_profile="rp_profits_8am_orb",
        instrument="NQ",
        session_timezone="America/New_York",
        max_trades_per_session=1,
        allowed_trade_windows=(TRADE_WINDOW,),
        key_zone_start_time="08:00",
        key_zone_end_time="08:15",
        key_zone_min_width_points=None,
        key_zone_max_width_points=None,
        key_zone_min_width_atr=None,
        key_zone_max_width_atr=None,
        trade_window_required=True,
        strong_close_min_body_range_ratio=0.6,
        strong_close_min_boundary_distance=0.25,
        strong_close_min_boundary_distance_atr=0.5,
        rr_multiple=2.0,
        rp_entry_mode=entry_mode,
        rp_stop_mode="zone_boundary",
        rp_stop_buffer_points=0.0,
        rp_partial_at_r=1.0,
        rp_partial_fraction=0.5,
        rp_move_stop_to_breakeven_after_partial=True,
        rp_runner_target_r=None,
        rp_runner_trailing_stop_mode="atr",
        rp_runner_trailing_atr_multiple=1.5,
        reentry_body_ratio_min=0.5,
        reentry_min_close_distance_points=0.25,
        reentry_min_close_distance_atr=0.25,
        rp_reentry_stop_mode="signal_or_range",
        rp_reentry_stop_buffer_points=0.0,
        rp_reentry_max_entry_time="10:30",
        rp_reentry_tp1_mode="vwap_or_mid",
        rp_reentry_partial_fraction=0.5,
        rp_reentry_breakeven_after_tp1=True,
    )


def run_mode(entry_mode: str, frame: pd.DataFrame) -> tuple[dict[str, object], Path]:
    output_dir = BASE_OUTPUT_DIR / f"rp_profits_8am_orb_full_{entry_mode}"
    output_dir.mkdir(parents=True, exist_ok=True)

    strategy_config = build_strategy_config(entry_mode)
    detector = RPProfits8AMSetupDetector(RPProfits8AMConfig.from_strategy_config(strategy_config))
    setups, diagnostics = detector.detect_with_diagnostics(frame)
    result = BarBacktestEngine(BacktestConfig(position_size=2.0, commission_per_unit=1.40)).run(
        frame,
        setups,
        BacktestRunConfig(strategy_name=f"rp_profits_8am_orb_{entry_mode}"),
    )

    setup_map = {setup.setup_id: setup for setup in setups}
    trade_log = pd.DataFrame([trade_row(trade, setup_map.get(trade.setup_id)) for trade in result.trades])
    trade_log.to_csv(output_dir / "trade_log.csv", index=False)

    setup_summary = pd.DataFrame([setup_row(setup) for setup in setups])
    setup_summary.to_csv(output_dir / "setup_summary.csv", index=False)

    day_audit = diagnostics.audit_frame.copy()
    if not day_audit.empty:
        executed_days = {(pd.Timestamp(trade.trade_day), trade.symbol) for trade in result.trades if trade.trade_day is not None}
        day_audit["trade_executed"] = day_audit.apply(lambda row: (pd.Timestamp(row["trade_day"]), row["symbol"]) in executed_days, axis=1)
    day_audit.to_csv(output_dir / "day_audit.csv", index=False)

    metrics = calculate_summary_metrics(result.trades, result.equity_curve, 100000.0)
    payload = {
        "rows": len(frame),
        "entry_mode": entry_mode,
        "setups_detected": len(setups),
        "trades_executed": result.total_trades,
        "win_rate": metrics.get("win_rate"),
        "loss_rate": metrics.get("loss_rate"),
        "profit_factor": metrics.get("profit_factor"),
        "sharpe": metrics.get("sharpe"),
        "expectancy": metrics.get("expectancy"),
        "net_pnl": metrics.get("net_pnl"),
        "gross_profit": metrics.get("gross_profit"),
        "gross_loss": metrics.get("gross_loss"),
        "average_trade_pnl": metrics.get("average_trade_pnl"),
        "average_win": metrics.get("average_win"),
        "average_loss": metrics.get("average_loss"),
        "average_r": metrics.get("average_r"),
        "max_drawdown": metrics.get("max_drawdown"),
        "max_drawdown_pct": metrics.get("max_drawdown_pct"),
        "max_unrealized_profit": metrics.get("max_unrealized_profit"),
        "max_unrealized_loss": metrics.get("max_unrealized_loss"),
        "average_mfe": metrics.get("average_mfe"),
        "average_mae": metrics.get("average_mae"),
        "average_mfe_r": metrics.get("average_mfe_r"),
        "average_mae_r": metrics.get("average_mae_r"),
        "percent_partial_taken": metrics.get("percent_partial_taken"),
        "percent_target_exits": metrics.get("percent_target_exits"),
        "percent_stop_exits": metrics.get("percent_stop_exits"),
        "percent_trailing_stop_exits": metrics.get("percent_trailing_stop_exits"),
        "percent_session_end_exits": metrics.get("percent_session_end_exits"),
        "diagnostics": diagnostics.counts,
    }
    (output_dir / "metrics.json").write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    return payload, output_dir


def setup_row(setup) -> dict[str, object]:
    return {
        "setup_id": setup.setup_id,
        "trade_day": setup.context.get("trade_day"),
        "timestamp": setup.timestamp,
        "symbol": setup.symbol,
        "direction": setup.direction.value,
        "entry_family": setup.context.get("entry_family"),
        "entry_reference": setup.entry_reference,
        "stop_reference": setup.stop_reference,
        "target_reference": setup.target_reference,
        "range_width": setup.features.get("range_width"),
        "range_width_atr": setup.features.get("range_width_atr"),
        "displacement_strength": setup.features.get("displacement_strength"),
        "range_break_side": setup.context.get("range_break_side"),
        "tp1_price": setup.context.get("first_liquidity_target_price"),
        "runner_target_price": runner_target_from_setup(setup),
    }


def trade_row(trade, setup) -> dict[str, object]:
    features = setup.features if setup is not None else {}
    context = setup.context if setup is not None else {}
    return {
        "trade_id": trade.trade_id,
        "symbol": trade.symbol,
        "trade_day": trade.trade_day,
        "direction": trade.side.value,
        "entry_family": context.get("entry_family"),
        "setup_time": trade.setup_time,
        "entry_time": trade.entry_time,
        "exit_time": trade.exit_time,
        "entry_price": trade.entry_price,
        "initial_stop_price": trade.initial_stop_price,
        "final_stop_price": trade.final_stop_price,
        "partial_exit_price": trade.partial_exit_price,
        "partial_exit_time": trade.partial_exit_time,
        "partial_exit_fraction": trade.partial_exit_fraction,
        "runner_exit_price": trade.runner_exit_price,
        "runner_exit_time": trade.runner_exit_time,
        "exit_price_blended": trade.exit_price_blended,
        "pnl": trade.pnl,
        "pnl_r": trade.pnl_r,
        "bars_held": trade.bars_held,
        "exit_reason": trade.exit_reason,
        "partial_taken": trade.partial_taken,
        "moved_to_breakeven": trade.moved_to_breakeven,
        "trailing_stop_used": trade.trailing_stop_used,
        "max_favorable_excursion": trade.max_favorable_excursion,
        "max_adverse_excursion": trade.max_adverse_excursion,
        "max_favorable_excursion_r": trade.max_favorable_excursion_r,
        "max_adverse_excursion_r": trade.max_adverse_excursion_r,
        "max_unrealized_profit": trade.max_unrealized_profit,
        "max_unrealized_loss": trade.max_unrealized_loss,
        "range_width": features.get("range_width"),
        "range_width_atr": features.get("range_width_atr"),
        "displacement_strength": features.get("displacement_strength"),
        "vwap_at_entry": features.get("vwap_at_entry", context.get("vwap_at_entry")),
        "retracement_depth": features.get("retracement_depth"),
        "range_break_side": context.get("range_break_side"),
        "broke_outside_range": context.get("broke_outside_range"),
        "closed_back_inside": context.get("closed_back_inside"),
        "reentry_signal_time": context.get("reentry_signal_time"),
        "tp1_price": context.get("first_liquidity_target_price"),
        "runner_target_price": runner_target_from_setup(setup),
        "stop_anchor_used": context.get("stop_anchor_used"),
    }


def runner_target_from_setup(setup) -> float | None:
    if setup is None:
        return None
    runner_targets = setup.context.get("runner_targets", [])
    if runner_targets:
        return float(runner_targets[0]["price"])
    return None


def format_metric(value: object) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--entry-mode", choices=["displacement_vwap_pullback", "range_reentry_vwap", "hybrid"], default="displacement_vwap_pullback")
    args = parser.parse_args()

    frame = pd.read_parquet(INPUT_PATH).reset_index(drop=True)
    payload, output_dir = run_mode(args.entry_mode, frame)

    print(f"rows={payload['rows']}")
    print(f"setups_detected={payload['setups_detected']}")
    print(f"trades_executed={payload['trades_executed']}")
    print(f"win_rate={format_metric(payload['win_rate'])}")
    print(f"loss_rate={format_metric(payload['loss_rate'])}")
    print(f"profit_factor={format_metric(payload['profit_factor'])}")
    print(f"sharpe={format_metric(payload['sharpe'])}")
    print(f"expectancy={format_metric(payload['expectancy'])}")
    print(f"net_pnl={format_metric(payload['net_pnl'])}")
    print(f"max_drawdown={format_metric(payload['max_drawdown'])}")
    print(f"outputs={output_dir}")


if __name__ == "__main__":
    main()

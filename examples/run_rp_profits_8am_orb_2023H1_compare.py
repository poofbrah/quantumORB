from __future__ import annotations

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
from setups.rp_profits_8am_orb import RPProfits8AMConfig, RPProfits8AMSetupDetector

INPUT_PATH = ROOT / "data" / "processed" / "nq_1min_2022_2025.parquet"
OUTPUT_DIR = ROOT / "outputs" / "rp_profits_8am_orb_2023H1_compare"
START_DATE = pd.Timestamp("2023-01-01", tz="America/New_York")
END_DATE = pd.Timestamp("2023-06-30 23:59:59", tz="America/New_York")
ENTRY_MODES = ("midpoint_limit", "midpoint_bos_confirmation")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    price_frame = pd.read_parquet(INPUT_PATH)
    timestamps = pd.to_datetime(price_frame["timestamp"])
    filtered = price_frame.loc[(timestamps >= START_DATE) & (timestamps <= END_DATE)].reset_index(drop=True)

    summaries: list[dict[str, object]] = []
    for entry_mode in ENTRY_MODES:
        summary = run_mode(filtered, entry_mode)
        summaries.append(summary)
        save_outputs(summary)

    print(f"rows={len(filtered)}")
    print("")
    print("entry_mode                 setups_detected  trades_executed  win_rate  profit_factor  net_pnl  max_drawdown")
    for summary in summaries:
        print(
            f"{summary['entry_mode']:<26} {summary['setups_detected']:<15} {summary['trades_executed']:<16} "
            f"{format_metric(summary['win_rate']):<9} {format_metric(summary['profit_factor']):<14} "
            f"{format_metric(summary['net_pnl']):<8} {format_metric(summary['max_drawdown'])}"
        )
    print("")
    print(f"outputs={OUTPUT_DIR}")



def run_mode(frame: pd.DataFrame, entry_mode: str) -> dict[str, object]:
    strategy_config = StrategyConfig(
        strategy_profile="rp_profits_8am_orb",
        instrument="NQ",
        session_timezone="America/New_York",
        max_trades_per_session=1,
        allowed_trade_windows=("09:00-11:00",),
        key_zone_start_time="08:00",
        key_zone_end_time="08:15",
        trade_window_required=True,
        bias_requires_clean_break=False,
        stop_scan_low_points=5.0,
        stop_scan_high_points=10.0,
        stop_default_points=8.0,
        rr_multiple=4.0,
        rp_entry_mode=entry_mode,
        bos_lookback_bars=3,
        bos_require_same_direction_candle=True,
        bos_invalidate_on_opposite_zone_break=True,
    )

    detector = RPProfits8AMSetupDetector(RPProfits8AMConfig.from_strategy_config(strategy_config))
    setups, diagnostics = detector.detect_with_diagnostics(frame)
    result = BarBacktestEngine(BacktestConfig(position_size=2.0, commission_per_unit=1.40)).run(
        frame,
        setups,
        BacktestRunConfig(strategy_name=f"rp_profits_8am_orb_{entry_mode}_2023H1"),
    )

    return {
        "entry_mode": entry_mode,
        "rows": len(frame),
        "setups_detected": len(setups),
        "trades_executed": result.total_trades,
        "win_rate": result.win_rate,
        "profit_factor": result.profit_factor,
        "net_pnl": result.net_pnl,
        "max_drawdown": result.max_drawdown,
        "diagnostics": diagnostics.counts,
        "trades": [
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
            for trade in result.trades
        ],
    }



def save_outputs(summary: dict[str, object]) -> None:
    suffix = summary["entry_mode"]
    trades_path = OUTPUT_DIR / f"trade_log_{suffix}.csv"
    metrics_path = OUTPUT_DIR / f"metrics_{suffix}.json"

    pd.DataFrame(summary["trades"]).to_csv(trades_path, index=False)
    payload = dict(summary)
    payload.pop("trades", None)
    payload["trade_log_path"] = str(trades_path)
    metrics_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")



def format_metric(value: object) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


if __name__ == "__main__":
    main()

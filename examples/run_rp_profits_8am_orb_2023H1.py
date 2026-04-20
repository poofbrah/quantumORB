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
from labeling.future_path import ForwardSetupLabeler, LabelerConfig
from setups.rp_profits_8am_orb import RPProfits8AMConfig, RPProfits8AMSetupDetector

INPUT_PATH = ROOT / "data" / "processed" / "nq_1min_2022_2025.parquet"
OUTPUT_DIR = ROOT / "outputs" / "rp_profits_8am_orb_2023H1"
SETUPS_PATH = OUTPUT_DIR / "setup_summary.csv"
TRADES_PATH = OUTPUT_DIR / "trade_log.csv"
METRICS_PATH = OUTPUT_DIR / "metrics.json"
START_DATE = pd.Timestamp("2023-01-01", tz="America/New_York")
END_DATE = pd.Timestamp("2023-06-30 23:59:59", tz="America/New_York")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    price_frame = pd.read_parquet(INPUT_PATH)
    timestamps = pd.to_datetime(price_frame["timestamp"])
    filtered = price_frame.loc[(timestamps >= START_DATE) & (timestamps <= END_DATE)].reset_index(drop=True)

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
    )

    detector = RPProfits8AMSetupDetector(RPProfits8AMConfig.from_strategy_config(strategy_config))
    setups, diagnostics = detector.detect_with_diagnostics(filtered)
    labeled = ForwardSetupLabeler(LabelerConfig(horizon_bars=60)).label(filtered, setups)
    result = BarBacktestEngine(BacktestConfig(position_size=2.0, commission_per_unit=1.40)).run(
        filtered,
        setups,
        BacktestRunConfig(strategy_name="rp_profits_8am_orb_2023H1"),
    )

    build_setup_summary(labeled).to_csv(SETUPS_PATH, index=False)
    build_trade_log(result).to_csv(TRADES_PATH, index=False)
    payload = {
        "rows": len(filtered),
        "setups_detected": len(setups),
        "trades_executed": result.total_trades,
        "win_rate": result.win_rate,
        "profit_factor": result.profit_factor,
        "net_pnl": result.net_pnl,
        "max_drawdown": result.max_drawdown,
        "diagnostics": diagnostics.counts,
        "setup_summary_path": str(SETUPS_PATH),
        "trade_log_path": str(TRADES_PATH),
    }
    METRICS_PATH.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")

    print(f"rows={len(filtered)}")
    print(f"setups_detected={len(setups)}")
    print(f"trades_executed={result.total_trades}")
    print(f"win_rate={format_metric(result.win_rate)}")
    print(f"profit_factor={format_metric(result.profit_factor)}")
    print(f"net_pnl={format_metric(result.net_pnl)}")
    print(f"max_drawdown={format_metric(result.max_drawdown)}")
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
                "entry_reference": setup.entry_reference,
                "stop_reference": setup.stop_reference,
                "target_reference": setup.target_reference,
                "stop_anchor_used": setup.context.get("stop_anchor_used"),
                "label": labeled.label,
                "realized_return": labeled.realized_return,
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


def format_metric(value: object) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


if __name__ == "__main__":
    main()

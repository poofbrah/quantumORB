from __future__ import annotations

import json
import sys
import time
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
TRADES_PATH_RR1 = ROOT / "outputs" / "debug_trades_2023H1_rr1.csv"
METRICS_PATH_RR1 = ROOT / "outputs" / "debug_metrics_2023H1_rr1.json"

START_DATE = pd.Timestamp("2023-01-01", tz="America/New_York")
END_DATE = pd.Timestamp("2023-06-30 23:59:59", tz="America/New_York")


def main() -> None:
    overall_start = time.perf_counter()

    print("[1/4] Loading parquet...")
    price_frame = pd.read_parquet(INPUT_PATH)

    print("[2/4] Filtering 2023H1 window...")
    timestamps = pd.to_datetime(price_frame["timestamp"])
    filtered = price_frame.loc[(timestamps >= START_DATE) & (timestamps <= END_DATE)].reset_index(drop=True)

    print("[3/4] Building features...")
    featured = build_feature_frame(filtered, opening_range_minutes=15)

    print("[4/4] Running 2R vs 1R comparison...")
    baseline = run_variant(featured, target_r_multiple=2.0, strategy_name="nq_am_displacement_orb_2023H1_debug_rr2")
    rr1 = run_variant(featured, target_r_multiple=1.0, strategy_name="nq_am_displacement_orb_2023H1_debug_rr1")

    TRADES_PATH_RR1.parent.mkdir(parents=True, exist_ok=True)
    build_trade_log(rr1["result"]).to_csv(TRADES_PATH_RR1, index=False)
    METRICS_PATH_RR1.write_text(
        json.dumps(build_metrics_payload(rr1, rows_loaded=len(price_frame), rows_after_filter=len(filtered)), indent=2, default=str),
        encoding="utf-8",
    )

    print_summary(len(price_frame), len(filtered), baseline, rr1, time.perf_counter() - overall_start)


def run_variant(featured: pd.DataFrame, target_r_multiple: float, strategy_name: str) -> dict[str, object]:
    strategy_config = StrategyConfig(
        strategy_profile="nq_am_displacement_orb",
        instrument="NQ",
        latest_entry_time="11:30",
        target_rule="r_multiple",
        target_r_multiple=target_r_multiple,
        require_retest=False,
        runner_trail_rule=None,
        breakeven_enabled=False,
        breakeven_after_first_draw=True,
        partial_take_profit_enabled=False,
    )

    detector = ORBSetupDetector(ORBConfig.from_strategy_config(strategy_config))
    setups, diagnostics = detector.detect_with_diagnostics(featured)
    labeled = ForwardSetupLabeler(LabelerConfig(horizon_bars=20)).label(featured, setups)
    result = BarBacktestEngine(BacktestConfig(position_size=2.0)).run(
        featured,
        setups,
        BacktestRunConfig(strategy_name=strategy_name),
    )
    return {
        "target_r_multiple": target_r_multiple,
        "setups": setups,
        "diagnostics": diagnostics,
        "labeled": labeled,
        "result": result,
    }


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
            }
        )
    return pd.DataFrame(rows)


def build_metrics_payload(variant: dict[str, object], rows_loaded: int, rows_after_filter: int) -> dict[str, object]:
    result = variant["result"]
    return {
        "input_path": str(INPUT_PATH),
        "trades_path": str(TRADES_PATH_RR1),
        "rows_loaded": rows_loaded,
        "rows_after_filter": rows_after_filter,
        "target_r_multiple": variant["target_r_multiple"],
        "setups_detected": len(variant["setups"]),
        "trades_executed": result.total_trades,
        "win_rate": result.win_rate,
        "profit_factor": result.profit_factor,
        "net_pnl": result.net_pnl,
        "max_drawdown": result.max_drawdown,
    }


def print_summary(rows_loaded: int, rows_after_filter: int, baseline: dict[str, object], rr1: dict[str, object], total_time: float) -> None:
    base_result = baseline["result"]
    rr1_result = rr1["result"]
    print(f"rows_loaded={rows_loaded}")
    print(f"rows_after_filter={rows_after_filter}")
    print("comparison=rr2_vs_rr1")
    print("metric,rr2,rr1")
    print(f"setups_detected,{len(baseline['setups'])},{len(rr1['setups'])}")
    print(f"trades_executed,{base_result.total_trades},{rr1_result.total_trades}")
    print(f"win_rate,{format_metric(base_result.win_rate)},{format_metric(rr1_result.win_rate)}")
    print(f"profit_factor,{format_metric(base_result.profit_factor)},{format_metric(rr1_result.profit_factor)}")
    print(f"net_pnl,{format_metric(base_result.net_pnl)},{format_metric(rr1_result.net_pnl)}")
    print(f"max_drawdown,{format_metric(base_result.max_drawdown)},{format_metric(rr1_result.max_drawdown)}")
    print(f"total_time={total_time:.3f}s")
    print(f"saved_trades_rr1={TRADES_PATH_RR1}")
    print(f"saved_metrics_rr1={METRICS_PATH_RR1}")


def format_metric(value: object) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


if __name__ == "__main__":
    main()




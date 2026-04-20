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
OUTPUT_DIR = ROOT / "outputs" / "rp_profits_8am_orb_bos_window_compare_2023H1"
START_DATE = pd.Timestamp("2023-01-01", tz="America/New_York")
END_DATE = pd.Timestamp("2023-06-30 23:59:59", tz="America/New_York")
WINDOWS = (
    "08:15-09:30",
    "08:30-10:00",
    "09:00-10:30",
    "08:15-10:30",
)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    frame = pd.read_parquet(INPUT_PATH)
    timestamps = pd.to_datetime(frame["timestamp"])
    filtered = frame.loc[(timestamps >= START_DATE) & (timestamps <= END_DATE)].reset_index(drop=True)

    summaries: list[dict[str, object]] = []
    for window in WINDOWS:
        summary = run_window(filtered, window)
        summaries.append(summary)

    save_outputs(summaries)
    print_summary(len(filtered), summaries)



def run_window(frame: pd.DataFrame, window: str) -> dict[str, object]:
    strategy_config = StrategyConfig(
        strategy_profile="rp_profits_8am_orb",
        instrument="NQ",
        session_timezone="America/New_York",
        max_trades_per_session=1,
        allowed_trade_windows=(window,),
        key_zone_start_time="08:00",
        key_zone_end_time="08:15",
        trade_window_required=True,
        bias_requires_clean_break=False,
        stop_scan_low_points=5.0,
        stop_scan_high_points=10.0,
        stop_default_points=8.0,
        rr_multiple=4.0,
        rp_entry_mode="midpoint_bos_confirmation",
        bos_lookback_bars=3,
        bos_require_same_direction_candle=True,
        bos_invalidate_on_opposite_zone_break=True,
    )

    detector = RPProfits8AMSetupDetector(RPProfits8AMConfig.from_strategy_config(strategy_config))
    setups, diagnostics = detector.detect_with_diagnostics(frame)
    result = BarBacktestEngine(BacktestConfig(position_size=2.0, commission_per_unit=1.40)).run(
        frame,
        setups,
        BacktestRunConfig(strategy_name=f"rp_profits_8am_orb_bos_{window.replace(':', '').replace('-', '_')}_2023H1"),
    )

    return {
        "window": window,
        "setups_detected": len(setups),
        "trades_executed": result.total_trades,
        "win_rate": result.win_rate,
        "profit_factor": result.profit_factor,
        "net_pnl": result.net_pnl,
        "max_drawdown": result.max_drawdown,
        "diagnostics": diagnostics.counts,
    }



def save_outputs(summaries: list[dict[str, object]]) -> None:
    summary_frame = pd.DataFrame(
        [
            {
                "window": summary["window"],
                "setups_detected": summary["setups_detected"],
                "trades_executed": summary["trades_executed"],
                "win_rate": summary["win_rate"],
                "profit_factor": summary["profit_factor"],
                "net_pnl": summary["net_pnl"],
                "max_drawdown": summary["max_drawdown"],
            }
            for summary in summaries
        ]
    )
    summary_frame.to_csv(OUTPUT_DIR / "window_comparison.csv", index=False)
    (OUTPUT_DIR / "window_comparison.json").write_text(json.dumps(summaries, indent=2, default=str), encoding="utf-8")



def print_summary(rows: int, summaries: list[dict[str, object]]) -> None:
    print(f"rows={rows}")
    print("")
    print("window       setups_detected  trades_executed  win_rate  profit_factor  net_pnl  max_drawdown")
    for summary in summaries:
        print(
            f"{summary['window']:<12} {summary['setups_detected']:<15} {summary['trades_executed']:<16} "
            f"{format_metric(summary['win_rate']):<9} {format_metric(summary['profit_factor']):<14} "
            f"{format_metric(summary['net_pnl']):<8} {format_metric(summary['max_drawdown'])}"
        )



def format_metric(value: object) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


if __name__ == "__main__":
    main()

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
OUTPUT_DIR = ROOT / "outputs" / "rp_profits_8am_orb_bos_diagnostics_full_range_10_15"
TRADE_WINDOW = "09:00-10:30"


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    frame = pd.read_parquet(INPUT_PATH).reset_index(drop=True)
    strategy_config = StrategyConfig(
        strategy_profile="rp_profits_8am_orb",
        instrument="NQ",
        session_timezone="America/New_York",
        max_trades_per_session=1,
        allowed_trade_windows=(TRADE_WINDOW,),
        key_zone_start_time="08:00",
        key_zone_end_time="08:15",
        key_zone_min_width_points=10.0,
        key_zone_max_width_points=15.0,
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
        BacktestRunConfig(strategy_name="rp_profits_8am_orb_bos_diagnostics_full_range_10_15"),
    )

    audit = diagnostics.audit_frame.sort_values(["trade_day", "symbol"], kind="stable") if not diagnostics.audit_frame.empty else diagnostics.audit_frame
    if not audit.empty:
        rejection_summary = (
            audit["rejection_reason"].fillna("none").value_counts().rename_axis("rejection_reason").reset_index(name="days"))
        setup_days = int(audit["setup_emitted"].sum())
    else:
        rejection_summary = pd.DataFrame(columns=["rejection_reason", "days"])
        setup_days = 0

    audit_path = OUTPUT_DIR / "day_audit.csv"
    rejection_path = OUTPUT_DIR / "rejection_summary.csv"
    metrics_path = OUTPUT_DIR / "diagnostics_summary.json"

    audit.to_csv(audit_path, index=False)
    rejection_summary.to_csv(rejection_path, index=False)

    payload = {
        "rows": len(frame),
        "trade_window": TRADE_WINDOW,
        "setups_detected": len(setups),
        "trades_executed": result.total_trades,
        "win_rate": result.win_rate,
        "profit_factor": result.profit_factor,
        "net_pnl": result.net_pnl,
        "max_drawdown": result.max_drawdown,
        "diagnostics": diagnostics.counts,
        "setup_days": setup_days,
        "audit_path": str(audit_path),
        "rejection_summary_path": str(rejection_path),
    }
    metrics_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")

    print(f"rows={len(frame)}")
    print(f"trade_window={TRADE_WINDOW}")
    print(f"setups_detected={len(setups)}")
    print(f"trades_executed={result.total_trades}")
    print(f"win_rate={format_metric(result.win_rate)}")
    print(f"profit_factor={format_metric(result.profit_factor)}")
    print(f"net_pnl={format_metric(result.net_pnl)}")
    print(f"max_drawdown={format_metric(result.max_drawdown)}")
    print("")
    print("diagnostic_counts")
    for key, value in diagnostics.counts.items():
        print(f"{key}={value}")
    print("")
    print("top_rejection_reasons")
    if rejection_summary.empty:
        print("none")
    else:
        for _, row in rejection_summary.head(10).iterrows():
            print(f"{row['rejection_reason']}={int(row['days'])}")
    print("")
    print(f"outputs={OUTPUT_DIR}")



def format_metric(value: object) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


if __name__ == "__main__":
    main()




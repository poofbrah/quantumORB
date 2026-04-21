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
from data.preprocess import filter_session_hours
from evaluation.metrics import calculate_summary_metrics
from features.pipeline import build_feature_frame
from setups.orb_session_vwap_retest import ORBSessionVWAPRetestConfig, ORBSessionVWAPRetestDetector

INPUT_PATH = ROOT / "data" / "processed" / "nq_1min_2022_2025.parquet"
OUTPUT_DIR = ROOT / "outputs" / "orb_regime_mode_compare"
MODES = ("continuation", "range_reversion", "hybrid")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    frame = pd.read_parquet(INPUT_PATH)
    rth_frame = filter_session_hours(frame, session_start="09:30", session_end="16:00")
    featured = build_feature_frame(rth_frame, opening_range_minutes=15)

    rows: list[dict[str, object]] = []
    for mode in MODES:
        detector = ORBSessionVWAPRetestDetector(
            ORBSessionVWAPRetestConfig(
                opening_range_start_time="09:30",
                opening_range_end_time="09:45",
                opening_range_minutes=15,
                latest_entry_time="11:30",
                allowed_trade_windows=("09:45-11:30",),
                target_r_multiple=2.0,
                target_mode="liquidity",
                require_trend_alignment=(mode == "continuation"),
                entry_family_mode=mode,
            )
        )
        setups = detector.detect(featured)
        result = BarBacktestEngine(BacktestConfig()).run(
            featured,
            setups,
            BacktestRunConfig(strategy_name=f"orb_regime_{mode}"),
        )
        summary = calculate_summary_metrics(result.trades, result.equity_curve, 100000.0)
        rows.append(
            {
                "mode": mode,
                "rows": len(featured),
                "setups_detected": len(setups),
                "trades_executed": result.total_trades,
                "win_rate": result.win_rate,
                "profit_factor": result.profit_factor,
                "net_pnl": result.net_pnl,
                "max_drawdown": result.max_drawdown,
                "sharpe": summary.get("sharpe"),
                "sortino": summary.get("sortino"),
                "calmar": summary.get("calmar"),
            }
        )

    comparison = pd.DataFrame(rows)
    comparison.to_csv(OUTPUT_DIR / "mode_comparison.csv", index=False)
    (OUTPUT_DIR / "mode_comparison.json").write_text(json.dumps(rows, indent=2, default=str), encoding="utf-8")

    print(f"rows={len(featured)}")
    for row in rows:
        print(f"mode={row['mode']}")
        print(f"  setups_detected={row['setups_detected']}")
        print(f"  trades_executed={row['trades_executed']}")
        print(f"  win_rate={format_metric(row['win_rate'])}")
        print(f"  profit_factor={format_metric(row['profit_factor'])}")
        print(f"  net_pnl={format_metric(row['net_pnl'])}")
        print(f"  sharpe={format_metric(row['sharpe'])}")
    print(f"saved={OUTPUT_DIR}")


def format_metric(value: object) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


if __name__ == "__main__":
    main()

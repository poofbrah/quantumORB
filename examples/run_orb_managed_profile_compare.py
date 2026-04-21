from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
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
OUTPUT_DIR = ROOT / "outputs" / "orb_managed_profile_compare"


def build_variants() -> list[tuple[str, ORBSessionVWAPRetestConfig]]:
    base_kwargs = {
        "opening_range_start_time": "09:30",
        "opening_range_end_time": "09:45",
        "opening_range_minutes": 15,
        "latest_entry_time": "11:30",
        "allowed_trade_windows": ("09:45-11:30",),
        "target_mode": "liquidity",
        "target_r_multiple": 2.0,
        "liquidity_target_priority": ("pdh", "pdl", "h4_high", "h4_low", "day_high", "day_low", "london_high", "london_low"),
    }
    return [
        (
            "continuation_baseline",
            ORBSessionVWAPRetestConfig(
                require_trend_alignment=True,
                entry_family_mode="continuation",
                managed_profile_enabled=False,
                **base_kwargs,
            ),
        ),
        (
            "continuation_managed",
            ORBSessionVWAPRetestConfig(
                require_trend_alignment=True,
                entry_family_mode="continuation",
                managed_profile_enabled=True,
                partial_take_profit_r_multiple=1.0,
                partial_take_profit_fraction=0.5,
                managed_base_target_r_multiple=2.0,
                enable_runner_targets=True,
                enable_runner_trailing=True,
                runner_trail_atr_multiple=1.0,
                **base_kwargs,
            ),
        ),
        (
            "hybrid_baseline",
            ORBSessionVWAPRetestConfig(
                require_trend_alignment=False,
                entry_family_mode="hybrid",
                managed_profile_enabled=False,
                **base_kwargs,
            ),
        ),
        (
            "hybrid_managed",
            ORBSessionVWAPRetestConfig(
                require_trend_alignment=False,
                entry_family_mode="hybrid",
                managed_profile_enabled=True,
                partial_take_profit_r_multiple=1.0,
                partial_take_profit_fraction=0.5,
                managed_base_target_r_multiple=2.0,
                enable_runner_targets=True,
                enable_runner_trailing=True,
                runner_trail_atr_multiple=1.0,
                **base_kwargs,
            ),
        ),
    ]


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    frame = pd.read_parquet(INPUT_PATH)
    rth_frame = filter_session_hours(frame, session_start="09:30", session_end="16:00")
    featured = build_feature_frame(rth_frame, opening_range_minutes=15)

    rows: list[dict[str, object]] = []
    for name, config in build_variants():
        detector = ORBSessionVWAPRetestDetector(config)
        setups = detector.detect(featured)
        result = BarBacktestEngine(BacktestConfig(initial_capital=100000.0)).run(
            featured,
            setups,
            BacktestRunConfig(strategy_name=name),
        )
        summary = calculate_summary_metrics(result.trades, result.equity_curve, 100000.0)
        rows.append(
            {
                "variant": name,
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
                "percent_partial_taken": summary.get("percent_partial_taken"),
                "percent_target_exits": summary.get("percent_target_exits"),
                "percent_stop_exits": summary.get("percent_stop_exits"),
                "percent_trailing_stop_exits": summary.get("percent_trailing_stop_exits"),
                "percent_session_end_exits": summary.get("percent_session_end_exits"),
            }
        )

    comparison = pd.DataFrame(rows)
    comparison.to_csv(OUTPUT_DIR / "managed_profile_comparison.csv", index=False)
    (OUTPUT_DIR / "managed_profile_comparison.json").write_text(
        json.dumps(rows, indent=2, default=str),
        encoding="utf-8",
    )
    render_chart(comparison)

    print(f"rows={len(featured)}")
    print(comparison.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print(f"saved={OUTPUT_DIR}")


def render_chart(comparison: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    variants = comparison["variant"].tolist()

    axes[0, 0].bar(variants, comparison["net_pnl"], color="#2878B5")
    axes[0, 0].set_title("Net PnL")
    axes[0, 0].tick_params(axis="x", rotation=25)

    axes[0, 1].bar(variants, comparison["profit_factor"], color="#9ACD32")
    axes[0, 1].set_title("Profit Factor")
    axes[0, 1].tick_params(axis="x", rotation=25)

    axes[1, 0].bar(variants, comparison["sharpe"], color="#FF8C42")
    axes[1, 0].set_title("Sharpe")
    axes[1, 0].tick_params(axis="x", rotation=25)

    axes[1, 1].bar(variants, comparison["trades_executed"], color="#7D5BA6")
    axes[1, 1].set_title("Trades Executed")
    axes[1, 1].tick_params(axis="x", rotation=25)

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "managed_profile_comparison.png", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    main()

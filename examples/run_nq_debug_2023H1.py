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
FEATURED_PATH = ROOT / "data" / "processed" / "nq_1min_features_2023H1.parquet"
SETUPS_PATH = ROOT / "outputs" / "debug_setups_2023H1.csv"
TRADES_PATH = ROOT / "outputs" / "debug_trades_2023H1.csv"
METRICS_PATH = ROOT / "outputs" / "debug_metrics_2023H1.json"

START_DATE = pd.Timestamp("2023-01-01", tz="America/New_York")
END_DATE = pd.Timestamp("2023-06-30 23:59:59", tz="America/New_York")


def main() -> None:
    overall_start = time.perf_counter()

    print("[1/6] Loading parquet...")
    load_start = time.perf_counter()
    price_frame = pd.read_parquet(INPUT_PATH)
    load_time = time.perf_counter() - load_start
    rows_loaded = len(price_frame)

    print("[2/6] Filtering 2023H1 window...")
    filter_start = time.perf_counter()
    timestamps = pd.to_datetime(price_frame["timestamp"])
    filtered = price_frame.loc[(timestamps >= START_DATE) & (timestamps <= END_DATE)].reset_index(drop=True)
    filter_time = time.perf_counter() - filter_start
    rows_after_filter = len(filtered)

    print("[3/6] Building features...")
    feature_start = time.perf_counter()
    featured = build_feature_frame(filtered, opening_range_minutes=15)
    feature_time = time.perf_counter() - feature_start

    print("[4/6] Detecting setups...")
    setup_start = time.perf_counter()
    strategy_config = StrategyConfig(
        strategy_profile="nq_am_displacement_orb",
        instrument="NQ",
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
    setup_time = time.perf_counter() - setup_start

    print("[5/6] Labeling setups...")
    labeling_start = time.perf_counter()
    labeler = ForwardSetupLabeler(LabelerConfig(horizon_bars=20))
    labeled_setups = labeler.label(featured, setups)
    labeling_time = time.perf_counter() - labeling_start

    print("[6/6] Running backtest...")
    backtest_start = time.perf_counter()
    engine = BarBacktestEngine(BacktestConfig(position_size=2.0))
    result = engine.run(featured, setups, BacktestRunConfig(strategy_name="nq_am_displacement_orb_2023H1_debug"))
    backtest_time = time.perf_counter() - backtest_start

    total_time = time.perf_counter() - overall_start

    FEATURED_PATH.parent.mkdir(parents=True, exist_ok=True)
    SETUPS_PATH.parent.mkdir(parents=True, exist_ok=True)
    featured.to_parquet(FEATURED_PATH, index=False)
    build_setup_summary(labeled_setups, diagnostics.audit_frame).to_csv(SETUPS_PATH, index=False)
    build_trade_log(result).to_csv(TRADES_PATH, index=False)
    metrics = build_metrics_payload(
        rows_loaded=rows_loaded,
        rows_after_filter=rows_after_filter,
        result=result,
        setups_detected=len(setups),
        timings={
            "load_time": load_time,
            "filter_time": filter_time,
            "feature_time": feature_time,
            "setup_detection_time": setup_time,
            "labeling_time": labeling_time,
            "backtest_time": backtest_time,
            "total_time": total_time,
        },
    )
    METRICS_PATH.write_text(json.dumps(metrics, indent=2, default=str), encoding="utf-8")

    timings = {
        "load": load_time,
        "filter": filter_time,
        "feature": feature_time,
        "setup_detection": setup_time,
        "labeling": labeling_time,
        "backtest": backtest_time,
    }
    slowest_stage = max(timings, key=timings.get)

    print(f"rows_loaded={rows_loaded}")
    print(f"rows_after_filter={rows_after_filter}")
    print(f"setups_detected={len(setups)}")
    print(f"trades_executed={result.total_trades}")
    print(f"win_rate={format_metric(result.win_rate)}")
    print(f"profit_factor={format_metric(result.profit_factor)}")
    print(f"net_pnl={format_metric(result.net_pnl)}")
    print(f"max_drawdown={format_metric(result.max_drawdown)}")
    print(f"load_time={load_time:.3f}s")
    print(f"filter_time={filter_time:.3f}s")
    print(f"feature_time={feature_time:.3f}s")
    print(f"setup_detection_time={setup_time:.3f}s")
    print(f"labeling_time={labeling_time:.3f}s")
    print(f"backtest_time={backtest_time:.3f}s")
    print(f"total_time={total_time:.3f}s")
    print(f"slowest_stage={slowest_stage} ({timings[slowest_stage]:.3f}s)")


def build_setup_summary(labeled_setups, audit_frame: pd.DataFrame) -> pd.DataFrame:
    audit_columns = [
        "timestamp",
        "session_date",
        "breakout_side",
        "body_close_outside_or",
        "latest_entry_pass",
        "strong_close_pass",
        "retest_pass",
        "estimated_rr",
        "emitted_setup",
        "rejection_reason",
    ]
    audit_subset = audit_frame[audit_columns].copy() if not audit_frame.empty else pd.DataFrame(columns=audit_columns)
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
                "label": labeled.label,
                "realized_return": labeled.realized_return,
                "quality_bucket": labeled.quality_bucket,
            }
        )
    summary = pd.DataFrame(rows)
    if summary.empty or audit_subset.empty:
        return summary
    return summary.merge(audit_subset, on=["timestamp", "session_date"], how="left")


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


def build_metrics_payload(rows_loaded: int, rows_after_filter: int, result, setups_detected: int, timings: dict[str, float]) -> dict[str, object]:
    return {
        "input_path": str(INPUT_PATH),
        "featured_path": str(FEATURED_PATH),
        "setups_path": str(SETUPS_PATH),
        "trades_path": str(TRADES_PATH),
        "rows_loaded": rows_loaded,
        "rows_after_filter": rows_after_filter,
        "setups_detected": setups_detected,
        "trades_executed": result.total_trades,
        "win_rate": result.win_rate,
        "profit_factor": result.profit_factor,
        "net_pnl": result.net_pnl,
        "max_drawdown": result.max_drawdown,
        **timings,
    }


def format_metric(value: object) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


if __name__ == "__main__":
    main()




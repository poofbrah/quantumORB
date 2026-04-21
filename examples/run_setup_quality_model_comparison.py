from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from reporting import save_model_summary_table_image, save_threshold_metric_chart

BASELINE_DIR = ROOT / "outputs" / "setup_quality_research_orb_session_vwap_retest_liquidity"
LSTM_DIR = ROOT / "outputs" / "setup_quality_lstm_orb_session_vwap_retest_liquidity"
OUTPUT_DIR = ROOT / "outputs" / "setup_quality_model_comparison"


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    figures_dir = OUTPUT_DIR / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    baseline_summary = json.loads((BASELINE_DIR / "research_summary.json").read_text(encoding="utf-8"))
    lstm_summary = json.loads((LSTM_DIR / "lstm_summary.json").read_text(encoding="utf-8"))
    baseline_thresholds = pd.read_csv(BASELINE_DIR / "threshold_sweep.csv")
    lstm_thresholds = pd.read_csv(LSTM_DIR / "threshold_sweep.csv")

    comparison_rows = [
        {
            "model_name": "logistic_regression",
            "average_roc_auc": _find_model_value(baseline_summary["model_summaries"], "logistic_regression", "average_roc_auc"),
            "average_log_loss": _find_model_value(baseline_summary["model_summaries"], "logistic_regression", "average_log_loss"),
            "average_brier_score": _find_model_value(baseline_summary["model_summaries"], "logistic_regression", "average_brier_score"),
            "selected_setups": _find_model_value(baseline_summary["model_summaries"], "logistic_regression", "selected_setups"),
            "selected_win_rate": _find_model_value(baseline_summary["model_summaries"], "logistic_regression", "selected_win_rate"),
            "selected_average_r": _find_model_value(baseline_summary["model_summaries"], "logistic_regression", "selected_average_r"),
            "selected_profit_factor": _find_model_value(baseline_summary["model_summaries"], "logistic_regression", "selected_profit_factor"),
            "best_threshold": baseline_summary["best_thresholds"]["logistic_regression"]["best_threshold"],
            "best_trades": baseline_summary["best_thresholds"]["logistic_regression"]["trades_executed"],
            "best_win_rate": baseline_summary["best_thresholds"]["logistic_regression"]["win_rate"],
            "best_profit_factor": baseline_summary["best_thresholds"]["logistic_regression"]["profit_factor"],
            "best_net_pnl": baseline_summary["best_thresholds"]["logistic_regression"]["net_pnl"],
            "best_sharpe": baseline_summary["best_thresholds"]["logistic_regression"]["sharpe"],
        },
        {
            "model_name": "gradient_boosting",
            "average_roc_auc": _find_model_value(baseline_summary["model_summaries"], "gradient_boosting", "average_roc_auc"),
            "average_log_loss": _find_model_value(baseline_summary["model_summaries"], "gradient_boosting", "average_log_loss"),
            "average_brier_score": _find_model_value(baseline_summary["model_summaries"], "gradient_boosting", "average_brier_score"),
            "selected_setups": _find_model_value(baseline_summary["model_summaries"], "gradient_boosting", "selected_setups"),
            "selected_win_rate": _find_model_value(baseline_summary["model_summaries"], "gradient_boosting", "selected_win_rate"),
            "selected_average_r": _find_model_value(baseline_summary["model_summaries"], "gradient_boosting", "selected_average_r"),
            "selected_profit_factor": _find_model_value(baseline_summary["model_summaries"], "gradient_boosting", "selected_profit_factor"),
            "best_threshold": baseline_summary["best_thresholds"]["gradient_boosting"]["best_threshold"],
            "best_trades": baseline_summary["best_thresholds"]["gradient_boosting"]["trades_executed"],
            "best_win_rate": baseline_summary["best_thresholds"]["gradient_boosting"]["win_rate"],
            "best_profit_factor": baseline_summary["best_thresholds"]["gradient_boosting"]["profit_factor"],
            "best_net_pnl": baseline_summary["best_thresholds"]["gradient_boosting"]["net_pnl"],
            "best_sharpe": baseline_summary["best_thresholds"]["gradient_boosting"]["sharpe"],
        },
        {
            "model_name": "lstm",
            "average_roc_auc": lstm_summary["summary"]["average_roc_auc"],
            "average_log_loss": lstm_summary["summary"]["average_log_loss"],
            "average_brier_score": lstm_summary["summary"]["average_brier_score"],
            "selected_setups": lstm_summary["summary"]["selected_setups"],
            "selected_win_rate": lstm_summary["summary"]["selected_win_rate"],
            "selected_average_r": lstm_summary["summary"]["selected_average_r"],
            "selected_profit_factor": lstm_summary["summary"]["selected_profit_factor"],
            "best_threshold": lstm_summary["best_threshold"]["threshold"],
            "best_trades": lstm_summary["best_threshold"]["trades_executed"],
            "best_win_rate": lstm_summary["best_threshold"]["win_rate"],
            "best_profit_factor": lstm_summary["best_threshold"]["profit_factor"],
            "best_net_pnl": lstm_summary["best_threshold"]["net_pnl"],
            "best_sharpe": lstm_summary["best_threshold"]["sharpe"],
        },
    ]
    comparison = pd.DataFrame(comparison_rows)
    comparison.to_csv(OUTPUT_DIR / "model_comparison.csv", index=False)

    all_thresholds = pd.concat([baseline_thresholds, lstm_thresholds], ignore_index=True)
    all_thresholds.to_csv(OUTPUT_DIR / "combined_threshold_sweep.csv", index=False)

    payload = {
        "baseline_require_fvg_context": baseline_summary.get("require_fvg_context"),
        "baseline_require_trend_alignment": baseline_summary.get("require_trend_alignment"),
        "models": comparison_rows,
        "paths": {
            "comparison_csv": str(OUTPUT_DIR / "model_comparison.csv"),
            "combined_threshold_sweep": str(OUTPUT_DIR / "combined_threshold_sweep.csv"),
            "figures": str(figures_dir),
        },
    }
    (OUTPUT_DIR / "model_comparison_summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    save_model_summary_table_image(comparison, figures_dir / "model_comparison_table.png")
    save_threshold_metric_chart(all_thresholds, figures_dir / "threshold_profit_factor.png", metric="profit_factor", title="Threshold vs Profit Factor")
    save_threshold_metric_chart(all_thresholds, figures_dir / "threshold_sharpe.png", metric="sharpe", title="Threshold vs Sharpe")
    save_threshold_metric_chart(all_thresholds, figures_dir / "threshold_net_pnl.png", metric="net_pnl", title="Threshold vs Net PnL")

    print(f"baseline_require_fvg_context={baseline_summary.get('require_fvg_context')}")
    for row in comparison_rows:
        print(f"model={row['model_name']}")
        print(f"  average_roc_auc={format_metric(row['average_roc_auc'])}")
        print(f"  selected_profit_factor={format_metric(row['selected_profit_factor'])}")
        print(f"  best_threshold={format_metric(row['best_threshold'])}")
        print(f"  best_profit_factor={format_metric(row['best_profit_factor'])}")
        print(f"  best_net_pnl={format_metric(row['best_net_pnl'])}")
        print(f"  best_sharpe={format_metric(row['best_sharpe'])}")
    print(f"saved={OUTPUT_DIR}")


def _find_model_value(rows: list[dict[str, object]], model_name: str, key: str) -> object:
    for row in rows:
        if row.get("model_name") == model_name:
            return row.get(key)
    raise KeyError(f"Missing model row for {model_name}")


def format_metric(value: object) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


if __name__ == "__main__":
    main()

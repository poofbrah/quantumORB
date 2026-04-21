from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from reporting import (
    build_paper_experiment_comparison_frame,
    save_metric_bar_chart,
    save_model_summary_table_image,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the final setup-quality paper experiment for the primary NQ strategy."
    )
    parser.add_argument(
        "--output-dir",
        default=str(ROOT / "outputs" / "final_paper_experiment"),
        help="Directory for final paper-ready outputs.",
    )
    parser.add_argument(
        "--skip-lstm",
        action="store_true",
        help="Skip the LSTM stage if you only want classical-model comparisons.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    figures_dir = output_dir / "figures"
    baseline_dir = output_dir / "baseline_tabular"
    sequence_dir = output_dir / "sequence_dataset"
    lstm_dir = output_dir / "lstm"

    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    print("stage=baseline_tabular")
    run_script(
        "run_setup_quality_research_pipeline.py",
        [
            "--target-mode",
            "liquidity",
            "--output-dir",
            str(baseline_dir),
        ],
    )

    lstm_summary_payload: dict[str, object] | None = None
    if not args.skip_lstm:
        print("stage=sequence_dataset")
        run_script(
            "build_setup_quality_sequence_dataset.py",
            [
                "--target-mode",
                "liquidity",
                "--output-dir",
                str(sequence_dir),
            ],
        )

        print("stage=lstm")
        run_script(
            "run_setup_quality_lstm.py",
            [
                "--sequence-dir",
                str(sequence_dir),
                "--output-dir",
                str(lstm_dir),
            ],
        )
        lstm_summary_payload = json.loads((lstm_dir / "lstm_summary.json").read_text(encoding="utf-8"))

    print("stage=final_summary")
    baseline_summary = json.loads((baseline_dir / "research_summary.json").read_text(encoding="utf-8"))
    comparison = build_paper_experiment_comparison_frame(
        baseline_metrics=baseline_summary["baseline_all_setups"],
        baseline_best_thresholds=baseline_summary["best_thresholds"],
        lstm_best_threshold=lstm_summary_payload["best_threshold"] if lstm_summary_payload is not None else None,
    )
    comparison.to_csv(output_dir / "paper_experiment_comparison.csv", index=False)
    save_model_summary_table_image(comparison, figures_dir / "paper_experiment_table.png")
    save_metric_bar_chart(comparison, figures_dir / "paper_profit_factor.png", metric="profit_factor", title="Profit Factor by Selection Rule")
    save_metric_bar_chart(comparison, figures_dir / "paper_sharpe.png", metric="sharpe", title="Sharpe by Selection Rule")
    save_metric_bar_chart(comparison, figures_dir / "paper_net_pnl.png", metric="net_pnl", title="Net PnL by Selection Rule")

    payload = {
        "primary_strategy": "orb_session_vwap_retest_liquidity",
        "prediction_target": "P(success | setup, context)",
        "decision_rule": "Take setups whose predicted probability is at or above the chosen threshold.",
        "baseline_walk_forward_folds": baseline_summary.get("walk_forward_folds"),
        "baseline_walk_forward_paths": {
            "fold_results": baseline_summary["paths"].get("fold_results"),
            "predictions": baseline_summary["paths"].get("predictions"),
            "threshold_sweep": baseline_summary["paths"].get("threshold_sweep"),
        },
        "lstm_walk_forward_folds": lstm_summary_payload.get("walk_forward_folds") if lstm_summary_payload is not None else None,
        "lstm_walk_forward_paths": {
            "fold_results": lstm_summary_payload["paths"].get("fold_results"),
            "predictions": lstm_summary_payload["paths"].get("predictions"),
            "threshold_sweep": lstm_summary_payload["paths"].get("threshold_sweep"),
        } if lstm_summary_payload is not None else None,
        "baseline_output_dir": str(baseline_dir),
        "lstm_output_dir": str(lstm_dir) if lstm_summary_payload is not None else None,
        "comparison_rows": comparison.to_dict(orient="records"),
        "best_model_by_profit_factor": _best_row(comparison, "profit_factor"),
        "best_model_by_sharpe": _best_row(comparison, "sharpe"),
    }
    (output_dir / "paper_experiment_summary.json").write_text(
        json.dumps(payload, indent=2, default=str),
        encoding="utf-8",
    )

    print(f"prediction_target={payload['prediction_target']}")
    print("selection_rule=trade only when predicted probability exceeds the chosen threshold")
    print(f"baseline_walk_forward_folds={payload['baseline_walk_forward_folds']}")
    if payload["lstm_walk_forward_folds"] is not None:
        print(f"lstm_walk_forward_folds={payload['lstm_walk_forward_folds']}")
    print(comparison.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print(f"saved={output_dir}")


def run_script(script_name: str, extra_args: list[str]) -> None:
    script_path = ROOT / "examples" / script_name
    if not script_path.exists():
        raise FileNotFoundError(f"Missing script: {script_path}")
    subprocess.run([sys.executable, str(script_path), *extra_args], cwd=str(ROOT), check=True)


def _best_row(frame: pd.DataFrame, metric: str) -> dict[str, object]:
    ranked = frame.copy()
    ranked[metric] = pd.to_numeric(ranked[metric], errors="coerce")
    ranked = ranked.sort_values(metric, ascending=False, kind="stable")
    return ranked.iloc[0].to_dict()


if __name__ == "__main__":
    main()

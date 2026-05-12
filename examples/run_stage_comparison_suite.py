from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

ArtifactKind = Literal["json_single", "csv_table", "glob_json", "glob_csv", "file_exists"]


@dataclass(frozen=True)
class ArtifactSpec:
    path: str
    kind: ArtifactKind


@dataclass(frozen=True)
class ScriptSpec:
    key: str
    stage: str
    script: str
    args: tuple[str, ...] = ()
    artifacts: tuple[ArtifactSpec, ...] = field(default_factory=tuple)
    enabled_by_default: bool = True


STAGE_ORDER = [
    "data_prep",
    "debug",
    "core_nq",
    "rp_strategy",
    "ml",
    "paper",
    "meta",
]


SCRIPT_SPECS: tuple[ScriptSpec, ...] = (
    ScriptSpec(
        key="preprocess_nq_dataset",
        stage="data_prep",
        script="preprocess_nq_dataset.py",
        artifacts=(ArtifactSpec("data/processed/nq_1min_2022_2025.parquet", "file_exists"),),
    ),
    ScriptSpec(
        key="build_nq_setup_quality_dataset",
        stage="data_prep",
        script="build_nq_setup_quality_dataset.py",
        artifacts=(ArtifactSpec("outputs/setup_quality_dataset_nq_orb/dataset_summary.json", "json_single"),),
    ),
    ScriptSpec(
        key="build_setup_quality_sequence_dataset",
        stage="data_prep",
        script="build_setup_quality_sequence_dataset.py",
        artifacts=(ArtifactSpec("outputs/setup_quality_sequence_dataset_orb_session_vwap_retest_liquidity/build_summary.json", "json_single"),),
    ),
    ScriptSpec(
        key="run_nq_debug_2023H1",
        stage="debug",
        script="run_nq_debug_2023H1.py",
        artifacts=(ArtifactSpec("outputs/debug_metrics_2023H1.json", "json_single"),),
    ),
    ScriptSpec(
        key="run_nq_debug_2023H1_rr_compare",
        stage="debug",
        script="run_nq_debug_2023H1_rr_compare.py",
        artifacts=(ArtifactSpec("outputs/debug_metrics_2023H1_rr1.json", "json_single"),),
    ),
    ScriptSpec(
        key="run_nq_real_pipeline",
        stage="core_nq",
        script="run_nq_real_pipeline.py",
        args=("--instrument", "NQ"),
        artifacts=(ArtifactSpec("outputs/nq_am_displacement_orb_real_run/metrics.json", "json_single"),),
    ),
    ScriptSpec(
        key="run_nq_orb_session_vwap_retest_full_fixed_r",
        stage="core_nq",
        script="run_nq_orb_session_vwap_retest_full.py",
        args=("--target-mode", "fixed_r"),
        artifacts=(ArtifactSpec("outputs/nq_orb_session_vwap_retest_full_fixed_r/metrics.json", "json_single"),),
    ),
    ScriptSpec(
        key="run_nq_orb_session_vwap_retest_full_liquidity",
        stage="core_nq",
        script="run_nq_orb_session_vwap_retest_full.py",
        args=("--target-mode", "liquidity"),
        artifacts=(ArtifactSpec("outputs/nq_orb_session_vwap_retest_full_liquidity/metrics.json", "json_single"),),
    ),
    ScriptSpec(
        key="run_orb_regime_mode_compare",
        stage="core_nq",
        script="run_orb_regime_mode_compare.py",
        artifacts=(ArtifactSpec("outputs/orb_regime_mode_compare/mode_comparison.csv", "csv_table"),),
    ),
    ScriptSpec(
        key="run_orb_managed_profile_compare",
        stage="core_nq",
        script="run_orb_managed_profile_compare.py",
        artifacts=(ArtifactSpec("outputs/orb_managed_profile_compare/managed_profile_comparison.csv", "csv_table"),),
    ),
    ScriptSpec(
        key="run_rp_profits_8am_orb_2023H1",
        stage="rp_strategy",
        script="run_rp_profits_8am_orb_2023H1.py",
        artifacts=(ArtifactSpec("outputs/rp_profits_8am_orb_2023H1/metrics.json", "json_single"),),
    ),
    ScriptSpec(
        key="run_rp_profits_8am_orb_2023H1_compare",
        stage="rp_strategy",
        script="run_rp_profits_8am_orb_2023H1_compare.py",
        artifacts=(ArtifactSpec("outputs/rp_profits_8am_orb_2023H1_compare/metrics_*.json", "glob_json"),),
    ),
    ScriptSpec(
        key="run_rp_profits_8am_orb_bos_diagnostics_full",
        stage="rp_strategy",
        script="run_rp_profits_8am_orb_bos_diagnostics_full.py",
        artifacts=(
            ArtifactSpec("outputs/rp_profits_8am_orb_bos_diagnostics_full_range_10_15/diagnostics_summary.json", "json_single"),
            ArtifactSpec("outputs/rp_profits_8am_orb_bos_diagnostics_full_range_10_15/rejection_summary.csv", "csv_table"),
        ),
    ),
    ScriptSpec(
        key="run_rp_profits_8am_orb_bos_window_compare_2023H1",
        stage="rp_strategy",
        script="run_rp_profits_8am_orb_bos_window_compare_2023H1.py",
        artifacts=(ArtifactSpec("outputs/rp_profits_8am_orb_bos_window_compare_2023H1/window_comparison.csv", "csv_table"),),
    ),
    ScriptSpec(
        key="run_rp_profits_8am_orb_bos_window_compare_full",
        stage="rp_strategy",
        script="run_rp_profits_8am_orb_bos_window_compare_full.py",
        artifacts=(ArtifactSpec("outputs/rp_profits_8am_orb_full_window_compare/window_comparison.csv", "csv_table"),),
    ),
    ScriptSpec(
        key="run_rp_profits_8am_orb_full",
        stage="rp_strategy",
        script="run_rp_profits_8am_orb_full.py",
        artifacts=(ArtifactSpec("outputs/rp_profits_8am_orb_full/metrics.json", "json_single"),),
    ),
    ScriptSpec(
        key="run_rp_profits_8am_orb_full_compare",
        stage="rp_strategy",
        script="run_rp_profits_8am_orb_full_compare.py",
        artifacts=(ArtifactSpec("outputs/rp_profits_8am_orb_full_compare/metrics_*.json", "glob_json"),),
    ),
    ScriptSpec(
        key="run_rp_profits_8am_orb_full_entry_family_compare",
        stage="rp_strategy",
        script="run_rp_profits_8am_orb_full_entry_family_compare.py",
        artifacts=(ArtifactSpec("outputs/rp_profits_8am_orb_full_entry_family_compare/comparison.csv", "csv_table"),),
    ),
    ScriptSpec(
        key="run_rp_profits_8am_orb_vwap_displacement_full",
        stage="rp_strategy",
        script="run_rp_profits_8am_orb_vwap_displacement_full.py",
        artifacts=(ArtifactSpec("outputs/rp_profits_8am_orb_full_displacement_vwap_pullback/metrics.json", "json_single"),),
    ),
    ScriptSpec(
        key="run_setup_quality_baseline",
        stage="ml",
        script="run_setup_quality_baseline.py",
        artifacts=(
            ArtifactSpec("outputs/setup_quality_baseline_nq_orb/baseline_summary.json", "json_single"),
            ArtifactSpec("outputs/setup_quality_baseline_nq_orb/fold_results.csv", "csv_table"),
        ),
    ),
    ScriptSpec(
        key="run_setup_quality_research_pipeline",
        stage="ml",
        script="run_setup_quality_research_pipeline.py",
        artifacts=(
            ArtifactSpec("outputs/setup_quality_research_orb_session_vwap_retest_liquidity/research_summary.json", "json_single"),
            ArtifactSpec("outputs/setup_quality_research_orb_session_vwap_retest_liquidity/model_summary.csv", "csv_table"),
            ArtifactSpec("outputs/setup_quality_research_orb_session_vwap_retest_liquidity/threshold_sweep.csv", "csv_table"),
        ),
    ),
    ScriptSpec(
        key="run_setup_quality_lstm",
        stage="ml",
        script="run_setup_quality_lstm.py",
        artifacts=(
            ArtifactSpec("outputs/setup_quality_lstm_orb_session_vwap_retest_liquidity/lstm_summary.json", "json_single"),
            ArtifactSpec("outputs/setup_quality_lstm_orb_session_vwap_retest_liquidity/threshold_sweep.csv", "csv_table"),
        ),
    ),
    ScriptSpec(
        key="run_setup_quality_model_comparison",
        stage="ml",
        script="run_setup_quality_model_comparison.py",
        artifacts=(ArtifactSpec("outputs/setup_quality_model_comparison/model_comparison.csv", "csv_table"),),
    ),
    ScriptSpec(
        key="run_setup_quality_ga_optimizer",
        stage="ml",
        script="run_setup_quality_ga_optimizer.py",
        artifacts=(
            ArtifactSpec("outputs/setup_quality_ga_optimizer/ga_summary.json", "json_single"),
            ArtifactSpec("outputs/setup_quality_ga_optimizer/ga_fold_parameters.csv", "csv_table"),
        ),
    ),
    ScriptSpec(
        key="run_final_paper_experiment",
        stage="paper",
        script="run_final_paper_experiment.py",
        artifacts=(
            ArtifactSpec("outputs/final_paper_experiment/paper_experiment_summary.json", "json_single"),
            ArtifactSpec("outputs/final_paper_experiment/paper_experiment_comparison.csv", "csv_table"),
        ),
    ),
    ScriptSpec(
        key="run_github_repro_pipeline",
        stage="meta",
        script="run_github_repro_pipeline.py",
        artifacts=(ArtifactSpec("outputs/final_paper_experiment/paper_experiment_summary.json", "json_single"),),
        enabled_by_default=False,
    ),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run or collect staged experiment outputs and build a master comparison summary."
    )
    parser.add_argument("--rerun", action="store_true", help="Rerun the selected scripts instead of only collecting existing outputs.")
    parser.add_argument("--include-meta", action="store_true", help="Include meta-runners such as run_github_repro_pipeline.py.")
    parser.add_argument("--stages", nargs="*", choices=STAGE_ORDER, help="Only include the listed stages.")
    parser.add_argument("--only", nargs="*", help="Only include specific script keys.")
    parser.add_argument(
        "--output-dir",
        default=str(ROOT / "outputs" / "staged_example_runs"),
        help="Directory for the staged run log and comparison tables.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    specs = selected_specs(args)
    run_rows: list[dict[str, object]] = []
    metric_frames: list[pd.DataFrame] = []

    for spec in specs:
        started = time.perf_counter()
        status = "collected"
        error_text: str | None = None
        if args.rerun:
            status, error_text = execute_spec(spec)
        duration_seconds = time.perf_counter() - started

        artifact_rows, artifact_metrics = collect_artifacts(spec)
        if not artifact_rows and status == "collected":
            status = "missing_artifacts"
        run_rows.append(
            {
                "stage": spec.stage,
                "key": spec.key,
                "script": spec.script,
                "args": " ".join(spec.args),
                "status": status,
                "duration_seconds": round(duration_seconds, 4),
                "artifact_count": len(artifact_rows),
                "error": error_text,
            }
        )
        metric_frames.extend(artifact_metrics)
        print(f"stage={spec.stage} key={spec.key} status={status} artifacts={len(artifact_rows)}")

    run_log = pd.DataFrame(run_rows).sort_values(["stage", "key"], kind="stable")
    all_metrics = pd.concat(metric_frames, ignore_index=True) if metric_frames else pd.DataFrame()
    comparison = build_comparison_overview(all_metrics)

    run_log.to_csv(output_dir / "run_log.csv", index=False)
    all_metrics.to_csv(output_dir / "all_metrics_flat.csv", index=False)
    comparison.to_csv(output_dir / "comparison_overview.csv", index=False)
    (output_dir / "stage_summary.json").write_text(
        json.dumps(
            {
                "rerun": args.rerun,
                "stages": args.stages,
                "only": args.only,
                "run_log_path": str(output_dir / "run_log.csv"),
                "all_metrics_path": str(output_dir / "all_metrics_flat.csv"),
                "comparison_overview_path": str(output_dir / "comparison_overview.csv"),
                "scripts_included": [spec.key for spec in specs],
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print("")
    print(f"saved_run_log={output_dir / 'run_log.csv'}")
    print(f"saved_all_metrics={output_dir / 'all_metrics_flat.csv'}")
    print(f"saved_comparison={output_dir / 'comparison_overview.csv'}")
    if not comparison.empty:
        preview_columns = [column for column in ["stage", "key", "scenario", "model_name", "entry_mode", "window", "mode", "variant", "trades_executed", "win_rate", "profit_factor", "net_pnl", "max_drawdown", "sharpe", "sortino", "calmar"] if column in comparison.columns]
        print("")
        print(comparison[preview_columns].to_string(index=False, max_rows=50))


def selected_specs(args: argparse.Namespace) -> list[ScriptSpec]:
    specs = [spec for spec in SCRIPT_SPECS if args.include_meta or spec.enabled_by_default]
    if args.stages:
        allowed = set(args.stages)
        specs = [spec for spec in specs if spec.stage in allowed]
    if args.only:
        allowed_keys = set(args.only)
        specs = [spec for spec in specs if spec.key in allowed_keys]
    stage_index = {stage: idx for idx, stage in enumerate(STAGE_ORDER)}
    return sorted(specs, key=lambda spec: (stage_index.get(spec.stage, 999), spec.key))


def execute_spec(spec: ScriptSpec) -> tuple[str, str | None]:
    script_path = ROOT / "examples" / spec.script
    try:
        subprocess.run([sys.executable, str(script_path), *spec.args], cwd=str(ROOT), check=True)
        return "ran", None
    except subprocess.CalledProcessError as exc:
        return "failed", str(exc)


def collect_artifacts(spec: ScriptSpec) -> tuple[list[dict[str, object]], list[pd.DataFrame]]:
    artifact_rows: list[dict[str, object]] = []
    metric_frames: list[pd.DataFrame] = []
    for artifact in spec.artifacts:
        for path in resolve_paths(artifact):
            artifact_rows.append({"path": str(path), "kind": artifact.kind})
            frame = load_artifact_frame(spec, artifact, path)
            if frame is not None and not frame.empty:
                metric_frames.append(frame)
    return artifact_rows, metric_frames


def resolve_paths(artifact: ArtifactSpec) -> list[Path]:
    path = ROOT / artifact.path
    if artifact.kind in {"glob_json", "glob_csv"}:
        return sorted(path.parent.glob(path.name))
    return [path] if path.exists() else []


def load_artifact_frame(spec: ScriptSpec, artifact: ArtifactSpec, path: Path) -> pd.DataFrame | None:
    if artifact.kind == "file_exists":
        return pd.DataFrame([artifact_metadata(spec, artifact, path, {"exists": path.exists()})])
    if artifact.kind in {"json_single", "glob_json"}:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, list):
            frame = pd.json_normalize(payload, sep=".")
        else:
            frame = pd.json_normalize(payload, sep=".")
    elif artifact.kind in {"csv_table", "glob_csv"}:
        frame = pd.read_csv(path)
    else:
        return None

    if frame.empty:
        return None
    for column in frame.columns:
        if frame[column].map(lambda value: isinstance(value, (dict, list))).any():
            frame[column] = frame[column].map(json.dumps)
    frame.insert(0, "source_path", str(path.relative_to(ROOT)))
    frame.insert(0, "artifact_kind", artifact.kind)
    frame.insert(0, "script", spec.script)
    frame.insert(0, "key", spec.key)
    frame.insert(0, "stage", spec.stage)
    if "scenario" not in frame.columns:
        frame.insert(5, "scenario", path.stem)
    return frame


def artifact_metadata(spec: ScriptSpec, artifact: ArtifactSpec, path: Path, payload: dict[str, object]) -> dict[str, object]:
    row = payload.copy()
    row.update(
        {
            "stage": spec.stage,
            "key": spec.key,
            "script": spec.script,
            "artifact_kind": artifact.kind,
            "source_path": str(path.relative_to(ROOT)),
            "scenario": path.stem,
        }
    )
    return row


def build_comparison_overview(all_metrics: pd.DataFrame) -> pd.DataFrame:
    if all_metrics.empty:
        return pd.DataFrame()

    preferred_columns = [
        "stage",
        "key",
        "script",
        "scenario",
        "model_name",
        "entry_mode",
        "window",
        "mode",
        "variant",
        "target_mode",
        "rows",
        "dataset_rows",
        "setups",
        "setups_detected",
        "setups_found",
        "trades_executed",
        "trades_taken",
        "win_rate",
        "loss_rate",
        "profit_factor",
        "net_pnl",
        "gross_profit",
        "gross_loss",
        "average_trade_pnl",
        "average_win",
        "average_loss",
        "average_r",
        "expectancy",
        "max_drawdown",
        "max_drawdown_pct",
        "total_return",
        "sharpe",
        "sharpe_ratio",
        "sortino",
        "calmar",
        "average_roc_auc",
        "average_log_loss",
        "average_brier_score",
        "selected_profit_factor",
        "selected_win_rate",
        "selected_average_r",
        "walk_forward_folds",
        "baseline_walk_forward_folds",
        "lstm_walk_forward_folds",
        "percent_partial_taken",
        "percent_target_exits",
        "percent_stop_exits",
        "percent_trailing_stop_exits",
        "percent_session_end_exits",
        "source_path",
    ]
    present = [column for column in preferred_columns if column in all_metrics.columns]
    frame = all_metrics[present].copy()
    metric_candidates = [
        "setups",
        "setups_detected",
        "setups_found",
        "trades_executed",
        "trades_taken",
        "win_rate",
        "profit_factor",
        "net_pnl",
        "max_drawdown",
        "sharpe",
        "sharpe_ratio",
        "sortino",
        "calmar",
        "average_roc_auc",
        "selected_profit_factor",
    ]
    available_metrics = [column for column in metric_candidates if column in frame.columns]
    if available_metrics:
        mask = frame[available_metrics].notna().any(axis=1)
        frame = frame.loc[mask].reset_index(drop=True)
    return frame


if __name__ == "__main__":
    main()

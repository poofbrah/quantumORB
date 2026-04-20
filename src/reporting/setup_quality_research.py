from __future__ import annotations

import math
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import pandas as pd

from data.schemas import LabeledSetup, Trade


def build_setup_summary_frame(labeled_setups: list[LabeledSetup]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for item in labeled_setups:
        setup = item.setup
        rows.append(
            {
                "setup_id": setup.setup_id,
                "setup_name": setup.setup_name,
                "timestamp": setup.timestamp,
                "session_date": setup.session_date,
                "symbol": setup.symbol,
                "side": setup.direction.value,
                "entry_reference": setup.entry_reference,
                "stop_reference": setup.stop_reference,
                "target_reference": setup.target_reference,
                "label": item.label,
                "realized_return": item.realized_return,
                "realized_mae": item.realized_mae,
                "realized_mfe": item.realized_mfe,
                **{f"feature_{key}": value for key, value in setup.features.items()},
                **{f"context_{key}": value for key, value in setup.context.items() if isinstance(value, (str, int, float, bool, type(None)))},
            }
        )
    return pd.DataFrame(rows)


def build_trade_log_frame(trades: list[Trade]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for trade in trades:
        rows.append(
            {
                "trade_id": trade.trade_id,
                "setup_id": trade.setup_id,
                "setup_name": trade.setup_name,
                "symbol": trade.symbol,
                "trade_day": trade.trade_day,
                "setup_time": trade.setup_time,
                "entry_time": trade.entry_time,
                "exit_time": trade.exit_time,
                "side": trade.side.value,
                "entry_price": trade.entry_price,
                "exit_price": trade.exit_price,
                "initial_stop_price": trade.initial_stop_price,
                "final_stop_price": trade.final_stop_price,
                "partial_exit_price": trade.partial_exit_price,
                "partial_exit_time": trade.partial_exit_time,
                "partial_exit_fraction": trade.partial_exit_fraction,
                "runner_exit_price": trade.runner_exit_price,
                "runner_exit_time": trade.runner_exit_time,
                "exit_price_blended": trade.exit_price_blended,
                "size": trade.size,
                "pnl": trade.pnl,
                "pnl_r": trade.pnl_r,
                "return_pct": trade.return_pct,
                "bars_held": trade.bars_held,
                "exit_reason": trade.exit_reason,
                "partial_taken": trade.partial_taken,
                "moved_to_breakeven": trade.moved_to_breakeven,
                "trailing_stop_used": trade.trailing_stop_used,
                "max_favorable_excursion": trade.max_favorable_excursion,
                "max_adverse_excursion": trade.max_adverse_excursion,
                "max_favorable_excursion_r": trade.max_favorable_excursion_r,
                "max_adverse_excursion_r": trade.max_adverse_excursion_r,
                "max_unrealized_profit": trade.max_unrealized_profit,
                "max_unrealized_loss": trade.max_unrealized_loss,
            }
        )
    return pd.DataFrame(rows)


def select_best_threshold_row(
    threshold_results: pd.DataFrame,
    priority: tuple[str, ...] = ("profit_factor", "sharpe", "net_pnl"),
) -> pd.Series:
    if threshold_results.empty:
        raise ValueError("threshold_results must not be empty")
    ranked = threshold_results.copy()
    for column in priority:
        if column not in ranked.columns:
            raise ValueError(f"Missing threshold metric column: {column}")
        ranked[column] = pd.to_numeric(ranked[column], errors="coerce").fillna(float("-inf"))
    ranked["trades_executed"] = pd.to_numeric(ranked.get("trades_executed"), errors="coerce").fillna(0)
    ranked = ranked.sort_values(
        list(priority) + ["trades_executed", "threshold"],
        ascending=[False] * len(priority) + [False, True],
        kind="stable",
    )
    return ranked.iloc[0]


def save_label_distribution_chart(dataset: pd.DataFrame, path: Path) -> None:
    counts = dataset["label"].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(["loss/stop", "success/target"], [counts.get(0, 0), counts.get(1, 0)], color=["#d95f02", "#1b9e77"])
    ax.set_title("Setup Label Distribution")
    ax.set_ylabel("Count")
    ax.grid(axis="y", alpha=0.25)
    _save_figure(fig, path)


def save_fold_metric_chart(fold_results: pd.DataFrame, path: Path, metric: str = "roc_auc") -> None:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for model_name, group in fold_results.groupby("model_name", sort=False):
        ordered = group.sort_values("fold_id", kind="stable")
        ax.plot(ordered["fold_id"], ordered[metric], marker="o", linewidth=2, label=model_name)
    ax.set_title(f"Walk-Forward {metric.replace('_', ' ').title()} by Fold")
    ax.set_xlabel("Fold")
    ax.set_ylabel(metric.replace("_", " ").title())
    ax.grid(alpha=0.25)
    ax.legend()
    _save_figure(fig, path)


def save_threshold_metric_chart(
    threshold_results: pd.DataFrame,
    path: Path,
    metric: str,
    title: str,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for model_name, group in threshold_results.groupby("model_name", sort=False):
        ordered = group.sort_values("threshold", kind="stable")
        ax.plot(ordered["threshold"], ordered[metric], marker="o", linewidth=2, label=model_name)
    ax.set_title(title)
    ax.set_xlabel("Probability Threshold")
    ax.set_ylabel(metric.replace("_", " ").title())
    ax.grid(alpha=0.25)
    ax.legend()
    _save_figure(fig, path)


def save_equity_curve_chart(
    curves: Iterable[tuple[str, list[tuple[object, float]]]],
    path: Path,
    title: str,
) -> None:
    fig, ax = plt.subplots(figsize=(9, 4.5))
    for label, curve in curves:
        if not curve:
            continue
        timestamps = [point[0] for point in curve]
        values = [point[1] for point in curve]
        ax.plot(timestamps, values, linewidth=2, label=label)
    ax.set_title(title)
    ax.set_xlabel("Time")
    ax.set_ylabel("Equity")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.autofmt_xdate()
    _save_figure(fig, path)


def save_model_summary_table_image(model_summary: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, max(2.5, 1.2 + len(model_summary) * 0.5)))
    ax.axis("off")
    rounded = model_summary.copy()
    for column in rounded.columns:
        if pd.api.types.is_numeric_dtype(rounded[column]):
            rounded[column] = rounded[column].map(_format_numeric)
    table = ax.table(
        cellText=rounded.values,
        colLabels=rounded.columns,
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.4)
    ax.set_title("Model Summary", pad=12)
    _save_figure(fig, path)


def _format_numeric(value: object) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        if math.isfinite(value):
            return f"{value:.4f}"
        return "n/a"
    return str(value)


def _save_figure(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)

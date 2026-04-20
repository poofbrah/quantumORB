from __future__ import annotations

from pathlib import Path

from .specification import ORBStrategySpec


def format_strategy_spec(spec: ORBStrategySpec) -> str:
    lines = [
        f"Strategy Profile: {spec.profile_name}",
        f"Instrument: {spec.instrument}",
        f"NY Open Anchored: {'yes' if spec.ny_open_anchored else 'no'}",
        f"Session: {spec.session_name} ({spec.session_start}-{spec.session_end} {spec.session_timezone})",
        f"Latest Entry Time: {spec.latest_entry_time or 'none'}",
        f"Opening Range: {spec.opening_range_start_time}-{spec.opening_range_end_time} ({spec.opening_range_minutes} minutes)",
        f"Long Trigger: {spec.long_trigger}",
        f"Short Trigger: {spec.short_trigger}",
        f"Breakout Confirmation: {spec.breakout_confirmation_rule}",
        f"Retest Requirement: {'yes' if spec.require_retest else 'no'} ({spec.retest_rule or 'none'})",
        f"Trend Filter: {spec.trend_filter}{f' on {spec.trend_column}' if spec.trend_filter != 'none' else ''}",
        f"Volatility Filter: {'enabled' if spec.volatility_filter_enabled else 'disabled'}",
        f"Displacement Rule: {spec.displacement.rule}",
        f"Strong Close Definition: body_range_ratio>={spec.displacement.strong_close_min_body_range_ratio}, boundary_distance={spec.displacement.strong_close_min_boundary_distance}, boundary_distance_atr={spec.displacement.strong_close_min_boundary_distance_atr}",
        f"Early Counter-Liquidity Consumption Invalidates Breakout: {'yes' if spec.displacement.invalidate_on_early_counter_liquidity_consumption else 'no'}",
        f"Retest Rescue Allowed: {'yes' if spec.displacement.allow_retest_rescue_after_early_liquidity_consumption else 'no'}",
        f"Liquidity Target Mode: {spec.liquidity.mode}",
        f"Liquidity Priority: {', '.join(spec.liquidity.priority) if spec.liquidity.priority else 'none'}",
        f"London Sweep Context: {spec.liquidity.london_sweep_context_mode or 'none'}",
        f"First Draw Rule: {spec.management.first_draw_target_rule or 'none'}",
        f"Minimum RR Threshold: {spec.management.minimum_rr_threshold}",
        f"Entry Rule: {spec.entry_rule}",
        f"Stop Rule: {spec.stop.rule}",
        f"Stop Buffers: points={spec.stop.stop_buffer_points}, atr_multiple={spec.stop.stop_buffer_atr_multiple}, wick_reference={spec.stop.wick_reference_mode}, fallback={spec.stop.fallback_stop_mode}",
        f"Target Rule: {spec.target_rule} ({spec.target_r_multiple}R if applicable)",
        f"First Partial: {spec.management.partial_take_profit_rule or 'none'} fraction={spec.management.partial_take_profit_fraction}",
        f"Breakeven After First Draw: {'yes' if spec.management.breakeven_after_first_draw else 'no'}",
        f"Breakeven Rule: {spec.management.breakeven_rule or 'none'} trigger_r={spec.management.breakeven_trigger_r} after_partial={spec.management.breakeven_after_partial}",
        f"Runner Target Rule: {spec.management.runner_target_rule or 'none'}",
        f"Runner Target Priority: {', '.join(spec.management.runner_target_priority) if spec.management.runner_target_priority else 'none'}",
        f"Runner Trail Rule: {spec.management.runner_trail_rule or 'none'} atr_multiple={spec.management.runner_trail_atr_multiple}",
        f"Trailing Stop: {spec.management.trailing_stop_rule or 'none'}",
        f"Max Trades Per Session: {spec.max_trades_per_session}",
        f"Trade Windows: {', '.join(spec.allowed_trade_windows) if spec.allowed_trade_windows else 'all session'}",
        f"Allowed Days: {', '.join(spec.allowed_days_of_week) if spec.allowed_days_of_week else 'all days'}",
        f"News Skip Rules: {', '.join(spec.news_skip_rules) if spec.news_skip_rules else 'placeholder only'}",
        f"Discretionary Skip Reasons: {', '.join(spec.discretionary_skip_reasons) if spec.discretionary_skip_reasons else 'placeholder only'}",
        "Operational Notes: Strong-close breakout qualification, latest-entry cutoff, midpoint/wick stop placement, and fixed 2R targeting are operational. Trailing and multi-stage runner management are disabled for the current debugging pass; richer liquidity-context interpretation remains only partially formalized.",
    ]
    return "\n".join(lines)


def write_strategy_spec(spec: ORBStrategySpec, path: str | Path) -> None:
    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_path.write_text(format_strategy_spec(spec), encoding="utf-8")



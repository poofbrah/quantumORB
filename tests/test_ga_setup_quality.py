from __future__ import annotations

import pandas as pd

from ga.setup_quality import (
    GAOptimizerConfig,
    SetupQualityGenome,
    apply_setup_quality_genome,
    optimize_setup_quality_genome,
)


def make_prediction_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "setup_id": ["a", "b", "c", "d"],
            "probability": [0.8, 0.72, 0.55, 0.4],
            "label": [1, 1, 0, 0],
            "realized_return": [1.2, 0.8, -1.0, -0.6],
            "feature_breakout_strength": [0.4, 0.3, 0.2, 0.1],
            "feature_trend_spread": [0.5, 0.4, 0.3, 0.1],
            "feature_distance_to_vwap": [0.8, 1.2, 2.0, 3.0],
            "feature_relative_volume": [1.5, 1.2, 0.9, 0.8],
            "feature_range_width": [3.0, 4.0, 5.0, 6.0],
            "feature_rsi": [62.0, 59.0, 52.0, 50.5],
        }
    )


def test_apply_setup_quality_genome_filters_by_multiple_conditions() -> None:
    genome = SetupQualityGenome(
        probability_threshold=0.7,
        min_breakout_strength=0.25,
        min_abs_trend_spread=0.3,
        max_distance_to_vwap=1.5,
        min_relative_volume=1.0,
        min_range_width=2.0,
        max_range_width=5.0,
        min_rsi_center_distance=8.0,
    )

    selected = apply_setup_quality_genome(make_prediction_frame(), genome)

    assert list(selected["setup_id"]) == ["a", "b"]


def test_optimize_setup_quality_genome_returns_nonempty_result() -> None:
    result = optimize_setup_quality_genome(
        make_prediction_frame(),
        config=GAOptimizerConfig(population_size=8, generations=3, elite_size=3, min_trades=1, seed=3),
    )

    assert result.trades >= 1
    assert result.genome.probability_threshold >= 0.5

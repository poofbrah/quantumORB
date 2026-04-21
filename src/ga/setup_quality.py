from __future__ import annotations

import random
from dataclasses import dataclass
from statistics import mean

import numpy as np
import pandas as pd


@dataclass(slots=True)
class GAOptimizerConfig:
    population_size: int = 24
    generations: int = 10
    elite_size: int = 6
    mutation_rate: float = 0.25
    seed: int = 7
    min_trades: int = 20
    threshold_bounds: tuple[float, float] = (0.50, 0.85)
    breakout_strength_bounds: tuple[float, float] = (0.0, 0.8)
    trend_spread_bounds: tuple[float, float] = (0.0, 1.5)
    distance_to_vwap_bounds: tuple[float, float] = (0.1, 10.0)
    relative_volume_bounds: tuple[float, float] = (0.0, 3.5)
    range_width_bounds: tuple[float, float] = (0.0, 30.0)
    rsi_center_distance_bounds: tuple[float, float] = (0.0, 30.0)


@dataclass(slots=True)
class SetupQualityGenome:
    probability_threshold: float
    min_breakout_strength: float
    min_abs_trend_spread: float
    max_distance_to_vwap: float
    min_relative_volume: float
    min_range_width: float
    max_range_width: float
    min_rsi_center_distance: float


@dataclass(slots=True)
class SetupQualityGenomeScore:
    genome: SetupQualityGenome
    fitness: float
    trades: int
    win_rate: float | None
    profit_factor: float | None
    sharpe: float | None
    net_pnl: float
    max_drawdown: float


def optimize_setup_quality_genome(
    prediction_frame: pd.DataFrame,
    config: GAOptimizerConfig | None = None,
) -> SetupQualityGenomeScore:
    settings = config or GAOptimizerConfig()
    rng = random.Random(settings.seed)
    population = [_baseline_genome(settings)]
    population.extend(_random_genome(settings, rng) for _ in range(max(0, settings.population_size - 1)))
    best: SetupQualityGenomeScore | None = None

    for _ in range(settings.generations):
        scored = sorted(
            (_score_genome(genome, prediction_frame, settings) for genome in population),
            key=lambda item: item.fitness,
            reverse=True,
        )
        if best is None or scored[0].fitness > best.fitness:
            best = scored[0]

        elites = [item.genome for item in scored[: settings.elite_size]]
        next_population = list(elites)
        while len(next_population) < settings.population_size:
            parent_a = rng.choice(elites)
            parent_b = rng.choice(elites)
            child = _crossover(parent_a, parent_b, rng)
            if rng.random() < settings.mutation_rate:
                child = _mutate(child, settings, rng)
            next_population.append(child)
        population = next_population

    assert best is not None
    return best


def apply_setup_quality_genome(prediction_frame: pd.DataFrame, genome: SetupQualityGenome) -> pd.DataFrame:
    selected = prediction_frame.copy()
    selected = selected.loc[selected["probability"] >= genome.probability_threshold]
    selected = selected.loc[selected["feature_breakout_strength"].fillna(0.0) >= genome.min_breakout_strength]
    selected = selected.loc[selected["feature_trend_spread"].abs().fillna(0.0) >= genome.min_abs_trend_spread]
    selected = selected.loc[selected["feature_distance_to_vwap"].abs().fillna(float("inf")) <= genome.max_distance_to_vwap]
    selected = selected.loc[selected["feature_relative_volume"].fillna(0.0) >= genome.min_relative_volume]
    selected = selected.loc[selected["feature_range_width"].fillna(0.0) >= genome.min_range_width]
    selected = selected.loc[selected["feature_range_width"].fillna(float("inf")) <= genome.max_range_width]
    rsi_distance = (selected["feature_rsi"].fillna(50.0) - 50.0).abs()
    selected = selected.loc[rsi_distance >= genome.min_rsi_center_distance]
    return selected.copy()


def evaluate_setup_quality_genome(
    prediction_frame: pd.DataFrame,
    genome: SetupQualityGenome,
    config: GAOptimizerConfig | None = None,
) -> SetupQualityGenomeScore:
    settings = config or GAOptimizerConfig()
    return _score_genome(genome, prediction_frame, settings)


def genome_to_dict(genome: SetupQualityGenome) -> dict[str, float]:
    return {
        "probability_threshold": genome.probability_threshold,
        "min_breakout_strength": genome.min_breakout_strength,
        "min_abs_trend_spread": genome.min_abs_trend_spread,
        "max_distance_to_vwap": genome.max_distance_to_vwap,
        "min_relative_volume": genome.min_relative_volume,
        "min_range_width": genome.min_range_width,
        "max_range_width": genome.max_range_width,
        "min_rsi_center_distance": genome.min_rsi_center_distance,
    }


def _score_genome(genome: SetupQualityGenome, prediction_frame: pd.DataFrame, config: GAOptimizerConfig) -> SetupQualityGenomeScore:
    selected = apply_setup_quality_genome(prediction_frame, genome)
    trades = len(selected)
    if trades == 0:
        return SetupQualityGenomeScore(genome, fitness=-1e9, trades=0, win_rate=None, profit_factor=None, sharpe=None, net_pnl=0.0, max_drawdown=0.0)

    realized = selected["realized_return"].astype(float)
    gross_profit = float(realized[realized > 0].sum())
    gross_loss = float(realized[realized < 0].sum())
    profit_factor = (gross_profit / abs(gross_loss)) if gross_loss < 0 else None
    net_pnl = float(realized.sum())
    sharpe = _sharpe(realized.tolist())
    max_drawdown = _max_drawdown(realized.tolist())
    win_rate = float(selected["label"].mean()) if not selected.empty else None

    trade_penalty = 0.0 if trades >= config.min_trades else (config.min_trades - trades) * 0.15
    pf_component = 0.0 if profit_factor is None else profit_factor
    sharpe_component = 0.0 if sharpe is None else sharpe
    fitness = sharpe_component + (0.6 * pf_component) + (0.05 * net_pnl) - (0.3 * abs(max_drawdown)) - trade_penalty
    return SetupQualityGenomeScore(
        genome=genome,
        fitness=float(fitness),
        trades=trades,
        win_rate=win_rate,
        profit_factor=profit_factor,
        sharpe=sharpe,
        net_pnl=net_pnl,
        max_drawdown=max_drawdown,
    )


def _random_genome(config: GAOptimizerConfig, rng: random.Random) -> SetupQualityGenome:
    return SetupQualityGenome(
        probability_threshold=rng.uniform(*config.threshold_bounds),
        min_breakout_strength=rng.uniform(*config.breakout_strength_bounds),
        min_abs_trend_spread=rng.uniform(*config.trend_spread_bounds),
        max_distance_to_vwap=rng.uniform(*config.distance_to_vwap_bounds),
        min_relative_volume=config.relative_volume_bounds[0],
        min_range_width=config.range_width_bounds[0],
        max_range_width=config.range_width_bounds[1],
        min_rsi_center_distance=config.rsi_center_distance_bounds[0],
    )


def _baseline_genome(config: GAOptimizerConfig) -> SetupQualityGenome:
    return SetupQualityGenome(
        probability_threshold=config.threshold_bounds[0],
        min_breakout_strength=config.breakout_strength_bounds[0],
        min_abs_trend_spread=config.trend_spread_bounds[0],
        max_distance_to_vwap=config.distance_to_vwap_bounds[1],
        min_relative_volume=config.relative_volume_bounds[0],
        min_range_width=config.range_width_bounds[0],
        max_range_width=config.range_width_bounds[1],
        min_rsi_center_distance=config.rsi_center_distance_bounds[0],
    )


def _crossover(parent_a: SetupQualityGenome, parent_b: SetupQualityGenome, rng: random.Random) -> SetupQualityGenome:
    def mix(a: float, b: float) -> float:
        return a if rng.random() < 0.5 else b

    return SetupQualityGenome(
        probability_threshold=mix(parent_a.probability_threshold, parent_b.probability_threshold),
        min_breakout_strength=mix(parent_a.min_breakout_strength, parent_b.min_breakout_strength),
        min_abs_trend_spread=mix(parent_a.min_abs_trend_spread, parent_b.min_abs_trend_spread),
        max_distance_to_vwap=mix(parent_a.max_distance_to_vwap, parent_b.max_distance_to_vwap),
        min_relative_volume=0.0,
        min_range_width=0.0,
        max_range_width=30.0,
        min_rsi_center_distance=0.0,
    )


def _mutate(genome: SetupQualityGenome, config: GAOptimizerConfig, rng: random.Random) -> SetupQualityGenome:
    mutated = SetupQualityGenome(**genome_to_dict(genome))
    field_name = rng.choice(
        [
            "probability_threshold",
            "min_breakout_strength",
            "min_abs_trend_spread",
            "max_distance_to_vwap",
        ]
    )
    bounds_map = {
        "probability_threshold": config.threshold_bounds,
        "min_breakout_strength": config.breakout_strength_bounds,
        "min_abs_trend_spread": config.trend_spread_bounds,
        "max_distance_to_vwap": config.distance_to_vwap_bounds,
    }
    setattr(mutated, field_name, rng.uniform(*bounds_map[field_name]))
    mutated.min_relative_volume = 0.0
    mutated.min_range_width = 0.0
    mutated.max_range_width = 30.0
    mutated.min_rsi_center_distance = 0.0
    return mutated


def _sharpe(values: list[float]) -> float | None:
    if len(values) < 2:
        return None
    series = pd.Series(values, dtype="float64")
    std = float(series.std(ddof=1))
    if std == 0.0:
        return None
    return float((series.mean() / std) * np.sqrt(len(series)))


def _max_drawdown(values: list[float]) -> float:
    if not values:
        return 0.0
    equity = pd.Series(values, dtype="float64").cumsum()
    peaks = equity.cummax()
    drawdown = equity - peaks
    return float(drawdown.min())

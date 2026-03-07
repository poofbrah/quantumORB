from pathlib import Path

from config.loader import get_config, load_config


def test_load_config_default() -> None:
    config = load_config()
    assert config.project.name == "quantumORB"
    assert config.project.timezone == "America/New_York"
    assert config.strategy.active_strategy == "orb"
    assert config.data.raw_data_dir.name == "raw"


def test_get_config_is_cached() -> None:
    first = get_config()
    second = get_config()
    assert first is second


def test_load_config_from_explicit_path() -> None:
    config = load_config(Path("config/default.yaml"))
    assert config.backtest.initial_capital == 100000.0

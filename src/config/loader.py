from __future__ import annotations

from dataclasses import fields
from functools import lru_cache
from pathlib import Path
from typing import Any, TypeVar

import yaml

from .models import (
    AppConfig,
    BacktestConfig,
    DataConfig,
    IntegrationsConfig,
    ModelingConfig,
    ProjectConfig,
    StrategyConfig,
)

T = TypeVar("T")


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _resolve_path(path: str | Path | None) -> Path:
    if path is None:
        return _repo_root() / "config" / "default.yaml"
    path = Path(path)
    if path.is_absolute():
        return path
    return _repo_root() / path


def _build_dataclass(cls: type[T], values: dict[str, Any] | None) -> T:
    values = values or {}
    allowed = {field.name for field in fields(cls)}
    filtered = {key: value for key, value in values.items() if key in allowed}
    instance = cls(**filtered)

    if hasattr(instance, "root_dir") and getattr(instance, "root_dir") == Path("."):
        setattr(instance, "root_dir", _repo_root())

    if isinstance(instance, DataConfig):
        instance.raw_data_dir = _repo_root() / Path(instance.raw_data_dir)
        instance.processed_data_dir = _repo_root() / Path(instance.processed_data_dir)

    return instance


def load_config(path: str | Path | None = None) -> AppConfig:
    config_path = _resolve_path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}

    return AppConfig(
        project=_build_dataclass(ProjectConfig, payload.get("project")),
        data=_build_dataclass(DataConfig, payload.get("data")),
        backtest=_build_dataclass(BacktestConfig, payload.get("backtest")),
        strategy=_build_dataclass(StrategyConfig, payload.get("strategy")),
        modeling=_build_dataclass(ModelingConfig, payload.get("modeling")),
        integrations=_build_dataclass(IntegrationsConfig, payload.get("integrations")),
    )


@lru_cache(maxsize=1)
def get_config() -> AppConfig:
    return load_config()

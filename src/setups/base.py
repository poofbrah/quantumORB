from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd

from data.schemas import SetupEvent


class BaseSetupDetector(ABC):
    """Base interface for setup detectors that emit candidate setups."""

    @abstractmethod
    def detect(self, frame: pd.DataFrame) -> list[SetupEvent]:
        """Return candidate setups using only information available at decision time."""

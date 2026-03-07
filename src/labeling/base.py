from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd

from data.schemas import LabeledSetup, SetupEvent


class BaseSetupLabeler(ABC):
    @abstractmethod
    def label(self, frame: pd.DataFrame, setups: list[SetupEvent]) -> list[LabeledSetup]:
        """Attach forward-only labels using bars after the setup timestamp."""

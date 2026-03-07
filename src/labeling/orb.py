from __future__ import annotations

from .future_path import ForwardSetupLabeler, LabelerConfig, QualityBucketRule


class ORBLabeler(ForwardSetupLabeler):
    """Thin ORB-specific wrapper over the generic forward setup labeler."""

    def __init__(
        self,
        config: LabelerConfig | None = None,
        quality_rule: QualityBucketRule | None = None,
    ) -> None:
        super().__init__(config=config, quality_rule=quality_rule)

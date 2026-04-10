from dataclasses import dataclass

@dataclass
class TargetConfig:
    """Configure target construction."""
    target_factory: str = "TargetInitial"
    """Which `TargetFactory` should be used to build target matrices. Defined in `targetfactory.py`."""


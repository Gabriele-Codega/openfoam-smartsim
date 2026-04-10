from dataclasses import dataclass, field

from .referenceconfig import ReferenceConfig
from .metricconfig import MetricConfig
from .targetconfig import TargetConfig
from .optimconfig import OptimConfig

@dataclass
class TMOPConfig:
    """Configure TMOP."""
    reference:  ReferenceConfig = field(default_factory=ReferenceConfig)
    """Dataclass to handle shape function configuration."""
    metric: MetricConfig    = field(default_factory=MetricConfig)
    """Dataclass to handle metric configuration."""
    target: TargetConfig    = field(default_factory=TargetConfig)
    """Dataclass to handle target matrix configuration."""
    optim:  OptimConfig     = field(default_factory=OptimConfig)
    """Dataclass to handle optimiser configuration."""


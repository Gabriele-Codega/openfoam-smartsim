from dataclasses import dataclass
from typing import Optional 

@dataclass
class MetricConfig:
    """Configure the TMOP metric for optimisation."""
    metric_fn: str = "mu_66"
    """Name of the metric to use. Should be defined in `tmopmetrics.py`."""
    gamma: Optional[float] = None
    """Optional parameter for metrics defined as convex combinations of other metrics. Should be in the range [0,1]."""
    c: Optional[float] = 1e-3
    """Parameter for untangling. Controls the position of the barrier with respect to the minimum determinant."""
    d: Optional[float] = 1e-3
    """Parameter for worst case optimisation. Controls the position of the barrier with respect to the maximum value of the metric."""


import torch
from dataclasses import dataclass, field
from typing import Optional, Type

@dataclass
class ShapeConfig:
    """
    Configure the shape functions on the reference element.
    """
    shape_fn: str = "stable_mean_value_coordinates"
    """Name of the shape function. Should be defined in `shape_functions.py`"""
    regular_ref: bool = True # might change in the future to be a string
    """Whether to use a regular n-gon as reference. Note: only works with `True` at the moment."""
    n_samples: int = 10
    """Number of sample points per element."""

@dataclass
class MetricConfig:
    """Configure the TMOP metric for optimisation."""
    metric_fn: str = "mu_66"
    """Name of the metric to use. Should be defined in `tmopmetrics.py`."""
    gamma: Optional[float] = None
    """Optional parameter for metrics defined as convex combinations of other metrics. Should be in the range [0,1]."""
    untangle: bool = True
    """Whether to force mesh untangling. Note: if `True`, then the metric should be 'Non-barrier'."""
    c: float = 1e-3 # probably can be optional
    """Parameter for untangling. Controls the position of the barrier with respect to the minimum determinant."""
    d: float = 1e-3 # probably can be optional
    """Parameter for worst case optimisation. Controls the position of the barrier with respect to the maximum value of the metric."""

@dataclass
class TargetConfig:
    """Configure target construction."""
    preserve_size:          bool = True
    """Whether to preserve the size of the original mesh elements."""
    preserve_orientation:   bool = True
    """Whether to preserve the orientation of the original mesh elements."""
    preserve_skewness:      bool = True
    """Whether to preserve the internal angles of the original mesh elements. If `False` tries to make the elements as regular as possible."""
    preserve_aspect:        bool = True
    """Whether to preserve the aspect ration of the original mesh elements. If `False` triest to make the elements as regular as possible."""

@dataclass
class OptimConfig:
    """Configure the optimiser."""
    optimiser: Type[torch.optim.Optimizer] = torch.optim.Adam
    """Which optimiser should be used, specified as (subclass of) `torch.optim.Optimizer`. Using custom optimisers is allowed but requires importing the corresponding module in `tmop_motion.py`."""
    lr: float = 1e-3
    """Optimiser learning rate."""
    max_steps: int = 1000
    """Maximum number of optimisation epochs per timestep."""
    patience: int = 50
    """Maximum number of steps without improvement in the loss. After `patience` steps without improvement in the maximum value of the loss, optimisation stops."""
    rtol: float = 1e-2
    """Minimum accepted relative improvement in the loss. If the improvement is smaller than `rtol`, the counter of epochs since improvement is incremented and after `patience' epochs optimisation stops."""

@dataclass
class TMOPConfig:
    """Configure TMOP parameters."""
    shape:  ShapeConfig     = field(default_factory=ShapeConfig)
    """Dataclass to handle shape function configuration."""
    metric: MetricConfig    = field(default_factory=MetricConfig)
    """Dataclass to handle metric configuration."""
    target: TargetConfig    = field(default_factory=TargetConfig)
    """Dataclass to handle target matrix configuration."""
    optim:  OptimConfig     = field(default_factory=OptimConfig)
    """Dataclass to handle optimiser configuration."""

@dataclass
class Config:
    """Main configuration."""
    mpi_ranks: int = field(default=1, metadata={"help": "Number of MPI ranks from OpenFOAM"})
    """Number of MPI ranks in OpenFOAM."""
    device: Optional[str] = None
    """Which accelerator should be used in PyTorch. Defaults to GPU ('cuda') if available, otherwise falls back to CPU ('cpu')."""

    tmop: TMOPConfig = field(default_factory=TMOPConfig)
    """Dataclass to handle TMOP configuration."""

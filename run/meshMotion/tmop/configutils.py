import torch
from dataclasses import dataclass, field
from typing import Optional, Type, Dict

@dataclass
class ShapeConfig:
    """
    Configure the shape functions on the reference element.
    """
    shape_fn: str = "stable_mean_value_coordinates"
    """Name of the shape function. Should be defined in `shape_functions.py`."""
    reference_type: str = "regular"
    """What reference element should be used. Must be one of ['regular', 'initial', 'svd']"""
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
    target_factory: str = "TargetInitial"
    """Which `TargetFactory` should be used to build target matrices. Defined in `tmopfactory.py`."""
    # preserve_size:          bool = True
    # """Whether to preserve the size of the original mesh elements."""
    # preserve_orientation:   bool = True
    # """Whether to preserve the orientation of the original mesh elements."""
    # preserve_skewness:      bool = True
    # """Whether to preserve the internal angles of the original mesh elements. If `False` tries to make the elements as regular as possible."""
    # preserve_aspect:        bool = True
    # """Whether to preserve the aspect ration of the original mesh elements. If `False` triest to make the elements as regular as possible."""

@dataclass
class OptimConfig:
    """Configure the optimiser."""
    optimiser: Type[torch.optim.Optimizer] = torch.optim.Adam
    """Which optimiser should be used, specified as (subclass of) `torch.optim.Optimizer`. Using custom optimisers is allowed but requires importing the corresponding module in `tmop_motion.py`."""
    optimiser_kwargs: Dict = field(default_factory=dict)
    """Optimiser keyword arguments."""
    scheduler: Type[torch.optim.lr_scheduler.LRScheduler] = torch.optim.lr_scheduler.ReduceLROnPlateau
    """Which learning rate scheduler should be used, specified as (subclass of) `torch.optim.lr_scheduler.LRScheduler`."""
    scheduler_kwargs: Dict = field(default_factory=dict)
    """Scheduler keyword arguments."""
    max_steps: int = 1000
    """Maximum number of optimisation epochs per timestep."""
    patience: int = 50
    """Maximum number of steps without improvement in the loss. After `patience` steps without improvement in the maximum value of the loss, optimisation stops."""
    rtol: float = 1e-2
    """Minimum accepted relative improvement in the loss. If the improvement is smaller than `rtol`, the counter of epochs since improvement is incremented and after `patience' epochs optimisation stops."""
    batch_size: int = -1
    """Batch size. If equal to -1, then do not batch."""

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
    mpi_ranks: int = 1
    """Number of MPI ranks in OpenFOAM."""
    device: Optional[str] = None
    """Which accelerator should be used in PyTorch. Defaults to GPU ('cuda') if available, otherwise falls back to CPU ('cpu')."""
    mode: str = "incremental"
    """Either 'incremental' or 'points0'. If 'incremental', the motion is computed with respect to the mesh at the previous time step. If 'points0', the motion is computed with respect to the initial mesh."""

    tmop: TMOPConfig = field(default_factory=TMOPConfig)
    """Dataclass to handle TMOP configuration."""

import torch
from dataclasses import dataclass, field
from typing import Optional, Type

@dataclass
class ShapeConfig:
    shape_fn: str = "stable_mean_value_coordinates"
    regular_ref: bool = True # might change in the future to be a string
    n_samples: int = 10

@dataclass
class MetricConfig:
    metric_fn: str = "mu_66"
    gamma: Optional[float] = None
    untangle: bool = True
    c: float = 1e-3 # probably can be optional
    d: float = 1e-3 # probably can be optional

@dataclass
class TargetConfig:
    preserve_size:          bool = True
    preserve_orientation:   bool = True
    preserve_skewness:      bool = True
    preserve_aspect:        bool = True

@dataclass
class OptimConfig:
    optimiser: Type[torch.optim.Optimizer] = torch.optim.Adam
    lr: float = 1e-3
    max_steps: int = 1000
    patience: int = 50
    rtol: float = 1e-2

@dataclass
class TMOPConfig:
    shape:  ShapeConfig     = field(default_factory=ShapeConfig)
    metric: MetricConfig    = field(default_factory=MetricConfig)
    target: TargetConfig    = field(default_factory=TargetConfig)
    optim:  OptimConfig     = field(default_factory=OptimConfig)

@dataclass
class Config:
    mpi_ranks: int = 1
    device: Optional[str] = None

    tmop: TMOPConfig = field(default_factory=TMOPConfig)

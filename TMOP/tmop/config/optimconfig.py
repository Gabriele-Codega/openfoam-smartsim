from dataclasses import dataclass, field
from typing import Type, Dict

import torch

@dataclass
class OptimConfig:
    """Configure the optimiser."""
    optimiser: Type[torch.optim.Optimizer] = torch.optim.Adam
    """Which optimiser should be used, specified as (subclass of) `torch.optim.Optimizer`."""
    optimiser_kwargs: Dict = field(default_factory=dict)
    """Optimiser keyword arguments."""
    scheduler: Type[torch.optim.lr_scheduler.LRScheduler] = torch.optim.lr_scheduler.ReduceLROnPlateau
    """Which learning rate scheduler should be used, specified as (subclass of) `torch.optim.lr_scheduler.LRScheduler`. This will monitor the mean loss if the optimiser is `TMOPOptimiser`, or the maximum loss if it is `WCUOptimiser`."""
    scheduler_kwargs: Dict = field(default_factory=dict)
    """Scheduler keyword arguments."""
    t_scheduler: Type[torch.optim.lr_scheduler.LRScheduler] = torch.optim.lr_scheduler.ReduceLROnPlateau
    """Scheduler to adapt the learning rate based on the improvement of `t` in the untangling optimiser. Specified as (subclass of) `torch.optim.lr_scheduler.LRScheduler`."""
    t_scheduler_kwargs: Dict = field(default_factory=lambda: {"mode": "max"})
    """Keyword arguments for `t_scheduler`."""
    max_steps: int = 1000
    """Maximum number of optimisation epochs per timestep."""
    stopping_threshold: float = 1e-3
    """Optimisation stops if the loss (or `beta`, i.e. maximum value of the loss for `WCUOptimiser`) falls below this value."""
    patience: int = 50
    """Maximum number of steps without improvement in the loss. After `patience` steps without improvement in the maximum value of the loss, optimisation stops."""
    rtol: float = 1e-2
    """Minimum accepted relative improvement in the loss. If the improvement is smaller than `rtol`, the counter of epochs since improvement is incremented and after `patience' epochs optimisation stops."""
